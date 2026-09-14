"""PyQt6 tab: segment a region (starting with the lips) out of a UV-space texture.

Because the UV layout is fixed for a given character topology, one region mask can be
reused against any texture sharing that topology. This tab has two independent steps:

  1. Feature mask: detect lips and/or eyebrows directly on a UV texture's face island via
     texture_face_segment.bake_feature_mask (a real face-landmark model run straight on the
     2D texture — no multiview render set or 3D landmark transfer needed). Skippable if you
     already have a mask (e.g. generated on another platform).
  2. Segment texture: apply any UV-space mask to any UV-space texture, producing an
     RGBA PNG with the texture's RGB untouched and alpha set from the mask.

Qt glue only — the actual pipeline logic lives in texture_face_segment.py and
texture_segment.py (no Qt dependency there, so it stays independently scriptable/testable).
"""
from __future__ import annotations

import os
import traceback
from typing import List, Optional, Tuple

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matte_luminance_blend import load_rgb, save_rgb
from matte_luminance_ui import BatchBar, ImageViewer, list_images, _path_row
from texture_face_segment import bake_feature_mask
from texture_segment import segment_texture

# Combo label -> the feature group name(s) passed to bake_feature_mask.
FEATURE_CHOICES = {
    "Lips": ("lips",),
    "Eyebrows": ("left_eyebrow", "right_eyebrow"),
    "Lips + eyebrows": ("lips", "left_eyebrow", "right_eyebrow"),
}


# ---------------------------------------------------------------------------
# Step 1: detect a face feature directly on a UV texture and bake a mask
# ---------------------------------------------------------------------------
class FeatureMaskWorker(QThread):
    finished_ok = pyqtSignal(int, dict)
    failed = pyqtSignal(int, str)

    def __init__(self, job_id: int, params: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.job_id = job_id
        self.params = params

    def run(self) -> None:
        try:
            p = self.params
            texture = load_rgb(p["texture_path"])
            mask = bake_feature_mask(texture, features=p["features"], feather_px=p["feather_px"])
            save_rgb(p["output_path"], mask)
            result = {"output_path": p["output_path"], "preview": mask}
            self.finished_ok.emit(self.job_id, result)
        except Exception:
            self.failed.emit(self.job_id, traceback.format_exc())


class FeatureMaskPanel(QGroupBox):
    """Step 1: detect lips/eyebrows on a UV texture's face island. Skip if you already have a mask."""

    maskBaked = pyqtSignal(str)

    def __init__(self, root: str, parent: Optional[QWidget] = None) -> None:
        super().__init__("1. Detect feature mask (from texture)", parent)
        self._root = root
        self._worker: Optional[FeatureMaskWorker] = None
        self._job_id = 0

        form = QFormLayout(self)
        default_texture = os.path.join(
            self._root, "test_textures", "african_female_0003_albedo_from_concept.png"
        )
        self.texture_edit = _path_row(form, "Source texture", default_texture, root=self._root, parent=self)

        self.feature_combo = QComboBox()
        self.feature_combo.addItems(list(FEATURE_CHOICES))
        form.addRow("Feature", self.feature_combo)

        self.feather = QSpinBox()
        self.feather.setRange(0, 40)
        self.feather.setValue(6)
        form.addRow("Feather (px)", self.feather)

        default_output = os.path.join(self._root, "masks", "lips_mask.png")
        self.output_edit = _path_row(form, "Mask PNG", default_output, root=self._root, parent=self, save=True)

        self.run_btn = QPushButton("Detect mask")
        self.run_btn.setMinimumHeight(32)
        self.run_btn.clicked.connect(self._on_run)
        form.addRow(self.run_btn)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        form.addRow(self.status)

    def _on_run(self) -> None:
        texture_path = self.texture_edit.text().strip()
        output_path = self.output_edit.text().strip()
        if not texture_path or not os.path.isfile(texture_path):
            QMessageBox.warning(self, "Inputs", "Select a valid source texture.")
            return
        if not output_path:
            QMessageBox.warning(self, "Inputs", "Select an output PNG path.")
            return

        self._job_id += 1
        job_id = self._job_id
        self.run_btn.setEnabled(False)
        self.status.setText("Detecting…")

        params = dict(
            texture_path=texture_path,
            output_path=output_path,
            feather_px=self.feather.value(),
            features=FEATURE_CHOICES[self.feature_combo.currentText()],
        )
        self._worker = FeatureMaskWorker(job_id, params, self)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, job_id: int, result: dict) -> None:
        if job_id != self._job_id:
            return
        self.run_btn.setEnabled(True)
        self.status.setText(f"Done. Saved to {result['output_path']}")
        self.maskBaked.emit(result["output_path"])

    def _on_fail(self, job_id: int, tb: str) -> None:
        if job_id != self._job_id:
            return
        self.run_btn.setEnabled(True)
        self.status.setText("Failed — see dialog.")
        QMessageBox.critical(self, "Detect failed", tb)


# ---------------------------------------------------------------------------
# Step 2: apply a UV-space mask to any UV-space texture
# ---------------------------------------------------------------------------
class BatchSegmentWorker(QThread):
    """Applies one fixed region mask to many textures. Mirrors matte_luminance_ui.BatchWorker's
    per-file try/except + progress-signal shape so both tabs' batch behavior stays consistent.
    """

    fileDone = pyqtSignal(int, int, str)  # index, total, filename
    fileFailed = pyqtSignal(int, int, str, str)  # index, total, filename, error
    batchFinished = pyqtSignal(int, int, list)  # ok_count, total, failures[(filename, error)]

    def __init__(
        self,
        input_paths: List[str],
        mask_path: str,
        output_dir: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.input_paths = input_paths
        self.mask_path = mask_path
        self.output_dir = output_dir

    def run(self) -> None:
        total = len(self.input_paths)
        ok_count = 0
        failures: List[Tuple[str, str]] = []
        for i, path in enumerate(self.input_paths, start=1):
            name = os.path.basename(path)
            try:
                stem = os.path.splitext(name)[0]
                output_path = os.path.join(self.output_dir, f"{stem}.png")
                segment_texture(path, self.mask_path, output_path)
                ok_count += 1
                self.fileDone.emit(i, total, name)
            except Exception:
                err = traceback.format_exc()
                failures.append((name, err))
                print(f"[batch] failed on {name}:\n{err}")
                self.fileFailed.emit(i, total, name, err)
        self.batchFinished.emit(ok_count, total, failures)


class SegmentPanel(QGroupBox):
    """Step 2: extract whatever region a mask marks out of any texture sharing its UV layout."""

    segmented = pyqtSignal(str)

    def __init__(self, root: str, parent: Optional[QWidget] = None) -> None:
        super().__init__("2. Segment texture", parent)
        self._root = root
        self._batch_worker: Optional[BatchSegmentWorker] = None

        form = QFormLayout(self)
        self.texture_edit = _path_row(form, "Source texture", "", root=self._root, parent=self)
        default_mask = os.path.join(self._root, "masks", "lips_mask.png")
        self.mask_edit = _path_row(form, "Region mask", default_mask, root=self._root, parent=self)
        default_output = os.path.join(self._root, "output", "lips_segment.png")
        self.output_edit = _path_row(
            form, "Output (RGBA PNG)", default_output, root=self._root, parent=self, save=True
        )

        self.run_btn = QPushButton("Extract region")
        self.run_btn.setMinimumHeight(32)
        self.run_btn.clicked.connect(self._on_run)
        form.addRow(self.run_btn)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        form.addRow(self.status)

        self.batch_bar = BatchBar(self._root, "Run batch")
        self.batch_bar.runRequested.connect(self._on_run_batch)
        form.addRow(self.batch_bar)

    def set_mask_path(self, path: str) -> None:
        self.mask_edit.setText(path)

    def _on_run(self) -> None:
        texture = self.texture_edit.text().strip()
        mask = self.mask_edit.text().strip()
        output = self.output_edit.text().strip()
        if not texture or not os.path.isfile(texture):
            QMessageBox.warning(self, "Inputs", "Select a valid source texture.")
            return
        if not mask or not os.path.isfile(mask):
            QMessageBox.warning(self, "Inputs", "Select a valid region mask (bake one above, or pick an existing file).")
            return
        if not output:
            QMessageBox.warning(self, "Inputs", "Select an output PNG path.")
            return

        try:
            output_path = segment_texture(texture, mask, output)
        except Exception:
            QMessageBox.critical(self, "Segment failed", traceback.format_exc())
            self.status.setText("Failed — see dialog.")
            return

        self.status.setText(f"Done. Saved to {output_path}")
        self.segmented.emit(output_path)

    # -- batch: same mask, every texture in a folder -------------------------
    def _on_run_batch(self) -> None:
        mask = self.mask_edit.text().strip()
        input_dir = self.batch_bar.input_dir()
        output_dir = self.batch_bar.output_dir()
        if not mask or not os.path.isfile(mask):
            QMessageBox.warning(self, "Batch", "Select a valid region mask (bake one above, or pick an existing file).")
            return
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Batch", "Select a valid input folder.")
            return
        if not output_dir:
            QMessageBox.warning(self, "Batch", "Select an output folder.")
            return

        input_paths = list_images(input_dir)
        if not input_paths:
            QMessageBox.warning(self, "Batch", f"No images found in: {input_dir}")
            return

        os.makedirs(output_dir, exist_ok=True)
        self.batch_bar.set_running(True)
        self.run_btn.setEnabled(False)
        self.batch_bar.set_status(f"Processing 0/{len(input_paths)}…")

        self._batch_worker = BatchSegmentWorker(input_paths, mask, output_dir, self)
        self._batch_worker.fileDone.connect(self._on_batch_file_done)
        self._batch_worker.fileFailed.connect(self._on_batch_file_failed)
        self._batch_worker.batchFinished.connect(self._on_batch_finished)
        self._batch_worker.start()

    def _on_batch_file_done(self, index: int, total: int, name: str) -> None:
        self.batch_bar.set_status(f"Processing {index}/{total}: {name}")

    def _on_batch_file_failed(self, index: int, total: int, name: str, _error: str) -> None:
        self.batch_bar.set_status(f"Processing {index}/{total}: {name} — failed, see console")

    def _on_batch_finished(self, ok_count: int, total: int, failures: List[Tuple[str, str]]) -> None:
        self.batch_bar.set_running(False)
        self.run_btn.setEnabled(True)
        if failures:
            names = ", ".join(name for name, _ in failures)
            self.batch_bar.set_status(f"Batch done: {ok_count}/{total} saved. Failed: {names} (see console).")
        else:
            self.batch_bar.set_status(f"Batch done: {ok_count}/{total} saved.")


class TextureSegmentationTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = os.path.dirname(os.path.abspath(__file__))

        root_layout = QHBoxLayout(self)

        controls = QWidget()
        controls.setMinimumWidth(420)
        controls.setMaximumWidth(520)
        cl = QVBoxLayout(controls)

        self.feature_mask_panel = FeatureMaskPanel(root)
        self.segment_panel = SegmentPanel(root)
        cl.addWidget(self.feature_mask_panel)
        cl.addWidget(self.segment_panel)
        cl.addStretch(1)
        root_layout.addWidget(controls)

        self.tabs = QTabWidget()
        self.result_viewer = ImageViewer()
        self.mask_viewer = ImageViewer()
        self.tabs.addTab(self.result_viewer, "Segmented texture")
        self.tabs.addTab(self.mask_viewer, "Mask")
        root_layout.addWidget(self.tabs, stretch=1)

        self.feature_mask_panel.maskBaked.connect(self._on_mask_baked)
        self.segment_panel.segmented.connect(self._on_segmented)

    def _on_mask_baked(self, path: str) -> None:
        self.segment_panel.set_mask_path(path)
        self.mask_viewer.set_image(load_rgb(path))
        self.tabs.setCurrentWidget(self.mask_viewer)

    def _on_segmented(self, path: str) -> None:
        rgba = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        self.result_viewer.set_image(rgba)
        self.tabs.setCurrentWidget(self.result_viewer)
