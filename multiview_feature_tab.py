"""PyQt6 tab: bake an eyebrows+lips feature layer from a diffuse multiview render set.

Qt glue only — the actual pipeline lives in multiview_feature_bake.py (no Qt dependency
there, so it stays independently scriptable/testable).
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matte_luminance_blend import load_rgb
from matte_luminance_ui import ImageViewer, _path_row
from multiview_feature_bake import bake_feature_layer, discover_character_assets

VIEW_LABELS = ("front", "left", "right")
DEFAULT_ANGLES = {"front": 0.0, "left": -45.0, "right": 45.0}


def _angle_to_token(angle: float) -> str:
    """"y_<angle>[_neg]" — the view-token form multiview_gen's solver requires."""
    magnitude = abs(round(angle))
    token = f"y_{int(magnitude)}"
    if angle < 0:
        token += "_neg"
    return token


class MultiviewFeatureWorker(QThread):
    finished_ok = pyqtSignal(int, dict)
    failed = pyqtSignal(int, str)

    def __init__(self, job_id: int, params: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.job_id = job_id
        self.params = params

    def run(self) -> None:
        try:
            p = self.params
            view_files = {
                _angle_to_token(p["angles"][label]): (p["images"][label], p["landmarks"][label])
                for label in VIEW_LABELS
            }
            output_path = bake_feature_layer(
                view_files=view_files,
                template_landmarks_usd=p["template_landmarks_usd"],
                landmarks_variant=p["landmarks_variant"],
                glb_path=p["glb_path"],
                output_dir=p["output_dir"],
                output_image_name=p["output_image_name"],
                output_size=p["output_size"],
                feather_px=p["feather_px"],
            )
            result: Dict[str, Any] = {"output_path": output_path, "preview": load_rgb(output_path)}
            self.finished_ok.emit(self.job_id, result)
        except Exception:
            self.failed.emit(self.job_id, traceback.format_exc())


class MultiviewFeatureTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._root = os.path.dirname(os.path.abspath(__file__))
        self._worker: Optional[MultiviewFeatureWorker] = None
        self._job_id = 0

        root_layout = QHBoxLayout(self)

        # -- Controls column ---------------------------------------------------
        controls = QWidget()
        controls.setMinimumWidth(420)
        controls.setMaximumWidth(520)
        cl = QVBoxLayout(controls)

        # Character folder + discovery
        char_box = QGroupBox("Character assets")
        cf = QFormLayout(char_box)
        default_char_dir = os.path.join(self._root, "test_textures", "african_female_0003")
        self.char_folder = _path_row(
            cf, "Character folder", default_char_dir, root=self._root, parent=self, is_dir=True
        )
        discover_btn = QPushButton("Discover")
        discover_btn.clicked.connect(self._on_discover)
        cf.addRow(discover_btn)
        cl.addWidget(char_box)

        # Per-view image / landmark / angle rows
        self.image_edits: Dict[str, Any] = {}
        self.landmark_edits: Dict[str, Any] = {}
        self.angle_spins: Dict[str, QDoubleSpinBox] = {}
        for label in VIEW_LABELS:
            view_box = QGroupBox(label.capitalize())
            vf = QFormLayout(view_box)
            self.image_edits[label] = _path_row(vf, "Diffuse image", "", root=self._root, parent=self)
            self.landmark_edits[label] = _path_row(
                vf, "Landmarks (.json)", "", root=self._root, parent=self,
                file_filter="Landmarks (*.json);;All (*.*)",
            )
            spin = QDoubleSpinBox()
            spin.setRange(-180.0, 180.0)
            spin.setDecimals(1)
            spin.setSuffix(" deg")
            spin.setValue(DEFAULT_ANGLES[label])
            self.angle_spins[label] = spin
            vf.addRow("Camera angle", spin)
            cl.addWidget(view_box)

        # Template landmarks + head mesh
        tmpl_box = QGroupBox("Template")
        tf = QFormLayout(tmpl_box)
        default_landmarks_usd = os.path.join(self._root, "test_textures", "analysis", "landmarks.usd")
        self.landmarks_usd_edit = _path_row(
            tf, "Landmarks (.usd)", default_landmarks_usd, root=self._root, parent=self,
            file_filter="USD (*.usd *.usda *.usdc *.usdz);;All (*.*)",
        )
        self.landmarks_variant = QComboBox()
        self.landmarks_variant.addItems(["coco_extended", "coco"])
        tf.addRow("Variant", self.landmarks_variant)
        self.glb_edit = _path_row(
            tf, "Head mesh (.glb)", "", root=self._root, parent=self,
            file_filter="glTF (*.glb *.gltf);;All (*.*)",
        )
        cl.addWidget(tmpl_box)

        # Output
        out_box = QGroupBox("Output")
        of = QFormLayout(out_box)
        default_output = os.path.join(self._root, "output", "eyebrow_lip_mask.png")
        self.output_edit = _path_row(of, "Output PNG", default_output, root=self._root, parent=self, save=True)
        cl.addWidget(out_box)

        self.run_btn = QPushButton("Run")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._on_run)
        cl.addWidget(self.run_btn)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        cl.addWidget(self.status)
        cl.addStretch(1)

        root_layout.addWidget(controls)

        # -- Preview -------------------------------------------------------
        self.viewer = ImageViewer()
        root_layout.addWidget(self.viewer, stretch=1)

    # -- discovery ------------------------------------------------------------
    def _on_discover(self) -> None:
        folder = self.char_folder.text().strip()
        found = discover_character_assets(folder)
        for label in VIEW_LABELS:
            img = found.get(f"{label}_image")
            lm = found.get(f"{label}_landmarks")
            if img:
                self.image_edits[label].setText(img)
            if lm:
                self.landmark_edits[label].setText(lm)
        if found.get("head_glb"):
            self.glb_edit.setText(found["head_glb"])

        missing = [k for k, v in found.items() if v is None]
        if missing:
            self.status.setText(f"Discovered available assets. Not found: {', '.join(missing)}")
        else:
            self.status.setText("Discovered all character assets.")

    # -- run --------------------------------------------------------------
    def _gather_params(self) -> dict:
        images, landmarks, angles = {}, {}, {}
        for label in VIEW_LABELS:
            img = self.image_edits[label].text().strip()
            lm = self.landmark_edits[label].text().strip()
            if not img or not os.path.isfile(img):
                raise FileNotFoundError(f"{label}: select a valid diffuse image.")
            if not lm or not os.path.isfile(lm):
                raise FileNotFoundError(f"{label}: select a valid landmarks .json file.")
            images[label] = img
            landmarks[label] = lm
            angles[label] = self.angle_spins[label].value()

        landmarks_usd = self.landmarks_usd_edit.text().strip()
        if not landmarks_usd or not os.path.isfile(landmarks_usd):
            raise FileNotFoundError("Select a valid template landmarks (.usd) file.")

        glb_path = self.glb_edit.text().strip()
        if not glb_path or not os.path.isfile(glb_path):
            raise FileNotFoundError("Select a valid head mesh (.glb) file.")

        output_path = self.output_edit.text().strip()
        if not output_path:
            raise FileNotFoundError("Select an output PNG path.")

        return dict(
            images=images,
            landmarks=landmarks,
            angles=angles,
            template_landmarks_usd=landmarks_usd,
            landmarks_variant=self.landmarks_variant.currentText(),
            glb_path=glb_path,
            output_dir=os.path.dirname(output_path) or ".",
            output_image_name=os.path.basename(output_path),
            output_size=1024,
            feather_px=6,
        )

    def _on_run(self) -> None:
        try:
            params = self._gather_params()
        except Exception as exc:
            QMessageBox.warning(self, "Inputs", str(exc))
            return

        self._job_id += 1
        job_id = self._job_id
        self.run_btn.setEnabled(False)
        self.status.setText("Baking…")

        self._worker = MultiviewFeatureWorker(job_id, params, self)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, job_id: int, result: dict) -> None:
        if job_id != self._job_id:
            return
        self.run_btn.setEnabled(True)
        self.viewer.set_image(result["preview"])
        self.status.setText(f"Done. Saved to {result['output_path']}")

    def _on_fail(self, job_id: int, tb: str) -> None:
        if job_id != self._job_id:
            return
        self.run_btn.setEnabled(True)
        self.status.setText("Failed — see dialog.")
        QMessageBox.critical(self, "Bake failed", tb)
