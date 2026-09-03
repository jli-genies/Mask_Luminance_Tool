"""Simple PyQt6 UI for per-mask-channel luminance blend controls.

Run::

    python matte_luminance_ui.py
"""

from __future__ import annotations

import glob
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matte_luminance_blend import (
    DEFAULT_REGION_PALETTE,
    DEFAULT_SELF_LOCALITY_RADIUS,
    GATE_MODES,
    MaskChannel,
    build_local_diffuse_target,
    compute_channel_gate,
    composite_weights,
    load_rgb,
    make_diffuse_target,
    resize_to,
    run_channel_pipeline,
    save_rgb,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert uint8 RGB (or gray) numpy array to a full-resolution QPixmap."""
    arr = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    else:
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class _SliderSpin(QWidget):
    """Linked slider + spin box for one float parameter."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float = 0.1,
        decimals: int = 2,
        slider_scale: int = 100,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._scale = slider_scale

        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setValue(value)
        self.spin.setMinimumWidth(90)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(minimum * slider_scale), int(maximum * slider_scale))
        self.slider.setValue(int(round(value * slider_scale)))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, iv: int) -> None:
        fv = iv / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(fv)
        self.spin.blockSignals(False)
        self.valueChanged.emit(fv)

    def _on_spin(self, fv: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(fv * self._scale)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(fv)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float) -> None:
        self.spin.setValue(v)


# ---------------------------------------------------------------------------
# Image viewer: fit-to-panel by default, zoom + scroll when needed
# ---------------------------------------------------------------------------
class ImageViewer(QWidget):
    """Scrollable image view that fits the panel and supports zoom."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source: Optional[QPixmap] = None
        self._fit = True
        self._scale = 1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        self.btn_fit = QPushButton("Fit")
        self.btn_100 = QPushButton("100%")
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_in = QPushButton("+")
        for b in (self.btn_fit, self.btn_100, self.btn_zoom_out, self.btn_zoom_in):
            b.setFixedHeight(26)
            b.setMaximumWidth(56)
        self.zoom_label = QLabel("—")
        self.zoom_label.setMinimumWidth(56)
        bar.addWidget(self.btn_fit)
        bar.addWidget(self.btn_100)
        bar.addWidget(self.btn_zoom_out)
        bar.addWidget(self.btn_zoom_in)
        bar.addWidget(self.zoom_label)
        bar.addStretch(1)
        root.addLayout(bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QFrame.Shape.StyledPanel)
        self.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.label = QLabel("No image")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(64, 64)
        self.scroll.setWidget(self.label)
        root.addWidget(self.scroll, stretch=1)

        self.btn_fit.clicked.connect(self.fit_to_view)
        self.btn_100.clicked.connect(self.zoom_100)
        self.btn_zoom_in.clicked.connect(lambda: self.zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_by(0.8))

    def set_image(self, rgb: np.ndarray) -> None:
        self._source = _rgb_to_qpixmap(rgb)
        if self._fit:
            self.fit_to_view()
        else:
            self._apply_scale()

    def clear(self) -> None:
        self._source = None
        self.label.clear()
        self.label.setText("No image")
        self.label.setMinimumSize(64, 64)
        self.label.resize(200, 120)
        self.zoom_label.setText("—")

    def fit_to_view(self) -> None:
        self._fit = True
        if self._source is None or self._source.isNull():
            return
        vp = self.scroll.viewport().size()
        # Leave a little margin so scrollbars don't fight the fit size.
        avail_w = max(32, vp.width() - 4)
        avail_h = max(32, vp.height() - 4)
        fitted = self._source.scaled(
            avail_w,
            avail_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._scale = fitted.width() / max(1, self._source.width())
        self._show(fitted)

    def zoom_100(self) -> None:
        self._fit = False
        self._scale = 1.0
        self._apply_scale()

    def zoom_by(self, factor: float) -> None:
        if self._source is None:
            return
        self._fit = False
        self._scale = float(np.clip(self._scale * factor, 0.05, 8.0))
        self._apply_scale()

    def _apply_scale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        if abs(self._scale - 1.0) < 1e-6:
            self._show(self._source)
            return
        w = max(1, int(round(self._source.width() * self._scale)))
        h = max(1, int(round(self._source.height() * self._scale)))
        scaled = self._source.scaled(
            w,
            h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._show(scaled)

    def _show(self, pix: QPixmap) -> None:
        self.label.setText("")
        self.label.setPixmap(pix)
        self.label.setFixedSize(pix.size())
        self.zoom_label.setText(f"{100.0 * self._scale:.0f}%")

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit and self._source is not None:
            # Defer one tick so viewport size is settled.
            QTimer.singleShot(0, self.fit_to_view)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._source is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_by(1.15)
            elif delta < 0:
                self.zoom_by(1 / 1.15)
            event.accept()
            return
        super().wheelEvent(event)


# ---------------------------------------------------------------------------
# One control panel per mask channel
# ---------------------------------------------------------------------------
class ChannelPanel(QGroupBox):
    """Editable controls for one MaskChannel: active toggle + its own params."""

    changed = pyqtSignal()
    removeRequested = pyqtSignal(object)

    def __init__(self, name: str, mask_path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(name, parent)
        self.setCheckable(True)
        self.setChecked(False)

        form = QFormLayout(self)

        mask_row = QWidget()
        mrl = QHBoxLayout(mask_row)
        mrl.setContentsMargins(0, 0, 0, 0)
        self.mask_edit = QLineEdit(mask_path)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._browse)
        mrl.addWidget(self.mask_edit)
        mrl.addWidget(browse_btn)
        form.addRow("Mask file", mask_row)

        self.gate_mode = QComboBox()
        self.gate_mode.addItems(list(GATE_MODES))
        # "weight" (plain grayscale) covers every mask observed so far, including
        # the "highlight" ones — they're pre-authored gradients, not color-coded
        # paint. "blue_paint"/"color_id" remain available as a manual override.
        self.gate_mode.setCurrentText("weight")
        form.addRow("Gate mode", self.gate_mode)

        self.fill_holes = QCheckBox("Fill enclosed holes (envelope)")
        form.addRow(self.fill_holes)

        self.region_row = QWidget()
        rrl = QHBoxLayout(self.region_row)
        rrl.setContentsMargins(0, 0, 0, 0)
        self.region_checks: Dict[str, QCheckBox] = {}
        for rname in DEFAULT_REGION_PALETTE:
            cb = QCheckBox(rname)
            cb.setChecked(True)
            self.region_checks[rname] = cb
            rrl.addWidget(cb)
        form.addRow("Regions", self.region_row)
        self.region_tolerance = _SliderSpin(0.0, 120.0, 40.0, step=1.0, decimals=0, slider_scale=1)
        form.addRow("Region tolerance", self.region_tolerance)

        self.threshold = _SliderSpin(0.0, 80.0, 12.0, step=0.5, decimals=1, slider_scale=10)
        self.radius = _SliderSpin(0.0, 64.0, 8.0, step=0.5, decimals=1, slider_scale=10)
        self.strength = _SliderSpin(0.0, 1.0, 0.85, step=0.01, decimals=2, slider_scale=100)
        self.diffuse_mix = _SliderSpin(0.0, 1.0, 0.0, step=0.01, decimals=2, slider_scale=100)
        form.addRow("Threshold (dL)", self.threshold)
        form.addRow("Blur radius (px)", self.radius)
        form.addRow("Diffuse strength", self.strength)
        form.addRow("Diffuse mix", self.diffuse_mix)

        self.use_infill = QCheckBox("Use in_fill (core algorithm)")
        self.use_infill.setChecked(True)
        self.spill_outside = QCheckBox("Allow blur to spill outside mask")
        form.addRow(self.use_infill)
        form.addRow(self.spill_outside)

        self.flat_fill = QCheckBox("Flat fill (mean skin color instead of infill/blur)")
        form.addRow(self.flat_fill)

        self.blend_group = QLineEdit()
        self.blend_group.setPlaceholderText("optional — e.g. skin_uniform")
        form.addRow("Blend group", self.blend_group)
        self.blend_weight = _SliderSpin(0.0, 5.0, 1.0, step=0.05, decimals=2, slider_scale=100)
        form.addRow("Blend weight (within group)", self.blend_weight)

        remove_btn = QPushButton("Remove channel")
        remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))
        form.addRow(remove_btn)

        self.gate_mode.currentTextChanged.connect(self._on_gate_mode_changed)
        for w in (self.threshold, self.radius, self.strength, self.diffuse_mix, self.region_tolerance, self.blend_weight):
            w.valueChanged.connect(lambda *_: self.changed.emit())
        for cb in (self.use_infill, self.spill_outside, self.fill_holes, self.flat_fill, *self.region_checks.values()):
            cb.toggled.connect(lambda *_: self.changed.emit())
        self.flat_fill.toggled.connect(self._on_flat_fill_toggled)
        self.mask_edit.editingFinished.connect(self.changed.emit)
        self.blend_group.editingFinished.connect(self.changed.emit)
        self.toggled.connect(lambda *_: self.changed.emit())

        self._on_gate_mode_changed(self.gate_mode.currentText())
        self._on_flat_fill_toggled(self.flat_fill.isChecked())

    def _on_gate_mode_changed(self, mode: str) -> None:
        self.fill_holes.setVisible(mode == "weight")
        self.region_row.setVisible(mode == "color_id")
        self.region_tolerance.setVisible(mode == "color_id")
        self.changed.emit()

    def _on_flat_fill_toggled(self, on: bool) -> None:
        # All three are ignored by MaskChannel.flat_fill — the flat
        # mean-color target always fully replaces the pixel (same as
        # diffuse_mix=1), and feathering is always outward-only regardless
        # of spill_outside — see feather_mask_outward().
        self.diffuse_mix.setVisible(not on)
        self.use_infill.setVisible(not on)
        self.spill_outside.setVisible(not on)
        self.changed.emit()

    def _browse(self) -> None:
        start = self.mask_edit.text().strip() or "."
        path, _ = QFileDialog.getOpenFileName(
            self, "Open mask", start, "Images (*.png *.jpg *.jpeg *.tif *.bmp);;All (*.*)"
        )
        if path:
            self.mask_edit.setText(path)
            self.changed.emit()

    def mask_path(self) -> str:
        return self.mask_edit.text().strip()

    def to_channel(self) -> MaskChannel:
        mode = self.gate_mode.currentText()
        regions = None
        if mode == "color_id":
            regions = [n for n, cb in self.region_checks.items() if cb.isChecked()] or None
        return MaskChannel(
            name=self.title(),
            mask_path=self.mask_path(),
            enabled=self.isChecked(),
            gate_mode=mode,
            threshold=self.threshold.value(),
            radius=self.radius.value(),
            strength=self.strength.value(),
            diffuse_mix=self.diffuse_mix.value(),
            use_infill=self.use_infill.isChecked(),
            spill_outside=self.spill_outside.isChecked(),
            fill_holes=self.fill_holes.isChecked(),
            regions=regions,
            region_tolerance=int(self.region_tolerance.value()),
            blend_group=self.blend_group.text().strip() or None,
            blend_weight=self.blend_weight.value(),
            flat_fill=self.flat_fill.isChecked(),
        )


class ProcessWorker(QThread):
    finished_ok = pyqtSignal(int, dict)  # job_id, result
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        job_id: int,
        params: dict,
        write_outputs: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.job_id = job_id
        self.params = params
        self.write_outputs = write_outputs

    def run(self) -> None:
        try:
            p = self.params
            sample: np.ndarray = p["sample"]
            channels: List[MaskChannel] = p["channels"]
            active = [ch for ch in channels if ch.enabled]

            mask_imgs: Dict[str, np.ndarray] = {}
            for ch in active:
                nearest = ch.gate_mode in ("blue_paint", "color_id")
                mask_imgs[ch.name] = resize_to(load_rgb(ch.mask_path), sample.shape[:2], nearest=nearest)

            feature_preserve = None
            if p["feature_preserve_path"]:
                feature_preserve = composite_weights(
                    resize_to(load_rgb(p["feature_preserve_path"]), sample.shape[:2], nearest=False)
                )

            palette = DEFAULT_REGION_PALETTE
            if p["diffuse_mode"] == "self":
                exclude = np.zeros(sample.shape[:2], dtype=np.float32)
                for ch in active:
                    exclude = np.maximum(exclude, compute_channel_gate(mask_imgs[ch.name], ch, palette))
                if feature_preserve is not None:
                    exclude = np.maximum(exclude, feature_preserve)
                diffuse_target = build_local_diffuse_target(sample, exclude, p["self_locality_radius"])
            else:
                diffuse_img = load_rgb(p["diffuse_path"])
                diffuse_target = make_diffuse_target(sample, diffuse_img, p["diffuse_mode"])

            working, soft_masks = run_channel_pipeline(
                sample, diffuse_target, mask_imgs, active, palette, p["luminance_only"], feature_preserve,
            )
            channel_masks = {
                name: np.clip(soft * 255.0, 0, 255).astype(np.uint8) for name, soft in soft_masks.items()
            }

            result: Dict[str, Any] = {
                "texture": working,
                "sample": sample,
                "channel_masks": channel_masks,
            }

            if self.write_outputs:
                out_tex = p["out_texture_path"]
                out_dir = p["out_masks_dir"]
                os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)
                save_rgb(out_tex, working)
                paths = {"texture": out_tex}
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    for name, mask_u8 in channel_masks.items():
                        mp = os.path.join(out_dir, f"{name}_mask.png")
                        save_rgb(mp, mask_u8)
                        paths[f"mask:{name}"] = mp
                result["paths"] = paths

            self.finished_ok.emit(self.job_id, result)
        except Exception:
            self.failed.emit(self.job_id, traceback.format_exc())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MatteBlendWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Matte Luminance Blend")
        self.resize(1450, 950)
        self._worker: Optional[ProcessWorker] = None
        self._job_id = 0
        self._pending_run: Optional[Tuple[dict, bool]] = None
        self._sample_cache: Optional[Tuple[str, np.ndarray]] = None
        self._root = os.path.dirname(os.path.abspath(__file__))
        self._channel_panels: List[ChannelPanel] = []

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._run_live_preview)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # --- Scrollable controls column -------------------------------------
        controls_host = QWidget()
        controls_host.setMinimumWidth(380)
        controls_host.setMaximumWidth(480)
        ch_layout = QVBoxLayout(controls_host)
        ch_layout.setContentsMargins(0, 0, 0, 0)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_inner = QWidget()
        self._controls_layout = QVBoxLayout(controls_inner)
        self._controls_layout.setContentsMargins(4, 4, 8, 4)
        controls_scroll.setWidget(controls_inner)
        ch_layout.addWidget(controls_scroll)
        splitter.addWidget(controls_host)

        # --- Inputs ---------------------------------------------------------
        io_box = QGroupBox("Inputs")
        io = QFormLayout(io_box)
        self.texture_edit = self._path_row(io, "Texture (albedo)", "", invalidate=True)
        self.diffuse_edit = self._path_row(io, "Diffuse", "", invalidate=True)
        self.feature_edit = self._path_row(io, "Feature preserve mask", "", invalidate=True)

        self.diffuse_mode = QComboBox()
        self.diffuse_mode.addItems(["self", "uv", "palette"])
        self.diffuse_mode.currentIndexChanged.connect(self._on_inputs_changed)
        io.addRow("Diffuse mode", self.diffuse_mode)

        self.self_locality_radius = _SliderSpin(
            8.0, 800.0, DEFAULT_SELF_LOCALITY_RADIUS, step=2.0, decimals=0, slider_scale=1
        )
        io.addRow("Self-mode locality radius (px)", self.self_locality_radius)
        self.self_locality_radius.valueChanged.connect(self._schedule_live_preview)
        self._controls_layout.addWidget(io_box)

        # --- Mask channels ----------------------------------------------------
        self.channels_box = QGroupBox("Mask channels")
        self.channels_layout = QVBoxLayout(self.channels_box)
        add_btn = QPushButton("Add mask…")
        add_btn.clicked.connect(self._add_channel_dialog)
        self.channels_layout.addWidget(add_btn)
        self._controls_layout.addWidget(self.channels_box)

        # --- Shared options ---------------------------------------------------
        shared_box = QGroupBox("Shared options")
        sf = QFormLayout(shared_box)
        self.chk_live = QCheckBox("Live preview (update on change)")
        self.chk_live.setChecked(True)
        self.chk_luma_only = QCheckBox("Luminance-only blend (keep chroma)")
        self.chk_luma_only.setChecked(True)
        sf.addRow(self.chk_live)
        sf.addRow(self.chk_luma_only)
        self._controls_layout.addWidget(shared_box)

        # --- Outputs --------------------------------------------------------
        out_box = QGroupBox("Outputs")
        of = QFormLayout(out_box)
        out_dir = os.path.join(self._root, "output")
        self.out_texture = self._path_row(
            of, "Corrected texture", os.path.join(out_dir, "albedo_matte.png"), save=True
        )
        self.out_masks_dir = self._path_row(
            of, "Channel masks dir (debug)", os.path.join(out_dir, "channel_masks"), save=True, is_dir=True
        )
        self._controls_layout.addWidget(out_box)

        self.run_btn = QPushButton("Process & Save")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._on_process)
        self._controls_layout.addWidget(self.run_btn)

        self.status = QLabel("Ready — check a mask active for live preview, or Process & Save.")
        self.status.setWordWrap(True)
        self._controls_layout.addWidget(self.status)
        self._controls_layout.addStretch(1)

        # --- Preview --------------------------------------------------------
        preview_wrap = QWidget()
        pl = QVBoxLayout(preview_wrap)
        pl.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.viewers: Dict[str, ImageViewer] = {}
        for key, title in (("result", "Result"), ("sample", "Original")):
            viewer = ImageViewer()
            self.tabs.addTab(viewer, title)
            self.viewers[key] = viewer
        pl.addWidget(self.tabs)
        hint = QLabel("Tip: Fit shows the whole image. Ctrl+scroll or +/− to zoom; scrollbars appear when zoomed in.")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        pl.addWidget(hint)
        splitter.addWidget(preview_wrap)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, 1010])

        self._seed_default_paths()
        self._discover_channels()

    # -- channel discovery / management --------------------------------------
    def _discover_channels(self) -> None:
        masks_dir = os.path.join(self._root, "masks")
        if not os.path.isdir(masks_dir):
            return
        found: List[str] = []
        for ext in IMAGE_EXTS:
            found.extend(glob.glob(os.path.join(masks_dir, f"*{ext}")))
        for path in sorted(found):
            name = os.path.splitext(os.path.basename(path))[0]
            self._add_channel_panel(name, path)

    def _add_channel_dialog(self) -> None:
        start = os.path.join(self._root, "masks")
        path, _ = QFileDialog.getOpenFileName(
            self, "Add mask", start, "Images (*.png *.jpg *.jpeg *.tif *.bmp);;All (*.*)"
        )
        if path:
            name = os.path.splitext(os.path.basename(path))[0]
            self._add_channel_panel(name, path)

    def _add_channel_panel(self, name: str, mask_path: str) -> None:
        panel = ChannelPanel(name, mask_path)
        panel.changed.connect(self._schedule_live_preview)
        panel.removeRequested.connect(self._remove_channel_panel)
        # Insert above the "Add mask…" button, which is always the last item.
        self.channels_layout.insertWidget(self.channels_layout.count() - 1, panel)
        self._channel_panels.append(panel)

    def _remove_channel_panel(self, panel: ChannelPanel) -> None:
        self._channel_panels.remove(panel)
        self.channels_layout.removeWidget(panel)
        panel.deleteLater()
        self._schedule_live_preview()

    # -- live preview wiring -------------------------------------------------
    def _on_inputs_changed(self, *_args) -> None:
        self._sample_cache = None
        self._schedule_live_preview()

    def _schedule_live_preview(self, *_args) -> None:
        if not self.chk_live.isChecked():
            return
        self._debounce.start()

    def _run_live_preview(self) -> None:
        if not self.chk_live.isChecked():
            return
        try:
            params = self._gather_params()
        except FileNotFoundError:
            return
        except Exception as exc:
            self.status.setText(f"Preview skipped: {exc}")
            return
        self._start_job(params, write_outputs=False)

    # -- path helpers --------------------------------------------------------
    def _path_row(
        self,
        form: QFormLayout,
        label: str,
        default: str,
        save: bool = False,
        invalidate: bool = False,
        is_dir: bool = False,
    ) -> QLineEdit:
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(default)
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.clicked.connect(lambda: self._browse(edit, save=save, invalidate=invalidate, is_dir=is_dir))
        hl.addWidget(edit)
        hl.addWidget(btn)
        form.addRow(label, row)
        if invalidate:
            edit.editingFinished.connect(self._on_inputs_changed)
        return edit

    def _browse(self, edit: QLineEdit, save: bool = False, invalidate: bool = False, is_dir: bool = False) -> None:
        start = edit.text().strip() or self._root
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select directory", start)
        elif save:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save image", start, "Images (*.png *.jpg *.jpeg *.tif);;All (*.*)"
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open image", start, "Images (*.png *.jpg *.jpeg *.tif *.bmp);;All (*.*)"
            )
        if path:
            edit.setText(path)
            if invalidate:
                self._on_inputs_changed()

    def _seed_default_paths(self) -> None:
        pass

    # -- sample cache ---------------------------------------------------------
    def _ensure_sample(self, texture_path: str) -> np.ndarray:
        if self._sample_cache is not None and self._sample_cache[0] == texture_path:
            return self._sample_cache[1]
        sample = load_rgb(texture_path)
        self._sample_cache = (texture_path, sample)
        return sample

    # -- process -------------------------------------------------------------
    def _gather_params(self) -> dict:
        texture = self.texture_edit.text().strip()
        diffuse = self.diffuse_edit.text().strip()
        diffuse_mode = self.diffuse_mode.currentText()
        if not texture or not os.path.isfile(texture):
            raise FileNotFoundError("Select a valid texture (albedo) image.")
        if diffuse_mode != "self" and (not diffuse or not os.path.isfile(diffuse)):
            raise FileNotFoundError("Select a valid diffuse image (or switch diffuse mode to 'self').")

        feature = self.feature_edit.text().strip() or None
        if feature and not os.path.isfile(feature):
            raise FileNotFoundError(f"Feature preserve mask not found: {feature}")

        channels: List[MaskChannel] = []
        for panel in self._channel_panels:
            ch = panel.to_channel()
            if ch.enabled:
                if not ch.mask_path or not os.path.isfile(ch.mask_path):
                    raise FileNotFoundError(f"Mask file not found for channel '{ch.name}': {ch.mask_path}")
            channels.append(ch)

        sample = self._ensure_sample(texture)

        return dict(
            sample=sample,
            channels=channels,
            diffuse_path=diffuse,
            diffuse_mode=diffuse_mode,
            self_locality_radius=self.self_locality_radius.value(),
            feature_preserve_path=feature,
            luminance_only=self.chk_luma_only.isChecked(),
            out_texture_path=self.out_texture.text().strip(),
            out_masks_dir=self.out_masks_dir.text().strip() or None,
        )

    def _start_job(self, params: dict, write_outputs: bool) -> None:
        if self._worker and self._worker.isRunning():
            self._pending_run = (params, write_outputs)
            self.status.setText("Updating…")
            return

        self._job_id += 1
        job_id = self._job_id
        if write_outputs:
            self.run_btn.setEnabled(False)
            self.status.setText("Processing & saving…")
        else:
            self.status.setText("Updating preview…")

        self._worker = ProcessWorker(job_id, params, write_outputs, self)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_process(self) -> None:
        try:
            params = self._gather_params()
        except Exception as exc:
            QMessageBox.warning(self, "Inputs", str(exc))
            return
        self._start_job(params, write_outputs=True)

    def _rebuild_mask_tabs(self, channel_masks: Dict[str, np.ndarray]) -> None:
        while self.tabs.count() > 2:
            w = self.tabs.widget(2)
            self.tabs.removeTab(2)
            w.deleteLater()
        for name, mask_u8 in channel_masks.items():
            viewer = ImageViewer()
            viewer.set_image(mask_u8)
            self.tabs.addTab(viewer, name)

    def _on_done(self, job_id: int, result: dict) -> None:
        if job_id != self._job_id:
            return

        self.run_btn.setEnabled(True)
        self.viewers["result"].set_image(result["texture"])
        self.viewers["sample"].set_image(result["sample"])
        self._rebuild_mask_tabs(result["channel_masks"])

        if "paths" in result:
            paths = result["paths"]
            lines = [f"Saved.\nTexture → {paths['texture']}"]
            mask_paths = [v for k, v in paths.items() if k.startswith("mask:")]
            if mask_paths:
                lines.append(f"Channel masks → {os.path.dirname(mask_paths[0])}")
            self.status.setText("\n".join(lines))
            self.tabs.setCurrentIndex(0)
        else:
            self.status.setText("Live preview updated.")

        pending = self._pending_run
        self._pending_run = None
        if pending is not None:
            params, write_outputs = pending
            self._start_job(params, write_outputs)

    def _on_fail(self, job_id: int, tb: str) -> None:
        if job_id != self._job_id:
            return
        self.run_btn.setEnabled(True)
        self.status.setText("Failed — see dialog.")
        QMessageBox.critical(self, "Process failed", tb)
        pending = self._pending_run
        self._pending_run = None
        if pending is not None:
            params, write_outputs = pending
            self._start_job(params, write_outputs)


def main() -> None:
    app = QApplication(sys.argv)
    win = MatteBlendWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
