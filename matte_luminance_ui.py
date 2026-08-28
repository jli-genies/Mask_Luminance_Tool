"""Simple PyQt6 UI for per-pass matte / highlight blend controls.

Run::

    python matte_luminance_ui.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

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
    apply_blending_radius,
    apply_composite_pass,
    blur_highlights,
    build_region_gate,
    composite_skin_envelope,
    composite_weights,
    extract_blue_paint_mask,
    highlight_luminance_mask,
    load_rgb,
    luminance,
    luminance_delta_mask,
    make_diffuse_target,
    resize_to,
    sample_self_reference_skin_rgb,
    save_rgb,
)


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


def _compute(
    sample: np.ndarray,
    diffuse_target: np.ndarray,
    id_map: Optional[np.ndarray],
    hl_paint: Optional[np.ndarray],
    composite: Optional[np.ndarray],
    feature_preserve: Optional[np.ndarray],
    chin_mask: Optional[np.ndarray],
    regions: Optional[list],
    region_tolerance: int,
    threshold: float,
    radius: float,
    strength: float,
    concept_diffuse_mix: float,
    hl_threshold: float,
    hl_radius: float,
    hl_strength: float,
    hl_diffuse_mix: float,
    composite_interior_min: float,
    composite_interior_strength: float,
    composite_border_strength: float,
    composite_radius: float,
    exclude_highlights_from_diffuse: bool,
    luminance_only: bool,
    blur_outside_mask: bool,
    hl_blur_outside_mask: bool,
    use_infill: bool,
    enable_concept: bool,
    enable_composite: bool,
    enable_highlight: bool,
) -> Dict[str, np.ndarray]:
    """Run concept + composite + highlight passes on in-memory images."""
    gate = build_region_gate(id_map, DEFAULT_REGION_PALETTE, region_tolerance, regions)
    if isinstance(gate, np.ndarray) and gate.shape == ():
        gate = np.ones(sample.shape[:2], dtype=np.float32)
    elif id_map is None:
        gate = np.ones(sample.shape[:2], dtype=np.float32)

    spill = bool(blur_outside_mask)
    working = sample.copy()
    soft = np.zeros(sample.shape[:2], dtype=np.float32)
    hl_soft = np.zeros(sample.shape[:2], dtype=np.float32)
    composite_border = np.zeros(sample.shape[:2], dtype=np.float32)
    composite_diffuse = np.zeros(sample.shape[:2], dtype=np.float32)

    hl_gate = extract_blue_paint_mask(hl_paint) if hl_paint is not None else None

    if enable_concept:
        raw = luminance_delta_mask(sample, diffuse_target, threshold, gate)
        soft = apply_blending_radius(raw, radius)
        if not spill:
            soft = soft * gate
        else:
            soft = soft * (luminance(sample) >= 8.0).astype(np.float32)
        working = blur_highlights(
            sample,
            soft,
            radius,
            strength,
            diffuse_target=diffuse_target if concept_diffuse_mix > 0.0 else None,
            diffuse_mix=concept_diffuse_mix,
            luminance_only=luminance_only,
            use_infill=use_infill,
        )

    if enable_composite and composite is not None:
        interior_min = float(np.clip(composite_interior_min, 0.0, 255.0)) / 255.0
        hl_for_diffuse = hl_gate if exclude_highlights_from_diffuse else None
        working, composite_diffuse, composite_border = apply_composite_pass(
            working,
            diffuse_target,
            composite,
            composite_radius,
            composite_interior_strength,
            composite_border_strength,
            interior_min=interior_min,
            luminance_only=luminance_only,
            feature_preserve=feature_preserve,
            highlight_preserve=hl_for_diffuse,
            chin_mask=chin_mask,
        )

    if enable_highlight and hl_paint is not None:
        hl_raw = highlight_luminance_mask(working, hl_gate, hl_threshold)
        hl_soft = apply_blending_radius(hl_raw, hl_radius)
        hl_spill = spill or bool(hl_blur_outside_mask)
        if not hl_spill:
            hl_soft = hl_soft * hl_gate
        else:
            hl_soft = hl_soft * (luminance(working) >= 8.0).astype(np.float32)
        working = blur_highlights(
            working,
            hl_soft,
            hl_radius,
            hl_strength,
            diffuse_target=diffuse_target if hl_diffuse_mix > 0.0 else None,
            diffuse_mix=hl_diffuse_mix,
            luminance_only=luminance_only,
            use_infill=use_infill,
        )

    return {
        "texture": working,
        "concept_mask": np.clip(soft * 255.0, 0, 255).astype(np.uint8),
        "hl_mask": np.clip(hl_soft * 255.0, 0, 255).astype(np.uint8),
        "composite_diffuse": np.clip(composite_diffuse * 255.0, 0, 255).astype(np.uint8),
        "composite_border": np.clip(composite_border * 255.0, 0, 255).astype(np.uint8),
        "sample": sample,
    }


# ---------------------------------------------------------------------------
# Background worker (keeps UI responsive)
# ---------------------------------------------------------------------------
class ProcessWorker(QThread):
    finished_ok = pyqtSignal(int, dict)  # job_id, result
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        job_id: int,
        assets: Dict[str, Any],
        params: dict,
        write_outputs: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.job_id = job_id
        self.assets = assets
        self.params = params
        self.write_outputs = write_outputs

    def run(self) -> None:
        try:
            result = _compute(
                sample=self.assets["sample"],
                diffuse_target=self.assets["diffuse_target"],
                id_map=self.assets.get("id_map"),
                hl_paint=self.assets.get("hl_paint"),
                composite=self.assets.get("composite"),
                feature_preserve=self.assets.get("feature_preserve"),
                chin_mask=self.assets.get("chin_mask"),
                regions=self.params["regions"],
                region_tolerance=self.params["region_tolerance"],
                threshold=self.params["threshold"],
                radius=self.params["radius"],
                strength=self.params["strength"],
                concept_diffuse_mix=self.params["concept_diffuse_mix"],
                hl_threshold=self.params["hl_threshold"],
                hl_radius=self.params["hl_radius"],
                hl_strength=self.params["hl_strength"],
                hl_diffuse_mix=self.params["hl_diffuse_mix"],
                composite_interior_min=self.params["composite_interior_min"],
                composite_interior_strength=self.params["composite_interior_strength"],
                composite_border_strength=self.params["composite_border_strength"],
                composite_radius=self.params["composite_radius"],
                exclude_highlights_from_diffuse=self.params["exclude_highlights_from_diffuse"],
                luminance_only=self.params["luminance_only"],
                blur_outside_mask=self.params["blur_outside_mask"],
                hl_blur_outside_mask=self.params["hl_blur_outside_mask"],
                use_infill=self.params["use_infill"],
                enable_concept=self.params["enable_concept"],
                enable_composite=self.params["enable_composite"],
                enable_highlight=self.params["enable_highlight"],
            )
            if self.write_outputs:
                out_mask = self.params["out_mask_path"]
                out_hl = self.params["out_hl_mask_path"]
                out_comp = self.params["out_composite_border_path"]
                out_diff = self.params["out_composite_diffuse_path"]
                out_tex = self.params["out_texture_path"]
                os.makedirs(os.path.dirname(out_mask) or ".", exist_ok=True)
                os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)
                if out_hl:
                    os.makedirs(os.path.dirname(out_hl) or ".", exist_ok=True)
                if out_comp:
                    os.makedirs(os.path.dirname(out_comp) or ".", exist_ok=True)
                if out_diff:
                    os.makedirs(os.path.dirname(out_diff) or ".", exist_ok=True)
                save_rgb(out_mask, result["concept_mask"])
                if out_hl:
                    save_rgb(out_hl, result["hl_mask"])
                if out_comp:
                    save_rgb(out_comp, result["composite_border"])
                if out_diff:
                    save_rgb(out_diff, result["composite_diffuse"])
                save_rgb(out_tex, result["texture"])
                result["paths"] = {
                    "mask": out_mask,
                    "hl_mask": out_hl,
                    "composite_border": out_comp,
                    "composite_diffuse": out_diff,
                    "texture": out_tex,
                }
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
        self.resize(1400, 900)
        self._worker: Optional[ProcessWorker] = None
        self._job_id = 0
        self._pending_run: Optional[Tuple[dict, bool]] = None
        self._assets: Optional[Dict[str, Any]] = None
        self._assets_key: Optional[tuple] = None
        self._root = os.path.dirname(os.path.abspath(__file__))

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
        controls_host.setMinimumWidth(360)
        controls_host.setMaximumWidth(460)
        ch = QVBoxLayout(controls_host)
        ch.setContentsMargins(0, 0, 0, 0)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_inner = QWidget()
        cl = QVBoxLayout(controls_inner)
        cl.setContentsMargins(4, 4, 8, 4)
        controls_scroll.setWidget(controls_inner)
        ch.addWidget(controls_scroll)
        splitter.addWidget(controls_host)

        # --- Inputs ---------------------------------------------------------
        io_box = QGroupBox("Inputs")
        io = QFormLayout(io_box)
        self.texture_edit = self._path_row(io, "Texture (albedo)", "", invalidate=True)
        self.diffuse_edit = self._path_row(io, "Diffuse", "", invalidate=True)
        self.region_edit = self._path_row(io, "Region / concept mask", "", invalidate=True)
        self.highlight_edit = self._path_row(io, "Highlight paint mask", "", invalidate=True)
        self.composite_edit = self._path_row(io, "Composite skin mask", "", invalidate=True)
        self.feature_edit = self._path_row(io, "Feature preserve mask", "", invalidate=True)

        self.diffuse_mode = QComboBox()
        self.diffuse_mode.addItems(["uv", "palette", "self"])
        self.diffuse_mode.currentIndexChanged.connect(self._on_inputs_changed)
        io.addRow("Diffuse mode", self.diffuse_mode)

        self.chk_concept = QCheckBox("Enable concept (luminance-delta) pass")
        self.chk_concept.setChecked(True)
        self.chk_composite = QCheckBox("Enable composite pass (face=diffuse, edge=blur)")
        self.chk_composite.setChecked(True)
        self.chk_highlight = QCheckBox("Enable highlight soften pass")
        self.chk_highlight.setChecked(True)
        io.addRow(self.chk_concept)
        io.addRow(self.chk_composite)
        io.addRow(self.chk_highlight)
        cl.addWidget(io_box)

        # --- Composite pass -------------------------------------------------
        composite_box = QGroupBox("Composite pass  (full face diffuse, preserve features)")
        compf = QFormLayout(composite_box)
        self.composite_interior_min = _SliderSpin(
            200.0, 255.0, 250.0, step=1.0, decimals=0, slider_scale=1
        )
        self.composite_interior_strength = _SliderSpin(
            0.0, 1.0, 0.85, step=0.01, decimals=2, slider_scale=100
        )
        self.composite_border_strength = _SliderSpin(
            0.0, 1.0, 0.85, step=0.01, decimals=2, slider_scale=100
        )
        self.composite_radius = _SliderSpin(0.0, 64.0, 8.0, step=0.5, decimals=1, slider_scale=10)
        self.chk_exclude_hl_diffuse = QCheckBox("Exclude highlight paint from diffuse coverage")
        self.chk_exclude_hl_diffuse.setChecked(True)
        compf.addRow("Border detect min (L≥)", self.composite_interior_min)
        compf.addRow("Diffuse strength", self.composite_interior_strength)
        compf.addRow("Border blur strength", self.composite_border_strength)
        compf.addRow("Border blur radius (px)", self.composite_radius)
        compf.addRow(self.chk_exclude_hl_diffuse)
        cl.addWidget(composite_box)

        # --- Chin extrapolation mask (extends composite border band) --------
        chin_box = QGroupBox("Chin mask  (extends composite border band)")
        chinf = QFormLayout(chin_box)
        self.chin_edit = self._path_row(chinf, "Chin extrapolation mask", "", invalidate=True)
        cl.addWidget(chin_box)

        # --- Concept pass ---------------------------------------------------
        concept_box = QGroupBox("Concept pass  (region-gated luminance-delta correction)")
        cf = QFormLayout(concept_box)
        self.concept_threshold = _SliderSpin(0.0, 80.0, 12.0, step=0.5, decimals=1, slider_scale=10)
        self.concept_radius = _SliderSpin(0.0, 64.0, 8.0, step=0.5, decimals=1, slider_scale=10)
        self.concept_strength = _SliderSpin(0.0, 1.0, 0.85, step=0.01, decimals=2, slider_scale=100)
        self.concept_diffuse_mix = _SliderSpin(0.0, 1.0, 0.0, step=0.01, decimals=2, slider_scale=100)
        cf.addRow("Threshold (dL)", self.concept_threshold)
        cf.addRow("Blur / radius (px)", self.concept_radius)
        cf.addRow("Intensity / strength", self.concept_strength)
        cf.addRow("Diffuse mix (0=blur only)", self.concept_diffuse_mix)
        cl.addWidget(concept_box)

        # --- Highlight pass -------------------------------------------------
        hl_box = QGroupBox("Highlight pass  (blue paint)")
        hf = QFormLayout(hl_box)
        self.hl_threshold = _SliderSpin(0.0, 255.0, 180.0, step=1.0, decimals=1, slider_scale=10)
        self.hl_radius = _SliderSpin(0.0, 64.0, 6.0, step=0.5, decimals=1, slider_scale=10)
        self.hl_strength = _SliderSpin(0.0, 1.0, 0.70, step=0.01, decimals=2, slider_scale=100)
        self.hl_diffuse_mix = _SliderSpin(0.0, 1.0, 0.0, step=0.01, decimals=2, slider_scale=100)
        hf.addRow("Threshold (L)", self.hl_threshold)
        hf.addRow("Blur / radius (px)", self.hl_radius)
        hf.addRow("Intensity / strength", self.hl_strength)
        hf.addRow("Diffuse mix (0=blur only)", self.hl_diffuse_mix)
        cl.addWidget(hl_box)

        # --- Shared / regions -----------------------------------------------
        shared_box = QGroupBox("Shared options")
        sf = QFormLayout(shared_box)
        self.chk_live = QCheckBox("Live preview (update on slider change)")
        self.chk_live.setChecked(True)
        self.chk_luma_only = QCheckBox("Luminance-only blend (keep chroma)")
        self.chk_luma_only.setChecked(True)
        self.chk_spill = QCheckBox("Blur outside painted mask (both passes)")
        self.chk_hl_spill = QCheckBox("Highlight: spill outside blue paint")
        self.chk_use_infill = QCheckBox(
            "Use infill (push surrounding colors) instead of local blur (both passes)"
        )
        sf.addRow(self.chk_live)
        sf.addRow(self.chk_luma_only)
        sf.addRow(self.chk_spill)
        sf.addRow(self.chk_hl_spill)
        sf.addRow(self.chk_use_infill)

        region_row = QHBoxLayout()
        self.region_checks = {}
        for name in DEFAULT_REGION_PALETTE:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.region_checks[name] = cb
            region_row.addWidget(cb)
        sf.addRow("Regions", region_row)

        self.region_tol = _SliderSpin(0.0, 120.0, 40.0, step=1.0, decimals=0, slider_scale=1)
        sf.addRow("Region color tolerance", self.region_tol)
        cl.addWidget(shared_box)

        # --- Outputs --------------------------------------------------------
        out_box = QGroupBox("Outputs")
        of = QFormLayout(out_box)
        out_dir = os.path.join(self._root, "output")
        self.out_texture = self._path_row(
            of, "Corrected texture", os.path.join(out_dir, "albedo_matte.png"), save=True
        )
        self.out_mask = self._path_row(
            of, "Concept mask", os.path.join(out_dir, "blend_mask.png"), save=True
        )
        self.out_hl_mask = self._path_row(
            of, "Highlight mask", os.path.join(out_dir, "hl_blend_mask.png"), save=True
        )
        self.out_composite_border = self._path_row(
            of, "Composite border mask", os.path.join(out_dir, "composite_border_mask.png"), save=True
        )
        self.out_composite_diffuse = self._path_row(
            of, "Composite diffuse mask", os.path.join(out_dir, "composite_diffuse_mask.png"), save=True
        )
        cl.addWidget(out_box)

        self.run_btn = QPushButton("Process & Save")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._on_process)
        cl.addWidget(self.run_btn)

        self.status = QLabel("Ready — adjust sliders for live preview, or Process & Save.")
        self.status.setWordWrap(True)
        cl.addWidget(self.status)
        cl.addStretch(1)

        # --- Preview --------------------------------------------------------
        preview_wrap = QWidget()
        pl = QVBoxLayout(preview_wrap)
        pl.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.viewers: Dict[str, ImageViewer] = {}
        for key, title in (
            ("result", "Result"),
            ("sample", "Original"),
            ("concept_mask", "Concept mask"),
            ("composite_diffuse", "Composite diffuse"),
            ("composite_border", "Composite border"),
            ("hl_mask", "Highlight mask"),
        ):
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
        splitter.setSizes([420, 980])

        self._seed_default_paths()
        self._connect_live_controls()

    # -- live preview wiring -------------------------------------------------
    def _connect_live_controls(self) -> None:
        sliders = (
            self.concept_threshold,
            self.concept_radius,
            self.concept_strength,
            self.concept_diffuse_mix,
            self.composite_interior_min,
            self.composite_interior_strength,
            self.composite_border_strength,
            self.composite_radius,
            self.hl_threshold,
            self.hl_radius,
            self.hl_strength,
            self.hl_diffuse_mix,
            self.region_tol,
        )
        for w in sliders:
            w.valueChanged.connect(self._schedule_live_preview)

        for cb in (
            self.chk_concept,
            self.chk_composite,
            self.chk_highlight,
            self.chk_exclude_hl_diffuse,
            self.chk_luma_only,
            self.chk_spill,
            self.chk_hl_spill,
            *self.region_checks.values(),
        ):
            cb.toggled.connect(self._schedule_live_preview)

        self.chk_live.toggled.connect(self._on_live_toggled)

    def _on_live_toggled(self, on: bool) -> None:
        if on:
            self._schedule_live_preview()

    def _schedule_live_preview(self, *_args) -> None:
        if not self.chk_live.isChecked():
            return
        self._debounce.start()

    def _on_inputs_changed(self, *_args) -> None:
        self._assets = None
        self._assets_key = None
        self._schedule_live_preview()

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
    ) -> QLineEdit:
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(default)
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.clicked.connect(lambda: self._browse(edit, save=save, invalidate=invalidate))
        hl.addWidget(edit)
        hl.addWidget(btn)
        form.addRow(label, row)
        if invalidate:
            edit.editingFinished.connect(self._on_inputs_changed)
        return edit

    def _browse(self, edit: QLineEdit, save: bool = False, invalidate: bool = False) -> None:
        start = edit.text().strip() or self._root
        if save:
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
        masks = os.path.join(self._root, "masks")
        concept = os.path.join(masks, "mask_concept_texture.png")
        highlights = os.path.join(masks, "body_mat_mask_C_highlights_00.png")
        composite = os.path.join(masks, "head_composite_skin_mask.png")
        feature = os.path.join(masks, "head_extrapolation_mask.png")
        chin = os.path.join(masks, "head_extrapolation_mask_chin_area.png")
        if os.path.isfile(concept):
            self.region_edit.setText(concept)
        if os.path.isfile(highlights):
            self.highlight_edit.setText(highlights)
        if os.path.isfile(composite):
            self.composite_edit.setText(composite)
        if os.path.isfile(feature):
            self.feature_edit.setText(feature)
        if os.path.isfile(chin):
            self.chin_edit.setText(chin)

    # -- assets cache --------------------------------------------------------
    def _assets_cache_key(self, params: dict) -> tuple:
        return (
            params["texture_path"],
            params["diffuse_path"],
            params["diffuse_mode"],
            params["region_mask_path"],
            params["highlight_mask_path"],
            params["composite_mask_path"],
            params["feature_preserve_path"],
            params["chin_mask_path"],
        )

    def _ensure_assets(self, params: dict) -> Dict[str, Any]:
        key = self._assets_cache_key(params)
        if self._assets is not None and self._assets_key == key:
            return self._assets

        sample = load_rgb(params["texture_path"])

        id_map = None
        if params["region_mask_path"]:
            id_map = resize_to(
                load_rgb(params["region_mask_path"]), sample.shape[:2], nearest=True
            )

        hl_paint = None
        if params["highlight_mask_path"]:
            hl_paint = resize_to(
                load_rgb(params["highlight_mask_path"]), sample.shape[:2], nearest=True
            )

        composite = None
        if params["composite_mask_path"]:
            composite = resize_to(
                load_rgb(params["composite_mask_path"]), sample.shape[:2], nearest=False
            )

        feature_preserve = None
        if params["feature_preserve_path"]:
            feature_preserve = resize_to(
                load_rgb(params["feature_preserve_path"]), sample.shape[:2], nearest=False
            )

        chin_mask = None
        if params["chin_mask_path"]:
            chin_mask = resize_to(
                load_rgb(params["chin_mask_path"]), sample.shape[:2], nearest=False
            )

        if params["diffuse_mode"] == "self":
            if id_map is not None:
                gate = build_region_gate(id_map, DEFAULT_REGION_PALETTE, params["region_tolerance"], params["regions"])
                exclude = gate.copy()
            else:
                exclude = np.zeros(sample.shape[:2], dtype=np.float32)
            if hl_paint is not None:
                exclude = np.maximum(exclude, extract_blue_paint_mask(hl_paint))
            if feature_preserve is not None:
                exclude = np.maximum(exclude, composite_weights(feature_preserve))
            envelope_ref = composite_skin_envelope(composite) if composite is not None else None
            skin_rgb = sample_self_reference_skin_rgb(sample, exclude, envelope_ref)
            diffuse_target = np.empty(sample.shape, dtype=np.float32)
            diffuse_target[...] = skin_rgb
        else:
            diffuse_img = load_rgb(params["diffuse_path"])
            diffuse_target = make_diffuse_target(sample, diffuse_img, params["diffuse_mode"])

        self._assets = {
            "sample": sample,
            "diffuse_target": diffuse_target,
            "id_map": id_map,
            "hl_paint": hl_paint,
            "composite": composite,
            "feature_preserve": feature_preserve,
            "chin_mask": chin_mask,
        }
        self._assets_key = key
        return self._assets

    # -- process -------------------------------------------------------------
    def _gather_params(self) -> dict:
        texture = self.texture_edit.text().strip()
        diffuse = self.diffuse_edit.text().strip()
        diffuse_mode = self.diffuse_mode.currentText()
        if not texture or not os.path.isfile(texture):
            raise FileNotFoundError("Select a valid texture (albedo) image.")
        if diffuse_mode != "self" and (not diffuse or not os.path.isfile(diffuse)):
            raise FileNotFoundError("Select a valid diffuse image (or switch diffuse mode to 'self').")

        region = self.region_edit.text().strip() or None
        highlight = self.highlight_edit.text().strip() or None
        composite = self.composite_edit.text().strip() or None
        feature = self.feature_edit.text().strip() or None
        chin = self.chin_edit.text().strip() or None
        if region and not os.path.isfile(region):
            raise FileNotFoundError(f"Region mask not found: {region}")
        if self.chk_highlight.isChecked() and highlight and not os.path.isfile(highlight):
            raise FileNotFoundError(f"Highlight mask not found: {highlight}")
        if self.chk_composite.isChecked() and composite and not os.path.isfile(composite):
            raise FileNotFoundError(f"Composite mask not found: {composite}")
        if feature and not os.path.isfile(feature):
            raise FileNotFoundError(f"Feature preserve mask not found: {feature}")
        if self.chk_composite.isChecked() and chin and not os.path.isfile(chin):
            raise FileNotFoundError(f"Chin extrapolation mask not found: {chin}")

        regions = [n for n, cb in self.region_checks.items() if cb.isChecked()] or None

        return dict(
            texture_path=texture,
            diffuse_path=diffuse,
            diffuse_mode=diffuse_mode,
            region_mask_path=region,
            highlight_mask_path=highlight if self.chk_highlight.isChecked() else None,
            composite_mask_path=composite if self.chk_composite.isChecked() else None,
            feature_preserve_path=feature,
            chin_mask_path=chin if self.chk_composite.isChecked() else None,
            regions=regions,
            region_tolerance=int(self.region_tol.value()),
            threshold=self.concept_threshold.value(),
            radius=self.concept_radius.value(),
            strength=self.concept_strength.value(),
            concept_diffuse_mix=self.concept_diffuse_mix.value(),
            composite_interior_min=self.composite_interior_min.value(),
            composite_interior_strength=self.composite_interior_strength.value(),
            composite_border_strength=self.composite_border_strength.value(),
            composite_radius=self.composite_radius.value(),
            hl_threshold=self.hl_threshold.value(),
            hl_radius=self.hl_radius.value(),
            hl_strength=self.hl_strength.value(),
            hl_diffuse_mix=self.hl_diffuse_mix.value(),
            exclude_highlights_from_diffuse=self.chk_exclude_hl_diffuse.isChecked(),
            luminance_only=self.chk_luma_only.isChecked(),
            blur_outside_mask=self.chk_spill.isChecked(),
            hl_blur_outside_mask=self.chk_hl_spill.isChecked(),
            use_infill=self.chk_use_infill.isChecked(),
            enable_concept=self.chk_concept.isChecked(),
            enable_composite=self.chk_composite.isChecked(),
            enable_highlight=self.chk_highlight.isChecked(),
            out_mask_path=self.out_mask.text().strip(),
            out_hl_mask_path=self.out_hl_mask.text().strip(),
            out_composite_border_path=self.out_composite_border.text().strip(),
            out_composite_diffuse_path=self.out_composite_diffuse.text().strip(),
            out_texture_path=self.out_texture.text().strip(),
        )

    def _start_job(self, params: dict, write_outputs: bool) -> None:
        if self._worker and self._worker.isRunning():
            self._pending_run = (params, write_outputs)
            self.status.setText("Updating…")
            return

        try:
            assets = self._ensure_assets(params)
        except Exception as exc:
            if write_outputs:
                QMessageBox.warning(self, "Inputs", str(exc))
            else:
                self.status.setText(f"Preview skipped: {exc}")
            return

        self._job_id += 1
        job_id = self._job_id
        if write_outputs:
            self.run_btn.setEnabled(False)
            self.status.setText("Processing & saving…")
        else:
            self.status.setText("Updating preview…")

        self._worker = ProcessWorker(job_id, assets, params, write_outputs, self)
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

    def _on_done(self, job_id: int, result: dict) -> None:
        if job_id != self._job_id:
            return

        self.run_btn.setEnabled(True)
        self.viewers["result"].set_image(result["texture"])
        self.viewers["sample"].set_image(result["sample"])
        self.viewers["concept_mask"].set_image(result["concept_mask"])
        self.viewers["composite_diffuse"].set_image(result["composite_diffuse"])
        self.viewers["composite_border"].set_image(result["composite_border"])
        self.viewers["hl_mask"].set_image(result["hl_mask"])

        if "paths" in result:
            paths = result["paths"]
            self.status.setText(
                f"Saved.\nTexture → {paths['texture']}\n"
                f"Concept mask → {paths['mask']}\n"
                f"Composite border → {paths['composite_border']}\n"
                f"Highlight mask → {paths['hl_mask']}"
            )
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
