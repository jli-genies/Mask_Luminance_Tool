"""Simple PyQt6 UI for per-mask-channel luminance blend controls.

Run::

    python matte_luminance_ui.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
    composite_skin_envelope,
    composite_weights,
    compute_channel_gate,
    estimate_own_mask_color,
    load_combined_weight_mask,
    load_rgb,
    make_diffuse_target,
    resize_to,
    resolve_diffuse_color,
    run_channel_pipeline,
    save_rgb,
)
from texture_edit import apply_exposure_gamma

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_feature_preserve_paths(spec: Optional[str]) -> None:
    """Raises if any ';'-separated path in a feature-preserve mask spec doesn't exist."""
    if not spec:
        return
    for path in spec.split(";"):
        path = path.strip()
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"Feature preserve mask not found: {path}")


def _rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert uint8 RGB/RGBA (or gray) numpy array to a full-resolution QPixmap."""
    arr = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    elif arr.shape[2] == 4:
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
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
# Diffuse color: shows the value the pipeline resolved to, with a manual override
# ---------------------------------------------------------------------------
class DiffuseColorRow(QWidget):
    """Swatch showing the diffuse color the last run resolved to, plus an override.

    With override off, the swatch/text just mirror whatever ``ProcessWorker``
    reports back as the diffuse color it actually used (the auto-detected
    'self' mean, the 'palette' sample, or nothing for 'uv' mode — see
    ``set_unavailable``). Checking "Override" lets the user hand-pick a flat
    RGB instead, starting from whatever was last computed so it reads as
    "lock in this value" rather than resetting to some arbitrary default.
    """

    changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color: Tuple[float, float, float] = (128.0, 128.0, 128.0)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.swatch = QLabel()
        self.swatch.setFixedSize(28, 20)
        self.swatch.setFrameShape(QFrame.Shape.Box)

        self.text = QLabel("—")
        self.text.setMinimumWidth(100)

        self.override_check = QCheckBox("Override")
        self.pick_btn = QPushButton("Pick…")
        self.pick_btn.setEnabled(False)
        self.pick_btn.setFixedWidth(60)

        layout.addWidget(self.swatch)
        layout.addWidget(self.text)
        layout.addWidget(self.override_check)
        layout.addWidget(self.pick_btn)
        layout.addStretch(1)

        self.override_check.toggled.connect(self._on_override_toggled)
        self.pick_btn.clicked.connect(self._on_pick)

        self._update_swatch()

    def _on_override_toggled(self, on: bool) -> None:
        self.pick_btn.setEnabled(on)
        self.changed.emit()

    def _on_pick(self) -> None:
        r, g, b = (int(round(c)) for c in self._color)
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Pick diffuse color")
        if chosen.isValid():
            self._color = (float(chosen.red()), float(chosen.green()), float(chosen.blue()))
            self._update_swatch()
            self.changed.emit()

    def set_computed_color(self, rgb) -> None:
        """Updates the displayed color from the pipeline's actually-used value."""
        self._color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        self._update_swatch()

    def set_unavailable(self) -> None:
        self.text.setText("N/A (UV diffuse map)")

    def _update_swatch(self) -> None:
        r, g, b = (int(round(float(np.clip(c, 0, 255)))) for c in self._color)
        self.swatch.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #888;")
        self.text.setText(f"{r}, {g}, {b}")

    def is_override_enabled(self) -> bool:
        return self.override_check.isChecked()

    def override_color(self) -> Optional[Tuple[float, float, float]]:
        return self._color if self.is_override_enabled() else None

    def stored_color(self) -> Tuple[float, float, float]:
        """Raw color, whether or not override is currently enabled — for presets."""
        return self._color

    def set_preset_state(self, enabled: bool, color: Optional[Tuple[float, float, float]]) -> None:
        if color is not None:
            self._color = (float(color[0]), float(color[1]), float(color[2]))
            self._update_swatch()
        self.override_check.setChecked(enabled)


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
    moveUpRequested = pyqtSignal(object)
    moveDownRequested = pyqtSignal(object)

    def __init__(
        self,
        name: str,
        mask_path: str,
        parent: Optional[QWidget] = None,
        sample_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> None:
        super().__init__(name, parent)
        self.setCheckable(True)
        self.setChecked(False)
        self._sample_provider = sample_provider
        self._beard_color: Tuple[int, int, int] = (101, 67, 44)

        form = QFormLayout(self)

        self.mask_type = QComboBox()
        self.mask_type.addItems(["Custom", "Shadow", "Highlight", "Beard"])
        self.mask_type.setToolTip(
            "One-shot preset: applies canned Shadow/Highlight/Beard values to this panel's "
            "own fields below. Unlike the Blender addon, these fields are NOT kept in sync "
            "across channels afterward — re-pick the type to re-apply if you change your mind."
        )
        form.addRow("Mask type (preset)", self.mask_type)

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

        self.beard_color_row = QWidget()
        bcl = QHBoxLayout(self.beard_color_row)
        bcl.setContentsMargins(0, 0, 0, 0)
        self.beard_swatch = QLabel()
        self.beard_swatch.setFixedSize(28, 20)
        self.beard_swatch.setFrameShape(QFrame.Shape.Box)
        self.beard_color_text = QLabel("—")
        self.beard_color_text.setMinimumWidth(90)
        beard_pick_btn = QPushButton("Pick…")
        beard_pick_btn.setFixedWidth(60)
        beard_sample_btn = QPushButton("Sample from mask")
        beard_sample_btn.setToolTip(
            "Averages the sample texture's own pixels under this channel's Mask file (the "
            "segmented beard mask, or any spatial region) to set the reference beard color."
        )
        beard_pick_btn.clicked.connect(self._on_pick_beard_color)
        beard_sample_btn.clicked.connect(self._on_sample_beard_color)
        bcl.addWidget(self.beard_swatch)
        bcl.addWidget(self.beard_color_text)
        bcl.addWidget(beard_pick_btn)
        bcl.addWidget(beard_sample_btn)
        bcl.addStretch(1)
        form.addRow("Beard color", self.beard_color_row)
        self._update_beard_swatch()

        self.threshold = _SliderSpin(0.0, 80.0, 12.0, step=0.5, decimals=1, slider_scale=10)
        self.radius = _SliderSpin(0.0, 200.0, 8.0, step=0.5, decimals=1, slider_scale=10)
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

        self.fill_from_own_mask = QCheckBox("Fill color from this mask's own average (not shared skin color)")
        self.fill_from_own_mask.setToolTip(
            "Only used when Flat fill is on. Instead of the shared clean-skin color every "
            "other flat-fill channel draws from, tint the fill with the mean color of the "
            "sample's own pixels under THIS channel's mask — e.g. a highlight mask filling "
            "with its own (brighter) average instead of matching a shadow channel's fill."
        )
        form.addRow(self.fill_from_own_mask)

        self.mask_authoritative = QCheckBox("Mask authoritative (use mask opacity directly)")
        self.mask_authoritative.setToolTip(
            "Skip the luminance-difference threshold and trust this mask's own painted "
            "opacity as full coverage. Fixes blotchy partial coverage inside a hand-painted "
            "mask whose pixels don't happen to differ much in luminance from the diffuse color."
        )
        form.addRow(self.mask_authoritative)

        self.blend_group = QLineEdit()
        self.blend_group.setPlaceholderText("optional — e.g. skin_uniform")
        form.addRow("Blend group", self.blend_group)
        self.blend_weight = _SliderSpin(0.0, 5.0, 1.0, step=0.05, decimals=2, slider_scale=100)
        form.addRow("Blend weight (within group)", self.blend_weight)

        order_row = QWidget()
        orl = QHBoxLayout(order_row)
        orl.setContentsMargins(0, 0, 0, 0)
        move_up_btn = QPushButton("▲ Move up")
        move_down_btn = QPushButton("▼ Move down")
        move_up_btn.setToolTip(
            "Move this channel earlier in the processing order, so later channels "
            "(further down the list) layer on top of it wherever masks overlap."
        )
        move_down_btn.setToolTip(
            "Move this channel later in the processing order, so it layers on top "
            "of earlier channels wherever masks overlap."
        )
        move_up_btn.clicked.connect(lambda: self.moveUpRequested.emit(self))
        move_down_btn.clicked.connect(lambda: self.moveDownRequested.emit(self))
        orl.addWidget(move_up_btn)
        orl.addWidget(move_down_btn)
        form.addRow(order_row)

        remove_btn = QPushButton("Remove channel")
        remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))
        form.addRow(remove_btn)

        self.gate_mode.currentTextChanged.connect(self._on_gate_mode_changed)
        for w in (self.threshold, self.radius, self.strength, self.diffuse_mix, self.region_tolerance, self.blend_weight):
            w.valueChanged.connect(lambda *_: self.changed.emit())
        for cb in (self.use_infill, self.spill_outside, self.fill_holes, self.flat_fill,
                   self.mask_authoritative, self.fill_from_own_mask, *self.region_checks.values()):
            cb.toggled.connect(lambda *_: self.changed.emit())
        self.flat_fill.toggled.connect(self._on_flat_fill_toggled)
        self.mask_type.currentTextChanged.connect(self._on_mask_type_changed)
        self.mask_edit.editingFinished.connect(self.changed.emit)
        self.blend_group.editingFinished.connect(self.changed.emit)
        self.toggled.connect(lambda *_: self.changed.emit())

        self._on_gate_mode_changed(self.gate_mode.currentText())
        self._on_flat_fill_toggled(self.flat_fill.isChecked())

    def _on_gate_mode_changed(self, mode: str) -> None:
        self.fill_holes.setVisible(mode == "weight")
        self.region_row.setVisible(mode == "color_id")
        self.region_tolerance.setVisible(mode in ("color_id", "beard_color"))
        self.beard_color_row.setVisible(mode == "beard_color")
        tol_label = self.layout().labelForField(self.region_tolerance)
        if tol_label is not None:
            tol_label.setText("Color tolerance" if mode == "beard_color" else "Region tolerance")
        self.changed.emit()

    def _on_flat_fill_toggled(self, on: bool) -> None:
        # All three are ignored by MaskChannel.flat_fill — the flat
        # mean-color target always fully replaces the pixel (same as
        # diffuse_mix=1), and feathering is always outward-only regardless
        # of spill_outside — see feather_mask_outward().
        self.diffuse_mix.setVisible(not on)
        self.use_infill.setVisible(not on)
        self.spill_outside.setVisible(not on)
        self.fill_from_own_mask.setVisible(on)
        self.changed.emit()

    def _on_mask_type_changed(self, label: str) -> None:
        """One-shot preset: applies canned values to this panel's own widgets.

        Unlike the Blender addon's Mask Type (which keeps every channel of a
        type live-linked to one shared setting), this is a lighter,
        explicitly scoped-down convenience for this standalone tool: it just
        fills in the fields below once, which you're then free to keep
        editing individually. "Custom" applies nothing — it's the neutral
        starting point, not a fourth preset.
        """
        if label == "Shadow":
            self.flat_fill.setChecked(True)
            self.mask_authoritative.setChecked(True)
        elif label == "Highlight":
            self.flat_fill.setChecked(False)
            self.mask_authoritative.setChecked(True)
            self.diffuse_mix.setValue(0.5)
        elif label == "Beard":
            # Same shadow-style correction as "Shadow", but gated by the picked beard color
            # on top of Mask file's own spatial region — see gate_mode="beard_color".
            self.flat_fill.setChecked(True)
            self.mask_authoritative.setChecked(True)
            self.gate_mode.setCurrentText("beard_color")
            sampled = self._sample_beard_color_from_mask()
            if sampled is not None:
                self._beard_color = sampled
                self._update_beard_swatch()
        self.changed.emit()

    def _update_beard_swatch(self) -> None:
        r, g, b = self._beard_color
        self.beard_swatch.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid #888;")
        self.beard_color_text.setText(f"{r}, {g}, {b}")

    def _on_pick_beard_color(self) -> None:
        r, g, b = self._beard_color
        chosen = QColorDialog.getColor(QColor(r, g, b), self, "Pick beard color")
        if chosen.isValid():
            self._beard_color = (chosen.red(), chosen.green(), chosen.blue())
            self._update_beard_swatch()
            self.changed.emit()

    def _sample_beard_color_from_mask(self) -> Optional[Tuple[int, int, int]]:
        """Mean color of the sample texture's pixels under this channel's own Mask file.

        Returns None (rather than raising) on any failure — no texture loaded yet, no/invalid
        mask file, or no coverage — so callers (an explicit button, or the "Beard" one-shot
        preset) can fall back to the current/default color instead of crashing.
        """
        if self._sample_provider is None:
            return None
        mask_path = self.mask_path()
        if not mask_path or not os.path.isfile(mask_path):
            return None
        sample = self._sample_provider()
        if sample is None:
            return None
        try:
            mask_img = resize_to(load_rgb(mask_path), sample.shape[:2], nearest=False)
            gate = composite_skin_envelope(mask_img) if self.fill_holes.isChecked() else composite_weights(mask_img)
            rgb = estimate_own_mask_color(sample, gate)
        except Exception:
            return None
        return tuple(int(round(c)) for c in rgb)

    def _on_sample_beard_color(self) -> None:
        sampled = self._sample_beard_color_from_mask()
        if sampled is None:
            QMessageBox.warning(
                self, "Sample beard color",
                "Could not sample a color — make sure Mask file points at a valid image and "
                "a texture is loaded (Texture (albedo), above)."
            )
            return
        self._beard_color = sampled
        self._update_beard_swatch()
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
            mask_authoritative=self.mask_authoritative.isChecked(),
            fill_from_own_mask=self.fill_from_own_mask.isChecked(),
            beard_color=self._beard_color if mode == "beard_color" else None,
        )

    def to_preset_dict(self) -> dict:
        """Full UI state for this channel (name/mask path included) — for presets."""
        return {
            "name": self.title(),
            "mask_path": self.mask_path(),
            "enabled": self.isChecked(),
            "mask_type": self.mask_type.currentText(),
            "gate_mode": self.gate_mode.currentText(),
            "fill_holes": self.fill_holes.isChecked(),
            "regions": {n: cb.isChecked() for n, cb in self.region_checks.items()},
            "region_tolerance": self.region_tolerance.value(),
            "threshold": self.threshold.value(),
            "radius": self.radius.value(),
            "strength": self.strength.value(),
            "diffuse_mix": self.diffuse_mix.value(),
            "use_infill": self.use_infill.isChecked(),
            "spill_outside": self.spill_outside.isChecked(),
            "flat_fill": self.flat_fill.isChecked(),
            "mask_authoritative": self.mask_authoritative.isChecked(),
            "fill_from_own_mask": self.fill_from_own_mask.isChecked(),
            "blend_group": self.blend_group.text().strip(),
            "blend_weight": self.blend_weight.value(),
            "beard_color": list(self._beard_color),
        }

    def apply_preset_dict(self, data: dict) -> None:
        """Restores UI state saved by ``to_preset_dict``.

        Signals are blocked while applying so intermediate widget updates don't
        each trigger a live-preview re-run, and so the "Mask type" combo's own
        one-shot preset logic (``_on_mask_type_changed``) doesn't clobber the
        explicit values we're about to set — the panel emits a single
        ``changed`` at the end instead.
        """
        self.blockSignals(True)
        try:
            self.mask_edit.setText(data.get("mask_path", self.mask_path()))
            self.setChecked(bool(data.get("enabled", self.isChecked())))

            mask_type = data.get("mask_type")
            if mask_type is not None:
                self.mask_type.blockSignals(True)
                self.mask_type.setCurrentText(mask_type)
                self.mask_type.blockSignals(False)

            gate_mode = data.get("gate_mode")
            if gate_mode is not None:
                self.gate_mode.setCurrentText(gate_mode)

            self.fill_holes.setChecked(bool(data.get("fill_holes", self.fill_holes.isChecked())))

            regions = data.get("regions")
            if regions:
                for rname, cb in self.region_checks.items():
                    cb.setChecked(bool(regions.get(rname, cb.isChecked())))

            for key, widget in (
                ("region_tolerance", self.region_tolerance),
                ("threshold", self.threshold),
                ("radius", self.radius),
                ("strength", self.strength),
                ("diffuse_mix", self.diffuse_mix),
                ("blend_weight", self.blend_weight),
            ):
                if key in data:
                    widget.setValue(data[key])

            self.use_infill.setChecked(bool(data.get("use_infill", self.use_infill.isChecked())))
            self.spill_outside.setChecked(bool(data.get("spill_outside", self.spill_outside.isChecked())))
            self.flat_fill.setChecked(bool(data.get("flat_fill", self.flat_fill.isChecked())))
            self.mask_authoritative.setChecked(
                bool(data.get("mask_authoritative", self.mask_authoritative.isChecked()))
            )
            self.fill_from_own_mask.setChecked(
                bool(data.get("fill_from_own_mask", self.fill_from_own_mask.isChecked()))
            )
            self.blend_group.setText(data.get("blend_group") or "")

            beard_color = data.get("beard_color")
            if beard_color:
                self._beard_color = tuple(int(c) for c in beard_color)
                self._update_beard_swatch()
        finally:
            self.blockSignals(False)

        # Re-apply visibility rules that depend on gate_mode / flat_fill (normally
        # driven by their toggled/changed signals, which were blocked above).
        self._on_gate_mode_changed(self.gate_mode.currentText())
        self._on_flat_fill_toggled(self.flat_fill.isChecked())
        self.changed.emit()


def _process_texture(params: dict, write_outputs: bool) -> Dict[str, Any]:
    """Runs the matte-blend pipeline for one texture. Shared by ``ProcessWorker`` (single-file /
    live preview) and ``BatchWorker`` (many files, same settings) so the pipeline logic lives
    in exactly one place.
    """
    p = params
    sample: np.ndarray = p["sample"]
    channels: List[MaskChannel] = p["channels"]
    active = [ch for ch in channels if ch.enabled]

    mask_imgs: Dict[str, np.ndarray] = {}
    for ch in active:
        nearest = ch.gate_mode in ("blue_paint", "color_id")
        mask_imgs[ch.name] = resize_to(load_rgb(ch.mask_path), sample.shape[:2], nearest=nearest)

    feature_preserve = None
    if p["feature_preserve_path"]:
        feature_preserve = load_combined_weight_mask(p["feature_preserve_path"], sample.shape[:2])

    palette = DEFAULT_REGION_PALETTE
    exclude = np.zeros(sample.shape[:2], dtype=np.float32)
    for ch in active:
        exclude = np.maximum(exclude, compute_channel_gate(mask_imgs[ch.name], ch, palette, sample=sample))
    if feature_preserve is not None:
        exclude = np.maximum(exclude, feature_preserve)

    diffuse_color_override = p.get("diffuse_color_override")
    diffuse_img = None
    if diffuse_color_override is not None:
        diffuse_target = np.broadcast_to(
            np.asarray(diffuse_color_override, dtype=np.float32), sample.shape
        ).copy()
    elif p["diffuse_mode"] == "self":
        diffuse_target = build_local_diffuse_target(sample, exclude, p["self_locality_radius"])
    else:
        diffuse_img = load_rgb(p["diffuse_path"])
        diffuse_target = make_diffuse_target(sample, diffuse_img, p["diffuse_mode"])

    flat_target = None
    if any(ch.flat_fill for ch in active):
        if diffuse_color_override is not None:
            flat_target = diffuse_target
        else:
            flat_target = build_local_diffuse_target(sample, exclude, p["self_locality_radius"], flat=True)

    own_mask_targets: Dict[str, np.ndarray] = {}
    for ch in active:
        if not ch.flat_fill:
            continue
        if ch.gate_mode == "beard_color":
            own_mask_targets[ch.name] = np.broadcast_to(
                np.asarray(ch.beard_color, dtype=np.float32), sample.shape
            ).copy()
        elif ch.fill_from_own_mask:
            gate = compute_channel_gate(mask_imgs[ch.name], ch, palette, sample=sample)
            color = estimate_own_mask_color(sample, gate)
            own_mask_targets[ch.name] = np.broadcast_to(color, sample.shape).astype(np.float32).copy()

    if diffuse_color_override is not None:
        diffuse_color = np.asarray(diffuse_color_override, dtype=np.float32)
    else:
        diffuse_color = resolve_diffuse_color(sample, p["diffuse_mode"], exclude, diffuse_img)

    working, soft_masks = run_channel_pipeline(
        sample, diffuse_target, mask_imgs, active, palette, p["luminance_only"], feature_preserve, flat_target,
        own_mask_targets,
    )

    exposure, gamma, shadow_bias = p["exposure"], p["gamma"], p["shadow_bias"]
    if exposure != 0.0 or gamma != 1.0 or shadow_bias != 0.0:
        working = apply_exposure_gamma(working, exposure=exposure, gamma=gamma, shadow_bias=shadow_bias)

    channel_masks = {
        name: np.clip(soft * 255.0, 0, 255).astype(np.uint8) for name, soft in soft_masks.items()
    }

    result: Dict[str, Any] = {
        "texture": working,
        "sample": sample,
        "channel_masks": channel_masks,
        "diffuse_color": diffuse_color,
    }

    if write_outputs:
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

    return result


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
            result = _process_texture(self.params, self.write_outputs)
            self.finished_ok.emit(self.job_id, result)
        except Exception:
            self.failed.emit(self.job_id, traceback.format_exc())


class BatchWorker(QThread):
    """Runs ``_process_texture`` for many input files under one fixed settings template.

    One bad file does not abort the run — failures are collected and reported at the end.
    """

    fileDone = pyqtSignal(int, int, str)  # index, total, filename
    fileFailed = pyqtSignal(int, int, str, str)  # index, total, filename, error
    batchFinished = pyqtSignal(int, int, list)  # ok_count, total, failures[(filename, error)]

    def __init__(
        self,
        input_paths: List[str],
        output_dir: str,
        base_params: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.input_paths = input_paths
        self.output_dir = output_dir
        self.base_params = base_params

    def run(self) -> None:
        total = len(self.input_paths)
        ok_count = 0
        failures: List[Tuple[str, str]] = []
        for i, path in enumerate(self.input_paths, start=1):
            name = os.path.basename(path)
            try:
                params = dict(self.base_params)
                params["sample"] = load_rgb(path)
                params["out_texture_path"] = os.path.join(self.output_dir, name)
                if params.get("out_masks_dir"):
                    stem = os.path.splitext(name)[0]
                    params["out_masks_dir"] = os.path.join(self.output_dir, "channel_masks", stem)
                _process_texture(params, write_outputs=True)
                ok_count += 1
                self.fileDone.emit(i, total, name)
            except Exception:
                err = traceback.format_exc()
                failures.append((name, err))
                print(f"[batch] failed on {name}:\n{err}")
                self.fileFailed.emit(i, total, name, err)
        self.batchFinished.emit(ok_count, total, failures)


# ---------------------------------------------------------------------------
# Shared path-row helper (module-level so other tabs can reuse it)
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.bmp);;All (*.*)"
DEFAULT_SAVE_FILTER = "Images (*.png *.jpg *.jpeg *.tif);;All (*.*)"


def _path_row(
    form: QFormLayout,
    label: str,
    default: str,
    root: str,
    parent: QWidget,
    save: bool = False,
    invalidate: bool = False,
    is_dir: bool = False,
    on_change: Optional[Any] = None,
    file_filter: Optional[str] = None,
) -> QLineEdit:
    row = QWidget()
    hl = QHBoxLayout(row)
    hl.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit(default)
    btn = QPushButton("…")
    btn.setFixedWidth(32)
    btn.clicked.connect(
        lambda: _browse(
            edit, root, parent, save=save, invalidate=invalidate, is_dir=is_dir,
            on_change=on_change, file_filter=file_filter,
        )
    )
    hl.addWidget(edit)
    hl.addWidget(btn)
    form.addRow(label, row)
    if invalidate and on_change is not None:
        edit.editingFinished.connect(on_change)
    return edit


def _browse(
    edit: QLineEdit,
    root: str,
    parent: QWidget,
    save: bool = False,
    invalidate: bool = False,
    is_dir: bool = False,
    on_change: Optional[Any] = None,
    file_filter: Optional[str] = None,
) -> None:
    start = edit.text().strip() or root
    if is_dir:
        path = QFileDialog.getExistingDirectory(parent, "Select directory", start)
    elif save:
        path, _ = QFileDialog.getSaveFileName(parent, "Save file", start, file_filter or DEFAULT_SAVE_FILTER)
    else:
        path, _ = QFileDialog.getOpenFileName(parent, "Open file", start, file_filter or DEFAULT_IMAGE_FILTER)
    if path:
        edit.setText(path)
        if invalidate and on_change is not None:
            on_change()


def list_images(folder: str) -> List[str]:
    """Sorted list of image files directly inside ``folder`` (matching ``IMAGE_EXTS``)."""
    found: List[str] = []
    for ext in IMAGE_EXTS:
        found.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(found)


# ---------------------------------------------------------------------------
# Shared batch-run bar: input folder + output folder + run button + status.
# Embedded by both MatteBlendPanel and SegmentPanel (multiview_feature_tab.py).
# ---------------------------------------------------------------------------
class BatchBar(QGroupBox):
    runRequested = pyqtSignal()

    def __init__(self, root: str, run_label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__("Batch", parent)
        self._root = root

        form = QFormLayout(self)
        self.input_edit = _path_row(
            form, "Input folder", "", root=root, parent=self, is_dir=True
        )
        self.output_edit = _path_row(
            form, "Output folder", os.path.join(root, "output", "batch"), root=root, parent=self, is_dir=True
        )

        self.run_btn = QPushButton(run_label)
        self.run_btn.setMinimumHeight(32)
        self.run_btn.clicked.connect(self.runRequested.emit)
        form.addRow(self.run_btn)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        form.addRow(self.status)

    def input_dir(self) -> str:
        return self.input_edit.text().strip()

    def output_dir(self) -> str:
        return self.output_edit.text().strip()

    def set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.input_edit.setEnabled(not running)
        self.output_edit.setEnabled(not running)

    def set_status(self, text: str) -> None:
        self.status.setText(text)


# ---------------------------------------------------------------------------
# Matte-blend tool panel (one page of the tabbed app)
# ---------------------------------------------------------------------------
class MatteBlendPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._worker: Optional[ProcessWorker] = None
        self._job_id = 0
        self._pending_run: Optional[Tuple[dict, bool]] = None
        self._sample_cache: Optional[Tuple[str, np.ndarray]] = None
        self._batch_worker: Optional[BatchWorker] = None
        self._root = os.path.dirname(os.path.abspath(__file__))
        self._presets_dir = os.path.join(self._root, "presets")
        self._channel_panels: List[ChannelPanel] = []

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._run_live_preview)

        root = QHBoxLayout(self)
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

        # --- Presets ----------------------------------------------------------
        presets_box = QGroupBox("Presets")
        ppf = QFormLayout(presets_box)
        self.preset_combo = QComboBox()
        ppf.addRow("Preset", self.preset_combo)

        preset_btn_row = QWidget()
        pbl = QHBoxLayout(preset_btn_row)
        pbl.setContentsMargins(0, 0, 0, 0)
        self.preset_load_btn = QPushButton("Load")
        self.preset_save_btn = QPushButton("Save…")
        self.preset_delete_btn = QPushButton("Delete")
        self.preset_refresh_btn = QPushButton("Refresh")
        for b in (self.preset_load_btn, self.preset_save_btn, self.preset_delete_btn, self.preset_refresh_btn):
            pbl.addWidget(b)
        ppf.addRow(preset_btn_row)
        self.preset_load_btn.clicked.connect(self._on_load_preset)
        self.preset_save_btn.clicked.connect(self._on_save_preset)
        self.preset_delete_btn.clicked.connect(self._on_delete_preset)
        self.preset_refresh_btn.clicked.connect(self._refresh_preset_list)
        self._controls_layout.addWidget(presets_box)

        # --- Inputs ---------------------------------------------------------
        io_box = QGroupBox("Inputs")
        io = QFormLayout(io_box)
        self.texture_edit = self._path_row(io, "Texture (albedo)", "", invalidate=True)
        self.diffuse_edit = self._path_row(io, "Diffuse", "", invalidate=True)
        self.feature_edit = self._path_row(io, "Feature preserve mask", "", invalidate=True)
        self.feature_edit.setToolTip(
            "Join several mask paths with ';' to union them, e.g. a lips mask plus a beard "
            "mask that only applies to concept renders that have one."
        )

        self.diffuse_mode = QComboBox()
        self.diffuse_mode.addItems(["self", "uv", "palette"])
        self.diffuse_mode.currentIndexChanged.connect(self._on_inputs_changed)
        io.addRow("Diffuse mode", self.diffuse_mode)

        self.self_locality_radius = _SliderSpin(
            8.0, 800.0, DEFAULT_SELF_LOCALITY_RADIUS, step=2.0, decimals=0, slider_scale=1
        )
        io.addRow("Self-mode locality radius (px)", self.self_locality_radius)
        self.self_locality_radius.valueChanged.connect(self._schedule_live_preview)

        self.diffuse_color_row = DiffuseColorRow()
        self.diffuse_color_row.changed.connect(self._schedule_live_preview)
        io.addRow("Diffuse color", self.diffuse_color_row)

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

        # --- Post-process (global exposure / gamma) --------------------------
        postproc_box = QGroupBox("Post-process")
        pf = QFormLayout(postproc_box)
        self.exposure = _SliderSpin(-4.0, 4.0, 0.0, step=0.05, decimals=2, slider_scale=100)
        self.gamma = _SliderSpin(0.1, 4.0, 1.0, step=0.02, decimals=2, slider_scale=100)
        self.shadow_bias = _SliderSpin(0.0, 1.0, 0.0, step=0.01, decimals=2, slider_scale=100)
        pf.addRow("Exposure (stops)", self.exposure)
        pf.addRow("Gamma", self.gamma)
        pf.addRow("Shadow bias", self.shadow_bias)
        for w in (self.exposure, self.gamma, self.shadow_bias):
            w.valueChanged.connect(self._schedule_live_preview)
        self._controls_layout.addWidget(postproc_box)

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

        # --- Batch (same settings above, applied to every texture in a folder) ---
        self.batch_bar = BatchBar(self._root, "Run batch")
        self.batch_bar.runRequested.connect(self._on_run_batch)
        self._controls_layout.addWidget(self.batch_bar)

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
        self._refresh_preset_list()

    # -- presets --------------------------------------------------------------
    def _preset_path(self, name: str) -> str:
        safe = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
        return os.path.join(self._presets_dir, f"{safe}.json")

    def _refresh_preset_list(self) -> None:
        current = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        if os.path.isdir(self._presets_dir):
            names = sorted(
                os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(self._presets_dir, "*.json"))
            )
            self.preset_combo.addItems(names)
        idx = self.preset_combo.findText(current)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def _channels_to_preset(self) -> List[dict]:
        return [panel.to_preset_dict() for panel in self._channel_panels]

    def to_preset_dict(self) -> dict:
        """Full UI state (everything except the input texture path) — for presets."""
        override_color = self.diffuse_color_row.stored_color()
        return {
            "version": 1,
            "diffuse_path": self.diffuse_edit.text().strip(),
            "diffuse_mode": self.diffuse_mode.currentText(),
            "feature_preserve_path": self.feature_edit.text().strip(),
            "self_locality_radius": self.self_locality_radius.value(),
            "diffuse_color_override_enabled": self.diffuse_color_row.is_override_enabled(),
            "diffuse_color_override": list(override_color),
            "luminance_only": self.chk_luma_only.isChecked(),
            "live_preview": self.chk_live.isChecked(),
            "exposure": self.exposure.value(),
            "gamma": self.gamma.value(),
            "shadow_bias": self.shadow_bias.value(),
            "out_texture_path": self.out_texture.text().strip(),
            "out_masks_dir": self.out_masks_dir.text().strip(),
            "channels": self._channels_to_preset(),
        }

    def apply_preset_dict(self, data: dict) -> None:
        self.diffuse_edit.setText(data.get("diffuse_path", ""))
        mode = data.get("diffuse_mode")
        if mode:
            self.diffuse_mode.setCurrentText(mode)
        self.feature_edit.setText(data.get("feature_preserve_path", ""))
        if "self_locality_radius" in data:
            self.self_locality_radius.setValue(data["self_locality_radius"])

        color = data.get("diffuse_color_override")
        enabled = bool(data.get("diffuse_color_override_enabled", False))
        self.diffuse_color_row.set_preset_state(enabled, tuple(color) if color else None)

        if "luminance_only" in data:
            self.chk_luma_only.setChecked(bool(data["luminance_only"]))
        if "live_preview" in data:
            self.chk_live.setChecked(bool(data["live_preview"]))
        if "exposure" in data:
            self.exposure.setValue(data["exposure"])
        if "gamma" in data:
            self.gamma.setValue(data["gamma"])
        if "shadow_bias" in data:
            self.shadow_bias.setValue(data["shadow_bias"])
        if "out_texture_path" in data:
            self.out_texture.setText(data["out_texture_path"])
        if "out_masks_dir" in data:
            self.out_masks_dir.setText(data["out_masks_dir"])

        # Rebuild mask channels from scratch to match the preset exactly.
        for panel in list(self._channel_panels):
            self._remove_channel_panel(panel)
        for ch_data in data.get("channels", []):
            self._add_channel_panel(ch_data.get("name", "channel"), ch_data.get("mask_path", ""))
            self._channel_panels[-1].apply_preset_dict(ch_data)

        self._sample_cache = None
        self._schedule_live_preview()

    def _on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save preset", "Preset name:", text=self.preset_combo.currentText()
        )
        name = name.strip()
        if not ok or not name:
            return
        os.makedirs(self._presets_dir, exist_ok=True)
        path = self._preset_path(name)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_preset_dict(), f, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save preset", f"Failed to save preset:\n{exc}")
            return
        self._refresh_preset_list()
        idx = self.preset_combo.findText(name)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.status.setText(f"Preset saved: {path}")

    def _on_load_preset(self) -> None:
        name = self.preset_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "Load preset", "Select a preset to load.")
            return
        path = self._preset_path(name)
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Load preset", f"Preset not found: {path}")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load preset", f"Failed to load preset:\n{exc}")
            return
        self.apply_preset_dict(data)
        self.status.setText(f"Preset loaded: {path}")

    def _on_delete_preset(self) -> None:
        name = self.preset_combo.currentText().strip()
        if not name:
            return
        path = self._preset_path(name)
        if not os.path.isfile(path):
            return
        reply = QMessageBox.question(
            self,
            "Delete preset",
            f"Delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            os.remove(path)
        except OSError as exc:
            QMessageBox.critical(self, "Delete preset", f"Failed to delete preset:\n{exc}")
            return
        self._refresh_preset_list()
        self.status.setText(f"Preset deleted: {name}")

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
        panel = ChannelPanel(name, mask_path, sample_provider=self._sample_for_color_pick)
        panel.changed.connect(self._schedule_live_preview)
        panel.removeRequested.connect(self._remove_channel_panel)
        panel.moveUpRequested.connect(lambda p: self._move_channel_panel(p, -1))
        panel.moveDownRequested.connect(lambda p: self._move_channel_panel(p, 1))
        # Insert above the "Add mask…" button, which is always the last item.
        self.channels_layout.insertWidget(self.channels_layout.count() - 1, panel)
        self._channel_panels.append(panel)

    def _remove_channel_panel(self, panel: ChannelPanel) -> None:
        self._channel_panels.remove(panel)
        self.channels_layout.removeWidget(panel)
        panel.deleteLater()
        self._schedule_live_preview()

    def _move_channel_panel(self, panel: ChannelPanel, direction: int) -> None:
        """Reorders ``panel`` by one slot; later slots are applied later, i.e. layer

        on top of earlier ones wherever their masks overlap (see
        ``run_channel_pipeline``).
        """
        idx = self._channel_panels.index(panel)
        new_idx = idx + direction
        if not (0 <= new_idx < len(self._channel_panels)):
            return
        self._channel_panels[idx], self._channel_panels[new_idx] = (
            self._channel_panels[new_idx],
            self._channel_panels[idx],
        )
        self.channels_layout.removeWidget(panel)
        self.channels_layout.insertWidget(new_idx, panel)
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
        """Thin forwarder to the module-level ``_path_row`` (shared with other tabs)."""
        return _path_row(
            form,
            label,
            default,
            root=self._root,
            parent=self,
            save=save,
            invalidate=invalidate,
            is_dir=is_dir,
            on_change=self._on_inputs_changed if invalidate else None,
        )

    def _seed_default_paths(self) -> None:
        pass

    # -- sample cache ---------------------------------------------------------
    def _ensure_sample(self, texture_path: str) -> np.ndarray:
        if self._sample_cache is not None and self._sample_cache[0] == texture_path:
            return self._sample_cache[1]
        sample = load_rgb(texture_path)
        self._sample_cache = (texture_path, sample)
        return sample

    def _sample_for_color_pick(self) -> Optional[np.ndarray]:
        """Passed to each ChannelPanel so its beard-color sampling can read the currently
        loaded texture (above) without the panel needing direct access to this whole tab.
        """
        texture = self.texture_edit.text().strip()
        if not texture or not os.path.isfile(texture):
            return None
        try:
            return self._ensure_sample(texture)
        except Exception:
            return None

    # -- process -------------------------------------------------------------
    def _gather_params(self) -> dict:
        texture = self.texture_edit.text().strip()
        diffuse = self.diffuse_edit.text().strip()
        diffuse_mode = self.diffuse_mode.currentText()
        diffuse_color_override = self.diffuse_color_row.override_color()
        if not texture or not os.path.isfile(texture):
            raise FileNotFoundError("Select a valid texture (albedo) image.")
        if diffuse_mode != "self" and diffuse_color_override is None and (not diffuse or not os.path.isfile(diffuse)):
            raise FileNotFoundError(
                "Select a valid diffuse image, switch diffuse mode to 'self', or enable a diffuse color override."
            )

        feature = self.feature_edit.text().strip() or None
        _validate_feature_preserve_paths(feature)

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
            diffuse_color_override=diffuse_color_override,
            self_locality_radius=self.self_locality_radius.value(),
            feature_preserve_path=feature,
            luminance_only=self.chk_luma_only.isChecked(),
            exposure=self.exposure.value(),
            gamma=self.gamma.value(),
            shadow_bias=self.shadow_bias.value(),
            out_texture_path=self.out_texture.text().strip(),
            out_masks_dir=self.out_masks_dir.text().strip() or None,
        )

    def _gather_batch_template_params(self) -> dict:
        """Like ``_gather_params`` but for a whole folder: validates the settings that are
        shared across every file (masks, diffuse, feature-preserve, post-process) without
        requiring a single texture to already be selected, and without loading a sample.
        """
        diffuse = self.diffuse_edit.text().strip()
        diffuse_mode = self.diffuse_mode.currentText()
        diffuse_color_override = self.diffuse_color_row.override_color()
        if diffuse_mode != "self" and diffuse_color_override is None and (not diffuse or not os.path.isfile(diffuse)):
            raise FileNotFoundError(
                "Select a valid diffuse image, switch diffuse mode to 'self', or enable a diffuse color override."
            )

        feature = self.feature_edit.text().strip() or None
        _validate_feature_preserve_paths(feature)

        channels: List[MaskChannel] = []
        for panel in self._channel_panels:
            ch = panel.to_channel()
            if ch.enabled:
                if not ch.mask_path or not os.path.isfile(ch.mask_path):
                    raise FileNotFoundError(f"Mask file not found for channel '{ch.name}': {ch.mask_path}")
            channels.append(ch)

        return dict(
            channels=channels,
            diffuse_path=diffuse,
            diffuse_mode=diffuse_mode,
            diffuse_color_override=diffuse_color_override,
            self_locality_radius=self.self_locality_radius.value(),
            feature_preserve_path=feature,
            luminance_only=self.chk_luma_only.isChecked(),
            exposure=self.exposure.value(),
            gamma=self.gamma.value(),
            shadow_bias=self.shadow_bias.value(),
            out_masks_dir=self.out_masks_dir.text().strip() or None,
        )

    def _on_run_batch(self) -> None:
        input_dir = self.batch_bar.input_dir()
        output_dir = self.batch_bar.output_dir()
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

        try:
            base_params = self._gather_batch_template_params()
        except Exception as exc:
            QMessageBox.warning(self, "Batch", str(exc))
            return

        os.makedirs(output_dir, exist_ok=True)
        self.batch_bar.set_running(True)
        self.run_btn.setEnabled(False)
        self.batch_bar.set_status(f"Processing 0/{len(input_paths)}…")

        self._batch_worker = BatchWorker(input_paths, output_dir, base_params, self)
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

        diffuse_color = result.get("diffuse_color")
        if diffuse_color is not None:
            self.diffuse_color_row.set_computed_color(diffuse_color)
        else:
            self.diffuse_color_row.set_unavailable()

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


# ---------------------------------------------------------------------------
# Top-level tabbed app window
# ---------------------------------------------------------------------------
class AppWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mask Luminance Tool")
        self.resize(1450, 950)

        from multiview_feature_tab import TextureSegmentationTab

        tabs = QTabWidget()
        tabs.addTab(MatteBlendPanel(), "Matte Blend")
        tabs.addTab(TextureSegmentationTab(), "Texture Segmentation")
        self.setCentralWidget(tabs)


def main() -> None:
    app = QApplication(sys.argv)
    win = AppWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
