"""
Watershed Distance Picker — PySide6 UI with industrial camera support.

Modes:
  - Live View: continuous camera capture, adjustable settings
  - Image Processing: frozen frame with watershed segmentation and
    click-to-pick distance measurement
"""

import sys
import os
import yaml
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox,
    QPushButton, QToolBar, QStatusBar, QSlider, QSpinBox,
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QVBoxLayout,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QPoint
from PySide6.QtGui import QImage, QPixmap, QKeyEvent

from camera import MindVisionCamera, CameraSettings, CameraSettingRanges
from processing import (
    WatershedProcessor, OverlayRenderer, SegmentationResult, DistanceResult,
    PICK_OBJECT1, PICK_OBJECT2, SHOW_RESULT,
)


# ════════════════════════════════════════════════════════════════════════
#  Camera Settings Window
# ════════════════════════════════════════════════════════════════════════

class CameraSettingsWindow(QWidget):
    """Floating window for camera parameter adjustments."""

    settings_changed = Signal(CameraSettings)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(320)
        self._block_signals = False
        self._build_ui()

    def _build_ui(self):
        layout = QFormLayout(self)

        # Auto-exposure
        self._ae_check = QCheckBox("Auto Exposure")
        self._ae_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._ae_check)

        # Exposure
        self._exposure_slider = QSlider(Qt.Horizontal)
        self._exposure_spin = QSpinBox()
        self._exposure_spin.setSuffix(" us")
        self._exposure_spin.setMinimum(100)
        self._exposure_spin.setMaximum(1000000)
        self._exposure_spin.setSingleStep(100)
        self._exposure_slider.valueChanged.connect(self._exposure_spin.setValue)
        self._exposure_spin.valueChanged.connect(self._exposure_slider.setValue)
        self._exposure_spin.valueChanged.connect(self._on_setting_changed)

        exp_row = QHBoxLayout()
        exp_row.addWidget(self._exposure_slider, 1)
        exp_row.addWidget(self._exposure_spin)
        layout.addRow("Exposure:", exp_row)

        # Gamma
        self._gamma_slider = QSlider(Qt.Horizontal)
        self._gamma_spin = QSpinBox()
        self._gamma_slider.valueChanged.connect(self._gamma_spin.setValue)
        self._gamma_spin.valueChanged.connect(self._gamma_slider.setValue)
        self._gamma_spin.valueChanged.connect(self._on_setting_changed)
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(self._gamma_slider, 1)
        gamma_row.addWidget(self._gamma_spin)
        layout.addRow("Gamma:", gamma_row)

        # Contrast
        self._contrast_slider = QSlider(Qt.Horizontal)
        self._contrast_spin = QSpinBox()
        self._contrast_slider.valueChanged.connect(self._contrast_spin.setValue)
        self._contrast_spin.valueChanged.connect(self._contrast_slider.setValue)
        self._contrast_spin.valueChanged.connect(self._on_setting_changed)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self._contrast_slider, 1)
        contrast_row.addWidget(self._contrast_spin)
        layout.addRow("Contrast:", contrast_row)

        # Analog Gain
        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_spin = QSpinBox()
        self._gain_slider.valueChanged.connect(self._gain_spin.setValue)
        self._gain_spin.valueChanged.connect(self._gain_slider.setValue)
        self._gain_spin.valueChanged.connect(self._on_setting_changed)
        gain_row = QHBoxLayout()
        gain_row.addWidget(self._gain_slider, 1)
        gain_row.addWidget(self._gain_spin)
        layout.addRow("Analog Gain:", gain_row)

    def set_ranges(self, ranges: CameraSettingRanges):
        """Configure slider/spinbox ranges from camera capability."""
        self._block_signals = True
        self._exposure_slider.setRange(ranges.exposure_min_us, ranges.exposure_max_us)
        self._exposure_slider.setSingleStep(ranges.exposure_step_us)
        self._exposure_spin.setRange(ranges.exposure_min_us, ranges.exposure_max_us)
        self._exposure_spin.setSingleStep(ranges.exposure_step_us)
        self._gamma_slider.setRange(ranges.gamma_min, ranges.gamma_max)
        self._gamma_spin.setRange(ranges.gamma_min, ranges.gamma_max)
        self._contrast_slider.setRange(ranges.contrast_min, ranges.contrast_max)
        self._contrast_spin.setRange(ranges.contrast_min, ranges.contrast_max)
        self._gain_slider.setRange(ranges.analog_gain_min, ranges.analog_gain_max)
        self._gain_spin.setRange(ranges.analog_gain_min, ranges.analog_gain_max)
        self._block_signals = False

    def set_values(self, settings: CameraSettings):
        """Set widget values without emitting signals."""
        self._block_signals = True
        self._ae_check.setChecked(settings.ae_enabled)
        self._exposure_slider.setValue(settings.exposure_us)
        self._exposure_spin.setValue(settings.exposure_us)
        self._exposure_slider.setEnabled(not settings.ae_enabled)
        self._exposure_spin.setEnabled(not settings.ae_enabled)
        self._gamma_slider.setValue(settings.gamma)
        self._gamma_spin.setValue(settings.gamma)
        self._contrast_slider.setValue(settings.contrast)
        self._contrast_spin.setValue(settings.contrast)
        self._gain_slider.setValue(settings.analog_gain)
        self._gain_spin.setValue(settings.analog_gain)
        self._block_signals = False

    def _on_setting_changed(self):
        if self._block_signals:
            return
        # Update exposure enabled state based on AE checkbox
        ae_on = self._ae_check.isChecked()
        self._exposure_slider.setEnabled(not ae_on)
        self._exposure_spin.setEnabled(not ae_on)

        settings = CameraSettings(
            exposure_us=self._exposure_spin.value(),
            gamma=self._gamma_spin.value(),
            contrast=self._contrast_spin.value(),
            analog_gain=self._gain_spin.value(),
            ae_enabled=ae_on,
        )
        self.settings_changed.emit(settings)


# ════════════════════════════════════════════════════════════════════════
#  Processing State
# ════════════════════════════════════════════════════════════════════════

class ProcessingState:
    """Holds all mutable state for the current processing image."""
    def __init__(self):
        self.image_bgr: np.ndarray | None = None
        self.image_rgb: np.ndarray | None = None
        self.seg: SegmentationResult | None = None
        self.dip_grey = None  # diplib.Image
        self.obj1_id: int | None = None
        self.obj2_id: int | None = None
        self.click1: tuple[int, int] | None = None
        self.click2: tuple[int, int] | None = None
        self.result: DistanceResult | None = None


# ════════════════════════════════════════════════════════════════════════
#  Main Window
# ════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self, camera: MindVisionCamera,
                 processor: WatershedProcessor,
                 config: dict):
        super().__init__()
        self._camera = camera
        self._processor = processor
        self._config = config

        self._current_mode: str = "live"
        self._proc_state: ProcessingState | None = None
        self._picker_state: int = PICK_OBJECT1
        self._last_live_frame: np.ndarray | None = None
        self._display_pixmap: QPixmap | None = None  # prevent GC

        self._build_ui()
        self._connect_signals()

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Watershed Distance Picker")
        self.setMinimumSize(800, 600)

        # Central image display
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("background-color: #1a1a1a;")
        self._image_label.setCursor(Qt.CrossCursor)
        self._image_label.mousePressEvent = self._on_image_click
        self.setCentralWidget(self._image_label)

        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Live View", "Image Processing"])
        self._mode_combo.setMinimumWidth(150)
        toolbar.addWidget(self._mode_combo)

        toolbar.addSeparator()

        self._grab_btn = QPushButton("Grab")
        self._grab_btn.setEnabled(False)
        toolbar.addWidget(self._grab_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setEnabled(False)
        toolbar.addWidget(self._reset_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setEnabled(False)
        toolbar.addWidget(self._save_btn)

        # Status bar
        self._status_label = QLabel("No camera connected")
        self.statusBar().addWidget(self._status_label, 1)

        # Camera settings window (floating)
        self._settings_window = CameraSettingsWindow(self)

    # ── Signal wiring ───────────────────────────────────────────────

    def _connect_signals(self):
        # Camera → UI
        self._camera.signals.frame_ready.connect(self._on_live_frame)
        self._camera.signals.grab_done.connect(self._on_grab_frame)
        self._camera.signals.error.connect(self._on_camera_error)

        # UI → actions
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._grab_btn.clicked.connect(self._on_grab_clicked)
        self._reset_btn.clicked.connect(self._on_reset)
        self._save_btn.clicked.connect(self._on_save)

        # Settings → camera
        self._settings_window.settings_changed.connect(
            self._camera.apply_settings)

    # ── Slots: camera frames ────────────────────────────────────────

    @Slot(np.ndarray)
    def _on_live_frame(self, frame: np.ndarray):
        """Display frame from continuous capture."""
        self._last_live_frame = frame.copy()
        self._display_image(frame)

    @Slot(np.ndarray)
    def _on_grab_frame(self, frame: np.ndarray):
        """Software-triggered frame received — enter processing."""
        self._enter_processing_with_image(frame)

    @Slot(str)
    def _on_camera_error(self, msg: str):
        self._status_label.setText(f"Camera error: {msg}")

    # ── Slots: mode and buttons ─────────────────────────────────────

    @Slot(int)
    def _on_mode_changed(self, index: int):
        if index == 0:
            self._switch_to_live()
        else:
            self._switch_to_processing()

    @Slot()
    def _on_grab_clicked(self):
        """Grab a new frame via software trigger."""
        self._status_label.setText("Grabbing frame...")
        self._camera.software_trigger()

    @Slot()
    def _on_reset(self):
        """Reset picker state, keep current image."""
        self._picker_state = PICK_OBJECT1
        if self._proc_state:
            self._proc_state.obj1_id = None
            self._proc_state.obj2_id = None
            self._proc_state.click1 = None
            self._proc_state.click2 = None
            self._proc_state.result = None
        self._refresh_overlay()

    @Slot()
    def _on_save(self):
        """Save current overlay image to PNG."""
        if self._proc_state and self._proc_state.image_bgr is not None:
            composited = OverlayRenderer.render(
                self._proc_state.image_bgr,
                self._proc_state.seg.labels,
                self._proc_state.obj1_id,
                self._proc_state.obj2_id,
                self._proc_state.click1,
                self._proc_state.click2,
                self._proc_state.result,
            )
            path = "distance_result.png"
            cv2.imwrite(path, composited)
            self._status_label.setText(f"Saved to {path}")

    # ── Mode transitions ────────────────────────────────────────────

    def _switch_to_live(self):
        """Switch to continuous capture mode."""
        self._current_mode = "live"
        self._grab_btn.setEnabled(False)
        self._reset_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._image_label.setCursor(Qt.ArrowCursor)
        self._proc_state = None
        self._camera.set_live_mode()
        self._status_label.setText("Live View")

    def _switch_to_processing(self):
        """Switch to software trigger mode."""
        self._current_mode = "processing"
        self._grab_btn.setEnabled(True)
        self._reset_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._image_label.setCursor(Qt.CrossCursor)

        # Switch camera to trigger mode
        self._camera.set_trigger_mode()

        # Use last live frame if available
        if self._last_live_frame is not None:
            self._enter_processing_with_image(self._last_live_frame)
        else:
            self._status_label.setText("No frame — click Grab to capture")

    def _enter_processing_with_image(self, frame: np.ndarray):
        """Set up processing state for a new image."""
        self._status_label.setText("Running watershed segmentation...")
        QApplication.processEvents()  # update UI before blocking call

        # Ensure BGR
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        state = ProcessingState()
        state.image_bgr = frame.copy()
        state.image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Watershed segmentation
        state.seg = self._processor.segment(frame)
        state.dip_grey = self._processor.make_dip_grey(frame)

        self._proc_state = state
        self._picker_state = PICK_OBJECT1
        self._refresh_overlay()

        n_regions = len([l for l in state.seg.region_sizes if l != 0])
        self._status_label.setText(
            f"Segmentation done ({n_regions} regions). Click on the FIRST object.")

    # ── Image click (object picking) ────────────────────────────────

    def _on_image_click(self, event):
        if self._current_mode != "processing":
            return
        if self._proc_state is None or self._proc_state.seg is None:
            return

        ix, iy = self._label_to_image(event.position().toPoint())
        if ix is None:
            return

        if self._picker_state == PICK_OBJECT1:
            label = self._processor.get_label_at(
                self._proc_state.seg.labels, ix, iy,
                self._proc_state.seg.region_sizes)
            if label is None:
                self._status_label.setText(
                    "Invalid region — click on an object (not boundary)")
                return
            self._proc_state.obj1_id = label
            self._proc_state.click1 = (ix, iy)
            self._picker_state = PICK_OBJECT2
            self._refresh_overlay()
            self._status_label.setText(
                f"Object 1: ID={label}. Click on the SECOND object.")

        elif self._picker_state == PICK_OBJECT2:
            label = self._processor.get_label_at(
                self._proc_state.seg.labels, ix, iy,
                self._proc_state.seg.region_sizes)
            if label is None:
                self._status_label.setText(
                    "Invalid region — click on an object (not boundary)")
                return
            if label == self._proc_state.obj1_id:
                self._status_label.setText(
                    "Same object — click on a DIFFERENT object")
                return

            self._proc_state.obj2_id = label
            self._proc_state.click2 = (ix, iy)

            # Compute distance
            self._status_label.setText("Computing distance...")
            QApplication.processEvents()

            result = self._processor.compute_distance(
                self._proc_state.image_rgb,
                self._proc_state.seg.labels,
                self._proc_state.dip_grey,
                self._proc_state.obj1_id,
                self._proc_state.obj2_id,
            )
            self._proc_state.result = result
            self._picker_state = SHOW_RESULT
            self._refresh_overlay()

            if result:
                self._status_label.setText(
                    f"Distance: {result.distance_mm:.3f} mm  |  "
                    f"Press Reset to measure again")
            else:
                self._status_label.setText(
                    "Distance computation failed. Press Reset.")

    # ── Rendering ───────────────────────────────────────────────────

    def _display_image(self, bgr_frame: np.ndarray):
        """Convert BGR numpy to QPixmap and display on QLabel."""
        if bgr_frame.ndim == 2:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_GRAY2RGB)
        elif bgr_frame.shape[2] == 1:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        # QImage needs contiguous data
        rgb_cont = np.ascontiguousarray(rgb)
        qimg = QImage(rgb_cont.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit label while preserving aspect ratio
        label_size = self._image_label.size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)
        self._display_pixmap = scaled  # prevent GC

    def _refresh_overlay(self):
        """Re-render overlays and display."""
        if self._proc_state is None or self._proc_state.image_bgr is None:
            return

        composited = OverlayRenderer.render(
            self._proc_state.image_bgr,
            self._proc_state.seg.labels,
            self._proc_state.obj1_id,
            self._proc_state.obj2_id,
            self._proc_state.click1,
            self._proc_state.click2,
            self._proc_state.result,
        )
        self._display_image(composited)

    def _label_to_image(self, pos: QPoint) -> tuple[int | None, int | None]:
        """Map QLabel click coordinates to original image coordinates."""
        pixmap = self._image_label.pixmap()
        if pixmap is None or self._proc_state is None:
            return None, None

        img_h, img_w = self._proc_state.image_bgr.shape[:2]
        label_w = self._image_label.width()
        label_h = self._image_label.height()

        # Pixmap is centered in the label
        pm_w = pixmap.width()
        pm_h = pixmap.height()
        offset_x = (label_w - pm_w) // 2
        offset_y = (label_h - pm_h) // 2

        # Position relative to pixmap
        px = pos.x() - offset_x
        py = pos.y() - offset_y

        if px < 0 or py < 0 or px >= pm_w or py >= pm_h:
            return None, None

        # Scale to image coordinates
        scale_x = img_w / pm_w
        scale_y = img_h / pm_h
        ix = int(px * scale_x)
        iy = int(py * scale_y)

        ix = max(0, min(ix, img_w - 1))
        iy = max(0, min(iy, img_h - 1))

        return ix, iy

    # ── Keyboard ────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean up camera on window close."""
        self._camera.close()
        super().closeEvent(event)


# ════════════════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create modules
    camera = MindVisionCamera()
    proc_cfg = config.get('processing', {})
    processor = WatershedProcessor(
        pixel_size=proc_cfg.get('pixel_size', 0.117027),
        gauss_sigma=proc_cfg.get('gauss_sigma', 0.4),
        morph_radius=proc_cfg.get('morph_radius', 3),
        min_region_size=proc_cfg.get('min_region_size', 500),
        watershed_connectivity=proc_cfg.get('watershed_connectivity', 1),
        watershed_max_depth=proc_cfg.get('watershed_max_depth', 3),
    )

    # Create UI
    window = MainWindow(camera, processor, config)
    window.show()

    # Auto-connect camera if available
    devices = camera.enumerate_devices()
    if devices:
        cam_cfg = config.get('camera', {})
        try:
            camera.open(devices[0]['dev_info'])

            # Apply config defaults
            default_settings = CameraSettings(
                exposure_us=cam_cfg.get('exposure_us', 30000),
                gamma=cam_cfg.get('gamma', 100),
                contrast=cam_cfg.get('contrast', 100),
                analog_gain=cam_cfg.get('analog_gain', 16),
                ae_enabled=cam_cfg.get('ae_enabled', False),
            )
            camera.apply_settings(default_settings)

            # Initialize settings window
            ranges = camera.get_setting_ranges()
            window._settings_window.set_ranges(ranges)
            window._settings_window.set_values(
                camera.get_current_settings())

            # Start live view
            camera.set_live_mode()
            window._status_label.setText(
                f"Connected: {devices[0]['name']} — Live View")
        except Exception as e:
            window._status_label.setText(
                f"Camera open failed: {e}")
    else:
        window._status_label.setText(
            "No camera found — connect camera and restart")

    window._settings_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
