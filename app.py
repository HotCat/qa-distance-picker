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
    QDoubleSpinBox,
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem, QHeaderView,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QPoint
from PySide6.QtGui import QImage, QPixmap, QKeyEvent

from camera import MindVisionCamera, CameraSettings, CameraSettingRanges
from processing import (
    WatershedProcessor, OverlayRenderer, SegmentationResult, DistanceResult,
    PICK_OBJECT1, PICK_OBJECT2, SHOW_RESULT,
)
from calibration import CalibrationWorker, CalibrationResult
from detect_lines import (
    LinesArcsWorker, LinesArcsResult,
    render_annotations,
)


def _app_dir() -> str:
    """Return the directory containing the app (source or frozen executable)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


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

        # Mirror checkboxes
        self._reverse_x_check = QCheckBox("Reverse X (Horizontal Mirror)")
        self._reverse_x_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._reverse_x_check)

        self._reverse_y_check = QCheckBox("Reverse Y (Vertical Mirror)")
        self._reverse_y_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._reverse_y_check)

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
        self._reverse_x_check.setChecked(settings.reverse_x)
        self._reverse_y_check.setChecked(settings.reverse_y)
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
            reverse_x=self._reverse_x_check.isChecked(),
            reverse_y=self._reverse_y_check.isChecked(),
        )
        self.settings_changed.emit(settings)


# ════════════════════════════════════════════════════════════════════════
#  Detection Results Window
# ════════════════════════════════════════════════════════════════════════

class ResultsWindow(QWidget):
    """Floating tree view showing detected lines and arcs with filter controls."""

    item_selected = Signal(str, str)  # ("line"|"arc", id) or ("", "")
    filters_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Results")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMinimumSize(480, 350)
        self._full_result: LinesArcsResult | None = None
        self._block_filters = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Filter bar ──────────────────────────────────────────────
        filter_layout = QHBoxLayout()

        # Line length filter
        filter_layout.addWidget(QLabel("Line len:"))
        self._line_min_spin = QDoubleSpinBox()
        self._line_min_spin.setRange(0.0, 500.0)
        self._line_min_spin.setDecimals(1)
        self._line_min_spin.setSingleStep(0.5)
        self._line_min_spin.setSuffix(" mm")
        self._line_min_spin.setValue(3.0)
        self._line_min_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._line_min_spin)

        filter_layout.addWidget(QLabel("–"))
        self._line_max_spin = QDoubleSpinBox()
        self._line_max_spin.setRange(0.0, 500.0)
        self._line_max_spin.setDecimals(1)
        self._line_max_spin.setSingleStep(0.5)
        self._line_max_spin.setSuffix(" mm")
        self._line_max_spin.setValue(200.0)
        self._line_max_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._line_max_spin)

        filter_layout.addSpacing(12)

        # Arc radius filter
        filter_layout.addWidget(QLabel("Arc r:"))
        self._arc_min_spin = QDoubleSpinBox()
        self._arc_min_spin.setRange(0.0, 500.0)
        self._arc_min_spin.setDecimals(1)
        self._arc_min_spin.setSingleStep(0.5)
        self._arc_min_spin.setSuffix(" mm")
        self._arc_min_spin.setValue(1.0)
        self._arc_min_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._arc_min_spin)

        filter_layout.addWidget(QLabel("–"))
        self._arc_max_spin = QDoubleSpinBox()
        self._arc_max_spin.setRange(0.0, 500.0)
        self._arc_max_spin.setDecimals(1)
        self._arc_max_spin.setSingleStep(0.5)
        self._arc_max_spin.setSuffix(" mm")
        self._arc_max_spin.setValue(50.0)
        self._arc_max_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._arc_max_spin)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # ── Tree ────────────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["ID", "Category", "Length (mm)", "Angle (°)", "Radius (mm)"])
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QTreeWidget.SingleSelection)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)

    def _on_filter_changed(self):
        if self._block_filters or self._full_result is None:
            return
        self._apply_filters()
        self.filters_changed.emit()

    def _apply_filters(self):
        result = self._full_result
        line_lo = self._line_min_spin.value()
        line_hi = self._line_max_spin.value()
        arc_lo = self._arc_min_spin.value()
        arc_hi = self._arc_max_spin.value()

        filtered_lines = [lr for lr in result.lines
                         if line_lo <= lr.length_mm <= line_hi]
        filtered_arcs = [ar for ar in result.arcs
                        if arc_lo <= ar.radius_mm <= arc_hi]

        self._populate_tree(filtered_lines, filtered_arcs)

    def _populate_tree(self, lines, arcs):
        self._tree.clear()
        font = self._tree.font()
        font.setBold(True)

        lines_item = QTreeWidgetItem(self._tree, [f"Lines ({len(lines)})", "", "", "", ""])
        lines_item.setExpanded(True)
        lines_item.setFont(0, font)
        for lr in lines:
            QTreeWidgetItem(lines_item, [
                lr.id, lr.category,
                f"{lr.length_mm:.2f}",
                f"{lr.angle_deg:.1f}",
                "",
            ])

        arcs_item = QTreeWidgetItem(self._tree, [f"Curves ({len(arcs)})", "", "", "", ""])
        arcs_item.setExpanded(True)
        arcs_item.setFont(0, font)
        for ar in arcs:
            QTreeWidgetItem(arcs_item, [
                ar.id, "",
                "", "",
                f"{ar.radius_mm:.2f}",
            ])

    def set_filters_from_config(self, det_cfg: dict):
        self._block_filters = True
        self._line_min_spin.setValue(det_cfg.get('line_min_mm', 3.0))
        self._line_max_spin.setValue(det_cfg.get('line_max_mm', 200.0))
        self._arc_min_spin.setValue(det_cfg.get('arc_min_mm', 1.0))
        self._arc_max_spin.setValue(det_cfg.get('arc_max_mm', 50.0))
        self._block_filters = False

    def filter_values(self) -> dict:
        return {
            'line_min_mm': self._line_min_spin.value(),
            'line_max_mm': self._line_max_spin.value(),
            'arc_min_mm': self._arc_min_spin.value(),
            'arc_max_mm': self._arc_max_spin.value(),
        }

    def _on_selection_changed(self, current, _previous):
        if current is None or current.parent() is None:
            self.item_selected.emit("", "")
            return
        item_id = current.text(0)
        parent_text = current.parent().text(0)
        if parent_text.startswith("Lines"):
            self.item_selected.emit("line", item_id)
        elif parent_text.startswith("Curves"):
            self.item_selected.emit("arc", item_id)
        else:
            self.item_selected.emit("", "")

    def update_results(self, result: LinesArcsResult):
        self._full_result = result
        self._apply_filters()

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
        self._calib_pending: bool = False
        self._active_worker: QThread | None = None
        self._template_line_grid: dict | None = None
        self._template_arc_grid: dict | None = None
        self._last_arclines_result: LinesArcsResult | None = None

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

        toolbar.addSeparator()

        self._calib_btn = QPushButton("Calibration")
        self._calib_btn.setEnabled(False)
        toolbar.addWidget(self._calib_btn)

        self._arclines_btn = QPushButton("Arclines")
        self._arclines_btn.setEnabled(False)
        toolbar.addWidget(self._arclines_btn)

        # Status bar
        self._status_label = QLabel("No camera connected")
        self.statusBar().addWidget(self._status_label, 1)

        # Camera settings window (floating)
        self._settings_window = CameraSettingsWindow(self)

        # Detection results window (floating)
        self._results_window = ResultsWindow(self)
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
        self._calib_btn.clicked.connect(self._on_calib_clicked)
        self._arclines_btn.clicked.connect(self._on_arclines_clicked)

        # Settings → camera
        self._settings_window.settings_changed.connect(
            self._camera.apply_settings)

        # Results window → highlight and filter
        self._results_window.item_selected.connect(
            self._on_result_item_selected)
        self._results_window.filters_changed.connect(
            self._on_filters_changed)

    # ── Slots: camera frames ────────────────────────────────────────

    @Slot(np.ndarray)
    def _on_live_frame(self, frame: np.ndarray):
        """Display frame from continuous capture."""
        self._last_live_frame = frame.copy()
        self._display_image(frame)

    @Slot(np.ndarray)
    def _on_grab_frame(self, frame: np.ndarray):
        """Software-triggered frame received — enter processing or calibration."""
        if self._calib_pending:
            self._calib_pending = False
            self._run_calibration(frame)
        else:
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
        self._last_arclines_result = None
        self._template_arc_grid = None
        self._template_line_grid = None
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
                pixel_size=self._processor.pixel_size,
            )
            path = "distance_result.png"
            cv2.imwrite(path, composited)
            self._status_label.setText(f"Saved to {path}")

    @Slot()
    def _on_calib_clicked(self):
        """Capture a chessboard frame and run calibration."""
        self._calib_pending = True
        self._status_label.setText("Capturing chessboard frame...")
        self._camera.software_trigger()

    @Slot()
    def _on_arclines_clicked(self):
        """Run line/arc detection on current processing image."""
        if self._proc_state is None or self._proc_state.image_bgr is None:
            self._status_label.setText("No image — click Grab first")
            return
        self._status_label.setText("Running line/arc detection...")
        QApplication.processEvents()

        proc_cfg = self._config.get('processing', {})
        det_cfg = self._config.get('detection', {})
        worker = LinesArcsWorker(
            self._proc_state.image_bgr,
            self._processor.pixel_size,
            gauss_sigma=proc_cfg.get('gauss_sigma', 0.4),
            morph_radius=proc_cfg.get('morph_radius', 3),
            grid_size_mm=det_cfg.get('grid_size_mm', 5.0),
            template_arc_grid=self._template_arc_grid,
            edge_segment_mm=det_cfg.get('edge_segment_mm', 10.0),
            template_line_grid=self._template_line_grid,
        )
        worker.done.connect(self._on_arclines_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    def _run_calibration(self, frame: np.ndarray):
        """Launch calibration worker on a grabbed frame."""
        cal_cfg = self._config.get('calibration', {})
        self._status_label.setText("Running calibration...")
        QApplication.processEvents()

        worker = CalibrationWorker(
            frame,
            cal_cfg.get('board_cols', 11),
            cal_cfg.get('board_rows', 8),
            cal_cfg.get('grid_size_mm', 5.0),
        )
        worker.done.connect(self._on_calibration_done)
        worker.error.connect(self._on_worker_error)
        self._active_worker = worker
        worker.start()

    @Slot(object)
    def _on_calibration_done(self, result: CalibrationResult):
        """Calibration finished — update pixel_size and config."""
        self._processor.pixel_size = result.pixel_size_mm
        self._config['processing']['pixel_size'] = result.pixel_size_mm
        self._active_worker = None
        self._status_label.setText(
            f"Calibrated: {result.pixel_size_mm:.6f} mm/px  "
            f"(board {result.board_cols}×{result.board_rows}, "
            f"grid {result.grid_size_mm} mm, "
            f"mean spacing {result.mean_spacing_px:.2f} px)")

    @Slot(object)
    def _on_arclines_done(self, result: LinesArcsResult):
        """Line/arc detection finished — store result and render."""
        self._active_worker = None
        self._last_arclines_result = result

        # Store template from first run
        if self._template_line_grid is None:
            self._template_line_grid = result.line_grid

        if self._template_arc_grid is None:
            self._template_arc_grid = result.arc_grid

        self._results_window.update_results(result)
        self._results_window.show()

        self._render_arclines()

        n_lines = len(result.lines)
        n_arcs = len(result.arcs)
        self._status_label.setText(
            f"Detected {n_lines} lines, {n_arcs} arcs")

    @Slot(str)
    def _on_worker_error(self, msg: str):
        """Handle errors from background workers."""
        self._active_worker = None
        self._status_label.setText(f"Error: {msg}")

    # ── Render pipeline ───────────────────────────────────────────────

    @Slot(str, str)
    def _on_result_item_selected(self, feature_type: str, feature_id: str):
        """Tree node selected — re-render with highlight."""
        self._render_arclines(highlight_type=feature_type,
                              highlight_id=feature_id)

    def _on_filters_changed(self):
        """Filter thresholds changed — re-render and update tree."""
        if self._last_arclines_result is not None:
            self._results_window._apply_filters()
            self._render_arclines()

    def _render_arclines(self, highlight_type="", highlight_id=""):
        """Render annotated image from detection data with current filters."""
        result = self._last_arclines_result
        if result is None or self._proc_state is None:
            return
        image = self._proc_state.image_bgr
        if image is None:
            return

        # Apply filters
        fv = self._results_window.filter_values()
        filtered_lines = [lr for lr in result.lines
                         if fv['line_min_mm'] <= lr.length_mm <= fv['line_max_mm']]
        filtered_arcs = [ar for ar in result.arcs
                        if fv['arc_min_mm'] <= ar.radius_mm <= fv['arc_max_mm']]

        annotated = render_annotations(
            image, filtered_lines, filtered_arcs,
            highlight_type=highlight_type, highlight_id=highlight_id)
        self._display_image(annotated)

    # ── Mode transitions ────────────────────────────────────────────

    def _switch_to_live(self):
        """Switch to continuous capture mode."""
        self._current_mode = "live"
        self._grab_btn.setEnabled(False)
        self._reset_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._calib_btn.setEnabled(False)
        self._arclines_btn.setEnabled(False)
        self._image_label.setCursor(Qt.ArrowCursor)
        self._proc_state = None
        self._last_arclines_result = None
        self._template_arc_grid = None
        self._template_line_grid = None
        self._camera.set_live_mode()
        self._status_label.setText("Live View")

    def _switch_to_processing(self):
        """Switch to software trigger mode."""
        self._current_mode = "processing"
        self._grab_btn.setEnabled(True)
        self._reset_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._calib_btn.setEnabled(True)
        self._arclines_btn.setEnabled(True)
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
            pixel_size=self._processor.pixel_size,
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
        """Save camera settings to config, then clean up camera."""
        # Save current settings from the settings window
        cam_cfg = self._config.get('camera', {})
        cam_cfg['exposure_us'] = self._settings_window._exposure_spin.value()
        cam_cfg['gamma'] = self._settings_window._gamma_spin.value()
        cam_cfg['contrast'] = self._settings_window._contrast_spin.value()
        cam_cfg['analog_gain'] = self._settings_window._gain_spin.value()
        cam_cfg['ae_enabled'] = self._settings_window._ae_check.isChecked()
        cam_cfg['reverse_x'] = self._settings_window._reverse_x_check.isChecked()
        cam_cfg['reverse_y'] = self._settings_window._reverse_y_check.isChecked()
        self._config['camera'] = cam_cfg

        # Save detection filter values
        det_cfg = self._config.get('detection', {})
        det_cfg.update(self._results_window.filter_values())
        self._config['detection'] = det_cfg

        config_path = os.path.join(_app_dir(), 'config.yaml')
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        except Exception:
            pass  # best-effort save

        self._camera.close()
        super().closeEvent(event)


# ════════════════════════════════════════════════════════════════════════
#  Main entry point
# ════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)

    # Load config — use config.yaml next to the executable (or source file);
    # on first run from a frozen build, extract the bundled default.
    config_path = os.path.join(_app_dir(), 'config.yaml')
    if not os.path.exists(config_path) and getattr(sys, 'frozen', False):
        bundled = os.path.join(sys._MEIPASS, 'config.yaml')
        if os.path.exists(bundled):
            import shutil
            shutil.copy2(bundled, config_path)
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
                reverse_x=cam_cfg.get('reverse_x', False),
                reverse_y=cam_cfg.get('reverse_y', False),
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
    window._results_window.set_filters_from_config(
        config.get('detection', {}))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
