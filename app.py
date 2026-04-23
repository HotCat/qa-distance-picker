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
    QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QLineEdit,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QPoint
from PySide6.QtGui import QImage, QPixmap, QKeyEvent, QShortcut, QKeySequence

from camera import MindVisionCamera, CameraSettings, CameraSettingRanges
from processing import (
    WatershedProcessor, OverlayRenderer, SegmentationResult, DistanceResult,
    PICK_OBJECT1, PICK_OBJECT2, SHOW_RESULT,
)
from calibration import CalibrationWorker, CalibrationResult
from detect_lines import (
    LinesArcsWorker, LinesArcsResult, FeaturePair,
    render_annotations, compute_feature_distance, compute_feature_pair_points,
    fuzzy_match_template_pairs, detect_lines_and_arcs,
)
from debug_overlay import (
    draw_grid_overlay, draw_edge_segments,
    compute_grid_cell, compute_edge_segments,
)
from alignment import (
    ransac_rigid_registration, render_alignment_overlay,
    AlignmentResult, RigidTransform,
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
    pair_added = Signal(str, str, str, str)  # (type_a, id_a, type_b, id_b)
    filters_changed = Signal()
    redetect_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Results")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMinimumSize(480, 350)
        self._full_result: LinesArcsResult | None = None
        self._block_filters = False
        self._block_selection = False
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
        self._tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)

        # Redetect button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._redetect_btn = QPushButton("Redetect")
        self._redetect_btn.setToolTip("Grab a fresh frame and re-run detection")
        self._redetect_btn.clicked.connect(self._on_redetect_clicked)
        btn_row.addWidget(self._redetect_btn)
        layout.addLayout(btn_row)

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

    def _on_selection_changed(self):
        if self._block_selection:
            return
        selected = self._tree.selectedItems()
        leaf_items = [item for item in selected if item.parent() is not None]

        if len(leaf_items) == 2:
            types_ids = []
            for item in leaf_items:
                parent = item.parent().text(0)
                type_ = "line" if parent.startswith("Lines") else "arc"
                types_ids.append((type_, item.text(0)))
            self.pair_added.emit(
                types_ids[0][0], types_ids[0][1],
                types_ids[1][0], types_ids[1][1])
            self._block_selection = True
            self._tree.clearSelection()
            self._block_selection = False
        elif len(leaf_items) == 1:
            item = leaf_items[0]
            item_id = item.text(0)
            parent_text = item.parent().text(0)
            if parent_text.startswith("Lines"):
                self.item_selected.emit("line", item_id)
            elif parent_text.startswith("Curves"):
                self.item_selected.emit("arc", item_id)
            else:
                self.item_selected.emit("", "")
        else:
            self.item_selected.emit("", "")

    def update_results(self, result: LinesArcsResult):
        self._full_result = result
        self._apply_filters()

    def _on_redetect_clicked(self):
        self._redetect_btn.setEnabled(False)
        self._redetect_btn.setText("Detecting...")
        self.redetect_requested.emit()

    def reset_redetect_button(self):
        self._redetect_btn.setEnabled(True)
        self._redetect_btn.setText("Redetect")


class PairListWindow(QWidget):
    """Floating window with feature pair table and tolerance controls.

    Supports multiple named configurations (templates) for different products.
    """

    config_confirmed = Signal(list)  # emits full list of configs
    pair_row_selected = Signal(int)  # -1 for no selection, >= 0 for row index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Pair Measurements")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMinimumSize(600, 350)
        self._all_configs: list[dict] = []  # each: {'name': str, 'pairs': list[FeaturePair]}
        self._current_index: int = -1  # -1 means new/unsaved config
        self._block_combo: bool = False
        self._last_matches: list | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Config selector row
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Select:"))
        self._config_combo = QComboBox()
        self._config_combo.setMinimumWidth(150)
        self._config_combo.currentIndexChanged.connect(self._on_config_selected)
        selector_row.addWidget(self._config_combo)

        self._new_btn = QPushButton("New")
        self._new_btn.clicked.connect(self._on_new_config)
        selector_row.addWidget(self._new_btn)

        self._delete_config_btn = QPushButton("Delete Config")
        self._delete_config_btn.clicked.connect(self._on_delete_config)
        selector_row.addWidget(self._delete_config_btn)
        selector_row.addStretch()
        layout.addLayout(selector_row)

        # Name input row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Enter configuration name")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Feature A", "Feature B", "Distance (mm)",
            "Lower (mm)", "Upper (mm)",
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._table.setColumnWidth(0, 180)
        self._table.setColumnWidth(1, 180)
        self._table.verticalHeader().show()
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.currentCellChanged.connect(self._on_row_changed)
        layout.addWidget(self._table)

        # Buttons
        btn_row = QHBoxLayout()
        self._delete_btn = QPushButton("Delete Row")
        self._delete_btn.clicked.connect(self._on_delete_row)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()
        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(self._confirm_btn)
        layout.addLayout(btn_row)

    @property
    def _pairs(self) -> list[FeaturePair]:
        """Return current config's pairs, or empty list for new config."""
        if 0 <= self._current_index < len(self._all_configs):
            return self._all_configs[self._current_index]['pairs']
        # For new config, return a temporary list stored on the object
        if not hasattr(self, '_temp_pairs'):
            self._temp_pairs: list[FeaturePair] = []
        return self._temp_pairs

    def _on_config_selected(self, index: int):
        """Combo box selection changed — switch to different config."""
        if self._block_combo:
            return
        # Sync current pairs before switching
        self._sync_current_config()
        # Load selected config
        self._current_index = index
        if 0 <= index < len(self._all_configs):
            cfg = self._all_configs[index]
            self._name_edit.setText(cfg['name'])
            self._refresh_table()
        self.pair_row_selected.emit(-1)

    def _on_new_config(self):
        """Start a new blank configuration."""
        self._sync_current_config()
        self._current_index = -1
        self._temp_pairs: list[FeaturePair] = []
        self._name_edit.clear()
        self._block_combo = True
        self._config_combo.setCurrentIndex(-1)
        self._block_combo = False
        self._refresh_table()
        self.pair_row_selected.emit(-1)

    def _on_delete_config(self):
        """Delete the currently selected configuration."""
        if 0 <= self._current_index < len(self._all_configs):
            name = self._all_configs[self._current_index]['name']
            self._all_configs.pop(self._current_index)
            self._refresh_combo()
            # Select first remaining config or go to new
            if self._all_configs:
                self._current_index = 0
                self._name_edit.setText(self._all_configs[0]['name'])
                self._refresh_table()
            else:
                self._on_new_config()
            self.pair_row_selected.emit(-1)

    def _sync_current_config(self):
        """Sync bounds from table into current config's pairs."""
        self._sync_bounds_from_table()
        # If we're on an existing config, update it
        if 0 <= self._current_index < len(self._all_configs):
            self._all_configs[self._current_index]['pairs'] = self._pairs.copy()
        # Also update name if different
        name = self._name_edit.text().strip()
        if name and 0 <= self._current_index < len(self._all_configs):
            self._all_configs[self._current_index]['name'] = name

    def _refresh_combo(self):
        """Repopulate combo box from _all_configs."""
        self._block_combo = True
        self._config_combo.clear()
        for cfg in self._all_configs:
            self._config_combo.addItem(cfg['name'])
        if 0 <= self._current_index < len(self._all_configs):
            self._config_combo.setCurrentIndex(self._current_index)
        self._block_combo = False

    def add_pair(self, pair: FeaturePair):
        """Add a pair to current config."""
        self._pairs.append(pair)
        self._refresh_table()

    def _on_row_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int):
        if 0 <= row < len(self._pairs):
            self.pair_row_selected.emit(row)
        else:
            self.pair_row_selected.emit(-1)

    def _refresh_table(self):
        pairs = self._pairs
        self._table.setRowCount(len(pairs))
        for i, pair in enumerate(pairs):
            # Feature IDs (read-only)
            fa = QTableWidgetItem(f"{pair.type_a[0].upper()}: {pair.id_a}")
            fa.setFlags(fa.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 0, fa)

            fb = QTableWidgetItem(f"{pair.type_b[0].upper()}: {pair.id_b}")
            fb.setFlags(fb.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 1, fb)

            # Distance (read-only)
            dist = QTableWidgetItem(f"{pair.distance_mm:.3f}")
            dist.setFlags(dist.flags() & ~Qt.ItemIsEditable)
            dist.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 2, dist)

            # Lower bound (editable)
            lower = QTableWidgetItem(f"{pair.lower_mm:.3f}")
            lower.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 3, lower)

            # Upper bound (editable)
            upper = QTableWidgetItem(f"{pair.upper_mm:.3f}")
            upper.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 4, upper)

    def _sync_bounds_from_table(self):
        """Read edited bound values back into pairs."""
        pairs = self._pairs
        for i, pair in enumerate(pairs):
            lo_item = self._table.item(i, 3)
            hi_item = self._table.item(i, 4)
            if lo_item:
                try:
                    pair.lower_mm = float(lo_item.text())
                except ValueError:
                    pass
            if hi_item:
                try:
                    pair.upper_mm = float(hi_item.text())
                except ValueError:
                    pass

    def _on_delete_row(self):
        """Delete selected row from current config."""
        row = self._table.currentRow()
        if 0 <= row < len(self._pairs):
            self._pairs.pop(row)
            self._refresh_table()
        self.pair_row_selected.emit(-1)

    def _on_confirm(self):
        """Confirm current config — save/update in _all_configs."""
        self._sync_bounds_from_table()
        name = self._name_edit.text().strip() or "unnamed"
        pairs_copy = self._pairs.copy()

        # Find existing config with this name
        existing_idx = -1
        for i, cfg in enumerate(self._all_configs):
            if cfg['name'] == name:
                existing_idx = i
                break

        if existing_idx >= 0:
            # Update existing
            self._all_configs[existing_idx]['pairs'] = pairs_copy
            self._current_index = existing_idx
        else:
            # Add new
            self._all_configs.append({'name': name, 'pairs': pairs_copy})
            self._current_index = len(self._all_configs) - 1

        # Refresh combo and table
        self._refresh_combo()
        self._refresh_table()

        # Emit full config list for saving
        self.config_confirmed.emit(self.get_config_list())

    def update_distances(self, lines, arcs, pixel_size_mm, image_bgr, config):
        """Re-compute distances for current config using fuzzy matching."""
        if not self._pairs:
            return
        img_h, img_w = image_bgr.shape[:2]
        det_cfg = config.get('detection', {})
        matches = fuzzy_match_template_pairs(
            self._pairs, lines, arcs,
            img_w, img_h, pixel_size_mm,
            det_cfg.get('edge_segment_mm', 10.0),
            det_cfg.get('grid_size_mm', 5.0),
            line_min_mm=det_cfg.get('line_min_mm', 0.0),
            line_max_mm=det_cfg.get('line_max_mm', float('inf')),
            arc_min_mm=det_cfg.get('arc_min_mm', 0.0),
            arc_max_mm=det_cfg.get('arc_max_mm', float('inf')),
        )
        # matches: list of (distance_mm | None, FeaturePair, pt_a, pt_b, det_a, det_b)
        for i, (dist, pair, pt_a, pt_b, det_a, det_b) in enumerate(matches):
            if i < len(self._pairs):
                self._pairs[i].distance_mm = dist if dist is not None else 0.0
        self._last_matches = matches
        self._refresh_table()

    def load_config(self, config_list: list):
        """Load all stored configurations from config.yaml."""
        self._all_configs.clear()
        for cfg in config_list:
            pairs = []
            for p in cfg.get('pairs', []):
                pairs.append(FeaturePair(
                    type_a=p.get('type_a', 'line'),
                    id_a=p.get('id_a', ''),
                    type_b=p.get('type_b', 'line'),
                    id_b=p.get('id_b', ''),
                    distance_mm=p.get('distance_mm', 0.0),
                    lower_mm=p.get('lower_mm', 0.0),
                    upper_mm=p.get('upper_mm', 0.0),
                ))
            self._all_configs.append({
                'name': cfg.get('name', 'unnamed'),
                'pairs': pairs,
            })
        self._refresh_combo()
        if self._all_configs:
            self._current_index = 0
            self._name_edit.setText(self._all_configs[0]['name'])
            self._refresh_table()
        else:
            self._on_new_config()

    def get_config_list(self) -> list:
        """Return all configs serialized for yaml.dump."""
        self._sync_current_config()
        result = []
        for cfg in self._all_configs:
            pairs_data = []
            for p in cfg['pairs']:
                pairs_data.append({
                    'type_a': p.type_a, 'id_a': p.id_a,
                    'type_b': p.type_b, 'id_b': p.id_b,
                    'distance_mm': round(p.distance_mm, 4),
                    'lower_mm': round(p.lower_mm, 3),
                    'upper_mm': round(p.upper_mm, 3),
                })
            result.append({'name': cfg['name'], 'pairs': pairs_data})
        return result

    def get_pair(self, index: int) -> FeaturePair | None:
        if 0 <= index < len(self._pairs):
            return self._pairs[index]
        return None

    def get_config_names(self) -> list[str]:
        """Return list of all config names for BatchInspect."""
        return [cfg['name'] for cfg in self._all_configs]

    def get_pairs_for_config(self, name: str) -> list[FeaturePair]:
        """Return pairs for a named config (for BatchInspect)."""
        for cfg in self._all_configs:
            if cfg['name'] == name:
                return cfg['pairs'].copy()
        return []


# ════════════════════════════════════════════════════════════════════════
#  Batch Inspect Worker
# ════════════════════════════════════════════════════════════════════════

class BatchInspectWorker(QThread):
    """Run detection + fuzzy matching for batch inspection."""

    done = Signal(list)  # list of (distance_mm | None, FeaturePair, pt_a, pt_b)
    error = Signal(str)

    def __init__(self, image_bgr: np.ndarray, template_pairs: list[FeaturePair],
                 pixel_size_mm: float, gauss_sigma: float, morph_radius: int,
                 grid_size_mm: float, segment_mm: float,
                 line_min_mm: float = 0.0,
                 line_max_mm: float = float('inf'),
                 arc_min_mm: float = 0.0,
                 arc_max_mm: float = float('inf')):
        super().__init__()
        self._image = image_bgr.copy()
        self._template_pairs = template_pairs
        self._pixel_size_mm = pixel_size_mm
        self._gauss_sigma = gauss_sigma
        self._morph_radius = morph_radius
        self._grid_size_mm = grid_size_mm
        self._segment_mm = segment_mm
        self._line_min_mm = line_min_mm
        self._line_max_mm = line_max_mm
        self._arc_min_mm = arc_min_mm
        self._arc_max_mm = arc_max_mm

    def run(self):
        try:
            # Run detection
            result = detect_lines_and_arcs(
                self._image, self._pixel_size_mm,
                gauss_sigma=self._gauss_sigma,
                morph_radius=self._morph_radius,
                grid_size_mm=self._grid_size_mm,
                edge_segment_mm=self._segment_mm,
            )
            img_h, img_w = self._image.shape[:2]
            # Run fuzzy matching
            matches = fuzzy_match_template_pairs(
                self._template_pairs,
                result.lines, result.arcs,
                img_w, img_h,
                self._pixel_size_mm,
                self._segment_mm,
                self._grid_size_mm,
                line_min_mm=self._line_min_mm,
                line_max_mm=self._line_max_mm,
                arc_min_mm=self._arc_min_mm,
                arc_max_mm=self._arc_max_mm,
            )
            self.done.emit(matches)
        except Exception as e:
            self.error.emit(f"Batch inspect failed: {e}")


# ════════════════════════════════════════════════════════════════════════
#  Batch Inspect Window
# ════════════════════════════════════════════════════════════════════════

class BatchInspectWindow(QWidget):
    """Floating window for batch inspection against saved templates."""

    inspect_requested = Signal()  # emitted when user clicks Inspect
    pair_row_selected = Signal(int)  # -1 for no selection, >= 0 for row index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Inspect")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMinimumSize(550, 350)
        self._results: list[tuple[float | None, FeaturePair]] = []
        self._config_pairs: dict[str, list[FeaturePair]] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Template selector
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Template:"))
        self._template_combo = QComboBox()
        self._template_combo.setMinimumWidth(150)
        self._template_combo.currentTextChanged.connect(self._on_template_changed)
        selector_row.addWidget(self._template_combo)
        selector_row.addStretch()
        layout.addLayout(selector_row)

        # Table: #, Feature A, Feature B, Distance, Lower, Upper, Pass?
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "Feature A", "Feature B", "Distance (mm)",
            "Lower (mm)", "Upper (mm)", "Pass",
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self._table.setColumnWidth(0, 180)
        self._table.setColumnWidth(1, 180)
        self._table.verticalHeader().show()
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.currentCellChanged.connect(self._on_row_changed)
        layout.addWidget(self._table)

        # Inspect button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._inspect_btn = QPushButton("Inspect")
        self._inspect_btn.clicked.connect(self._on_inspect_clicked)
        btn_row.addWidget(self._inspect_btn)
        layout.addLayout(btn_row)

    def set_config_names(self, configs: dict[str, list[FeaturePair]]):
        """Populate template combo with available config names and their pairs."""
        self._config_pairs = configs
        current = self._template_combo.currentText()
        self._template_combo.blockSignals(True)
        self._template_combo.clear()
        for name in configs:
            self._template_combo.addItem(name)
        # Restore selection if possible
        idx = self._template_combo.findText(current)
        if idx >= 0:
            self._template_combo.setCurrentIndex(idx)
        self._template_combo.blockSignals(False)
        # Show pairs for current selection
        self._on_template_changed(self._template_combo.currentText())

    def _on_template_changed(self, name: str):
        """Template selection changed — show pairs with blank distances."""
        pairs = self._config_pairs.get(name, [])
        self._table.setRowCount(len(pairs))
        for i, pair in enumerate(pairs):
            fa = QTableWidgetItem(f"{pair.type_a[0].upper()}: {pair.id_a}")
            fa.setFlags(fa.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 0, fa)

            fb = QTableWidgetItem(f"{pair.type_b[0].upper()}: {pair.id_b}")
            fb.setFlags(fb.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 1, fb)

            dist = QTableWidgetItem("—")
            dist.setFlags(dist.flags() & ~Qt.ItemIsEditable)
            dist.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 2, dist)

            lower = QTableWidgetItem(f"{pair.lower_mm:.3f}")
            lower.setFlags(lower.flags() & ~Qt.ItemIsEditable)
            lower.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 3, lower)

            upper = QTableWidgetItem(f"{pair.upper_mm:.3f}")
            upper.setFlags(upper.flags() & ~Qt.ItemIsEditable)
            upper.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 4, upper)

            pass_item = QTableWidgetItem("—")
            pass_item.setFlags(pass_item.flags() & ~Qt.ItemIsEditable)
            pass_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 5, pass_item)

    def get_selected_template(self) -> str:
        """Return currently selected template name."""
        return self._template_combo.currentText()

    def _on_inspect_clicked(self):
        """Notify MainWindow to grab frame and run inspection."""
        self._inspect_btn.setEnabled(False)
        self._inspect_btn.setText("Inspecting...")
        self.inspect_requested.emit()

    def show_results(self, results: list):
        """Display inspection results in table."""
        self._results = results
        self._table.setRowCount(len(results))

        for i, (dist, pair, _pt_a, _pt_b, _det_a, _det_b) in enumerate(results):
            # Feature IDs
            fa = QTableWidgetItem(f"{pair.type_a[0].upper()}: {pair.id_a}")
            fa.setFlags(fa.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 0, fa)

            fb = QTableWidgetItem(f"{pair.type_b[0].upper()}: {pair.id_b}")
            fb.setFlags(fb.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(i, 1, fb)

            # Distance
            if dist is not None:
                dist_item = QTableWidgetItem(f"{dist:.3f}")
            else:
                dist_item = QTableWidgetItem("N/A")
            dist_item.setFlags(dist_item.flags() & ~Qt.ItemIsEditable)
            dist_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 2, dist_item)

            # Lower bound
            lower = QTableWidgetItem(f"{pair.lower_mm:.3f}")
            lower.setFlags(lower.flags() & ~Qt.ItemIsEditable)
            lower.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 3, lower)

            # Upper bound
            upper = QTableWidgetItem(f"{pair.upper_mm:.3f}")
            upper.setFlags(upper.flags() & ~Qt.ItemIsEditable)
            upper.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(i, 4, upper)

            # Pass/Fail
            pass_item = QTableWidgetItem()
            pass_item.setFlags(pass_item.flags() & ~Qt.ItemIsEditable)
            pass_item.setTextAlignment(Qt.AlignCenter)
            if dist is None:
                pass_item.setText("—")
            elif pair.lower_mm == 0.0 and pair.upper_mm == 0.0:
                pass_item.setText("—")  # No tolerance set
            elif pair.lower_mm <= dist <= pair.upper_mm:
                pass_item.setText("✓")
                pass_item.setForeground(Qt.darkGreen)
            else:
                pass_item.setText("✗")
                pass_item.setForeground(Qt.red)
            self._table.setItem(i, 5, pass_item)

        # Re-enable button
        self._inspect_btn.setEnabled(True)
        self._inspect_btn.setText("Inspect")

    def _on_row_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int):
        """Emit row selection for overlay annotation."""
        if 0 <= row < len(self._results):
            self.pair_row_selected.emit(row)
        else:
            self.pair_row_selected.emit(-1)

    def reset_button(self):
        """Re-enable inspect button after error."""
        self._inspect_btn.setEnabled(True)
        self._inspect_btn.setText("Inspect")


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
#  Alignment Playground
# ════════════════════════════════════════════════════════════════════════

class AlignmentWorker(QThread):
    """Run RANSAC alignment in background thread."""

    done = Signal(object)  # AlignmentResult
    error = Signal(str)

    def __init__(self, tmpl_lines, tmpl_arcs, det_lines, det_arcs,
                 pixel_size_mm, n_iterations, inlier_threshold):
        super().__init__()
        self._tmpl_lines = tmpl_lines
        self._tmpl_arcs = tmpl_arcs
        self._det_lines = det_lines
        self._det_arcs = det_arcs
        self._pixel_size_mm = pixel_size_mm
        self._n_iterations = n_iterations
        self._inlier_threshold = inlier_threshold

    def run(self):
        try:
            result = ransac_rigid_registration(
                self._tmpl_lines, self._tmpl_arcs,
                self._det_lines, self._det_arcs,
                self._pixel_size_mm,
                n_iterations=self._n_iterations,
                inlier_threshold_mm=self._inlier_threshold,
            )
            if result is None:
                self.error.emit("Alignment failed: not enough inlier matches")
            else:
                self.done.emit(result)
        except Exception as e:
            self.error.emit(f"Alignment failed: {e}")


class AlignmentWindow(QWidget):
    """Floating window for alignment experiment playground."""

    align_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alignment Playground")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setMinimumSize(400, 300)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Status label
        self._status = QLabel("Need both template and detected features.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # RANSAC parameters
        param_group = QGroupBox("RANSAC Parameters")
        param_layout = QFormLayout(param_group)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(10, 1000)
        self._iter_spin.setValue(200)
        self._iter_spin.setToolTip("Number of RANSAC iterations")
        param_layout.addRow("Iterations:", self._iter_spin)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.5, 30.0)
        self._threshold_spin.setValue(5.0)
        self._threshold_spin.setSingleStep(0.5)
        self._threshold_spin.setSuffix(" mm")
        self._threshold_spin.setToolTip("Max distance for inlier classification")
        param_layout.addRow("Inlier threshold:", self._threshold_spin)

        layout.addWidget(param_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QFormLayout(results_group)

        self._rotation_label = QLabel("—")
        results_layout.addRow("Rotation:", self._rotation_label)

        self._translation_label = QLabel("—")
        results_layout.addRow("Translation:", self._translation_label)

        self._inliers_label = QLabel("—")
        results_layout.addRow("Inliers:", self._inliers_label)

        self._rms_label = QLabel("—")
        results_layout.addRow("Residual RMS:", self._rms_label)

        layout.addWidget(results_group)

        # Align button
        self._align_btn = QPushButton("Align")
        self._align_btn.clicked.connect(self._on_align_clicked)
        layout.addWidget(self._align_btn)

    def _on_align_clicked(self):
        self._align_btn.setEnabled(False)
        self._align_btn.setText("Computing...")
        self.align_requested.emit()

    def get_params(self) -> dict:
        return {
            'n_iterations': self._iter_spin.value(),
            'inlier_threshold_mm': self._threshold_spin.value(),
        }

    def show_results(self, result: AlignmentResult):
        t = result.transform
        self._rotation_label.setText(f"{t.rotation_deg:.2f} deg")
        self._translation_label.setText(
            f"({t.translation_mm[0]:.2f}, {t.translation_mm[1]:.2f}) mm")
        self._inliers_label.setText(f"{result.inlier_count}")
        self._rms_label.setText(f"{result.residual_rms_mm:.3f} mm")
        self.reset_button()

    def set_status(self, text: str):
        self._status.setText(text)

    def reset_button(self):
        self._align_btn.setEnabled(True)
        self._align_btn.setText("Align")


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
        self._batch_inspect_pending: bool = False
        self._redetect_pending: bool = False
        self._active_worker: QThread | None = None
        self._template_line_grid: dict | None = None
        self._template_arc_grid: dict | None = None
        self._template_lines: list | None = None
        self._template_arcs: list | None = None
        self._last_arclines_result: LinesArcsResult | None = None
        self._last_batch_results: list | None = None
        self._last_composited: np.ndarray | None = None
        self._show_grid_overlay: bool = False
        self._show_segments_overlay: bool = False

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
        self._image_label.setMouseTracking(True)
        self._image_label.mousePressEvent = self._on_image_click
        self._image_label.mouseMoveEvent = self._on_image_mouse_move
        self.setCentralWidget(self._image_label)

        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Live View", "Image Processing"])
        self._mode_combo.setMinimumWidth(150)
        toolbar.addWidget(self._mode_combo)

        # Toggle shortcut (works regardless of widget focus)
        self._toggle_shortcut = QShortcut(QKeySequence(Qt.Key_T), self)
        self._toggle_shortcut.setContext(Qt.ApplicationShortcut)

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

        self._grid_check = QCheckBox("Grid")
        self._grid_check.setToolTip("Show grid cell overlay for arc ID visualization")
        self._grid_check.setEnabled(False)  # Disabled until processing mode
        toolbar.addWidget(self._grid_check)

        self._segments_check = QCheckBox("Segments")
        self._segments_check.setToolTip("Show edge segment overlay for line ID visualization")
        self._segments_check.setEnabled(False)  # Disabled until processing mode
        toolbar.addWidget(self._segments_check)

        self._batch_btn = QPushButton("BatchInspect")
        self._batch_btn.setEnabled(False)
        toolbar.addWidget(self._batch_btn)

        self._alignment_btn = QPushButton("Alignment")
        self._alignment_btn.setEnabled(False)
        self._alignment_btn.setToolTip(
            "RANSAC rigid registration between template and product features")
        toolbar.addWidget(self._alignment_btn)

        # Status bar
        self._status_label = QLabel("No camera connected")
        self.statusBar().addWidget(self._status_label, 1)

        # Camera settings window (floating)
        self._settings_window = CameraSettingsWindow(self)

        # Detection results window (floating)
        self._results_window = ResultsWindow(self)

        # Feature pair measurement window (floating)
        self._pair_list_window = PairListWindow(self)

        # Batch inspect window (floating)
        self._batch_inspect_window = BatchInspectWindow(self)

        # Alignment playground window (floating)
        self._alignment_window = AlignmentWindow(self)

    # ── Signal wiring ───────────────────────────────────────────────

    def _connect_signals(self):
        # Camera → UI
        self._camera.signals.frame_ready.connect(self._on_live_frame)
        self._camera.signals.grab_done.connect(self._on_grab_frame)
        self._camera.signals.error.connect(self._on_camera_error)

        # UI → actions
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._toggle_shortcut.activated.connect(self._on_toggle_mode)
        self._grab_btn.clicked.connect(self._on_grab_clicked)
        self._reset_btn.clicked.connect(self._on_reset)
        self._save_btn.clicked.connect(self._on_save)
        self._calib_btn.clicked.connect(self._on_calib_clicked)
        self._arclines_btn.clicked.connect(self._on_arclines_clicked)
        self._batch_btn.clicked.connect(self._on_batch_inspect_clicked)
        self._alignment_btn.clicked.connect(self._on_alignment_clicked)
        self._batch_inspect_window.inspect_requested.connect(
            self._on_inspect_requested)
        self._batch_inspect_window.pair_row_selected.connect(
            self._on_batch_row_selected)
        self._alignment_window.align_requested.connect(
            self._on_alignment_run)

        # Settings → camera
        self._settings_window.settings_changed.connect(
            self._camera.apply_settings)

        # Results window → highlight, filter, and pair building
        self._results_window.item_selected.connect(
            self._on_result_item_selected)
        self._results_window.pair_added.connect(
            self._on_pair_added)
        self._results_window.filters_changed.connect(
            self._on_filters_changed)
        self._results_window.redetect_requested.connect(
            self._on_redetect_requested)

        # Debug overlay checkboxes
        self._grid_check.stateChanged.connect(self._on_grid_check_changed)
        self._segments_check.stateChanged.connect(self._on_segments_check_changed)

        # Pair list window → config save + measurement overlay
        self._pair_list_window.config_confirmed.connect(
            self._on_pair_config_confirmed)
        self._pair_list_window.pair_row_selected.connect(
            self._on_pair_row_selected)

    # ── Slots: camera frames ────────────────────────────────────────

    @Slot(np.ndarray)
    def _on_live_frame(self, frame: np.ndarray):
        """Display frame from continuous capture."""
        self._last_live_frame = frame.copy()
        self._display_image(frame)

    @Slot(np.ndarray)
    def _on_grab_frame(self, frame: np.ndarray):
        """Software-triggered frame received — route to calibration, batch inspect, or processing."""
        if self._calib_pending:
            self._calib_pending = False
            self._run_calibration(frame)
        elif self._batch_inspect_pending:
            self._batch_inspect_pending = False
            self._run_batch_inspect(frame)
        elif self._redetect_pending:
            self._redetect_pending = False
            self._run_redetect(frame)
        else:
            self._enter_processing_with_image(frame)

    @Slot(str)
    def _on_camera_error(self, msg: str):
        self._status_label.setText(f"Camera error: {msg}")
        self._batch_inspect_pending = False
        self._redetect_pending = False
        self._batch_inspect_window.reset_button()
        self._results_window.reset_redetect_button()

    # ── Slots: mode and buttons ─────────────────────────────────────

    @Slot()
    def _on_toggle_mode(self):
        """Toggle between Live View and Image Processing (keyboard shortcut)."""
        idx = self._mode_combo.currentIndex()
        self._mode_combo.setCurrentIndex(1 - idx)

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
        self._last_batch_results = None
        self._last_composited = None
        self._template_arc_grid = None
        self._template_line_grid = None
        self._template_lines = None
        self._template_arcs = None
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
    def _on_redetect_requested(self):
        """User pressed Redetect — grab fresh frame and re-run detection."""
        if self._current_mode != "processing":
            self._mode_combo.setCurrentIndex(1)
        self._status_label.setText("Redetect: capturing fresh frame...")
        self._redetect_pending = True
        self._camera.software_trigger()

    def _run_redetect(self, frame: np.ndarray):
        """Run detection pipeline on a freshly grabbed frame."""
        self._enter_processing_with_image(frame)
        self._last_arclines_result = None
        self._on_arclines_clicked()

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
        px_size = float(result.pixel_size_mm)
        self._processor.pixel_size = px_size
        self._config['processing']['pixel_size'] = px_size
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

        # Store template features for alignment playground
        if self._template_lines is None:
            self._template_lines = result.lines
            self._template_arcs = result.arcs

        self._results_window.update_results(result)
        self._results_window.show()
        self._pair_list_window.show()

        # Re-measure stored pairs with new geometry (fuzzy matching)
        self._pair_list_window.update_distances(
            result.lines, result.arcs, self._processor.pixel_size,
            self._proc_state.image_bgr, self._config)

        self._render_arclines()

        n_lines = len(result.lines)
        n_arcs = len(result.arcs)
        self._status_label.setText(
            f"Detected {n_lines} lines, {n_arcs} arcs")
        self._results_window.reset_redetect_button()

    @Slot(str)
    def _on_worker_error(self, msg: str):
        """Handle errors from background workers."""
        self._active_worker = None
        self._status_label.setText(f"Error: {msg}")
        self._results_window.reset_redetect_button()
        self._alignment_window.reset_button()

    # ── Batch inspect ────────────────────────────────────────────────

    @Slot()
    def _on_batch_inspect_clicked(self):
        """Open batch inspect window."""
        configs = {}
        for cfg in self._pair_list_window._all_configs:
            configs[cfg['name']] = cfg['pairs']
        if not configs:
            self._status_label.setText("No templates saved — create one first")
            return
        self._batch_inspect_window.set_config_names(configs)
        self._batch_inspect_window.show()

    def _on_inspect_requested(self):
        """User pressed Inspect — switch to processing mode, grab frame and run inspection."""
        # Refresh template data in case new pairs were added
        configs = {}
        for cfg in self._pair_list_window._all_configs:
            configs[cfg['name']] = cfg['pairs']
        self._batch_inspect_window.set_config_names(configs)

        # Auto-switch to Image Processing mode if in Live View
        if self._current_mode != "processing":
            self._mode_combo.setCurrentIndex(1)

        self._status_label.setText("Batch inspect: capturing frame...")
        self._batch_inspect_pending = True
        self._camera.software_trigger()

    def _run_batch_inspect(self, frame: np.ndarray):
        """Run batch inspection on grabbed frame."""
        # Store the frame as processing state so we can overlay results
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        state = ProcessingState()
        state.image_bgr = frame.copy()
        state.image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._proc_state = state
        self._last_arclines_result = None  # clear old arclines overlay

        template_name = self._batch_inspect_window.get_selected_template()
        template_pairs = self._pair_list_window.get_pairs_for_config(template_name)
        if not template_pairs:
            self._status_label.setText(
                f"No pairs in template '{template_name}'")
            self._batch_inspect_window.reset_button()
            return

        self._status_label.setText("Running batch inspection...")

        proc_cfg = self._config.get('processing', {})
        det_cfg = self._config.get('detection', {})
        worker = BatchInspectWorker(
            frame,
            template_pairs,
            pixel_size_mm=self._processor.pixel_size,
            gauss_sigma=proc_cfg.get('gauss_sigma', 0.4),
            morph_radius=proc_cfg.get('morph_radius', 3),
            grid_size_mm=det_cfg.get('grid_size_mm', 5.0),
            segment_mm=det_cfg.get('edge_segment_mm', 10.0),
            line_min_mm=det_cfg.get('line_min_mm', 0.0),
            line_max_mm=det_cfg.get('line_max_mm', float('inf')),
            arc_min_mm=det_cfg.get('arc_min_mm', 0.0),
            arc_max_mm=det_cfg.get('arc_max_mm', float('inf')),
        )
        worker.done.connect(self._on_batch_inspect_done)
        worker.error.connect(self._on_batch_inspect_error)
        self._active_worker = worker
        worker.start()

    @Slot(list)
    def _on_batch_inspect_done(self, results: list):
        """Batch inspection finished — show results and overlay on frame."""
        self._active_worker = None
        self._batch_inspect_window.show_results(results)

        # Store results for row-click annotation
        self._last_batch_results = results

        # Render feature geometry + measurement overlays on current frame
        if self._proc_state and self._proc_state.image_bgr is not None:
            from detect_lines import (render_measurement_overlay, LineResult,
                                      ArcResult, ARC_PALETTE)
            canvas = self._proc_state.image_bgr.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX

            for idx, (dist, pair, pt_a, pt_b, det_a, det_b) in enumerate(results):
                if det_a is None and det_b is None:
                    continue
                pair_color = ARC_PALETTE[idx % len(ARC_PALETTE)]

                # Dim color for pass/fail status tinting
                if dist is not None and not (pair.lower_mm == 0 and pair.upper_mm == 0):
                    if pair.lower_mm <= dist <= pair.upper_mm:
                        # Pass: keep pair color as-is
                        pass
                    else:
                        # Fail: shift toward red
                        pair_color = (
                            min(255, pair_color[0] + 100),
                            max(0, pair_color[1] - 80),
                            max(0, pair_color[2] - 80),
                        )

                for det in (det_a, det_b):
                    if det is None:
                        continue
                    if isinstance(det, LineResult):
                        p1 = det.start_px.astype(int)
                        p2 = det.end_px.astype(int)
                        cv2.line(canvas, tuple(p1), tuple(p2), pair_color, 2, cv2.LINE_AA)
                        cv2.circle(canvas, tuple(p1), 4, pair_color, -1)
                        cv2.circle(canvas, tuple(p2), 4, pair_color, -1)
                    elif isinstance(det, ArcResult):
                        cx, cy = int(det.center_px[0]), int(det.center_px[1])
                        r = int(det.radius_px)
                        cv2.circle(canvas, (cx, cy), r, pair_color, 2, cv2.LINE_AA)
                        cv2.circle(canvas, (cx, cy), 3, pair_color, -1)

            # Draw measurement lines on top
            for idx, (dist, pair, pt_a, pt_b, det_a, det_b) in enumerate(results):
                if dist is not None and pt_a is not None and pt_b is not None:
                    pair_color = ARC_PALETTE[idx % len(ARC_PALETTE)]
                    if dist is not None and not (pair.lower_mm == 0 and pair.upper_mm == 0):
                        if pair.lower_mm <= dist <= pair.upper_mm:
                            pass
                        else:
                            pair_color = (
                                min(255, pair_color[0] + 100),
                                max(0, pair_color[1] - 80),
                                max(0, pair_color[2] - 80),
                            )
                    render_measurement_overlay(canvas, pt_a, pt_b, dist, pair_color)
            self._show_composited(canvas)

        n_pass = sum(1 for dist, pair, _, _, _, _ in results
                     if dist is not None
                     and not (pair.lower_mm == 0 and pair.upper_mm == 0)
                     and pair.lower_mm <= dist <= pair.upper_mm)
        n_total = len(results)
        n_measured = sum(1 for dist, _, _, _, _, _ in results if dist is not None)
        n_fail = sum(1 for dist, pair, _, _, _, _ in results
                     if dist is not None
                     and not (pair.lower_mm == 0 and pair.upper_mm == 0)
                     and not (pair.lower_mm <= dist <= pair.upper_mm))
        self._status_label.setText(
            f"Batch inspect done: {n_measured}/{n_total} measured, "
            f"{n_pass} pass, {n_fail} fail")

    @Slot(str)
    def _on_batch_inspect_error(self, msg: str):
        """Batch inspection failed."""
        self._active_worker = None
        self._status_label.setText(f"Batch inspect error: {msg}")
        self._batch_inspect_window.reset_button()

    @Slot(int)
    def _on_batch_row_selected(self, row: int):
        """Batch inspect table row selected — draw red arrow annotation."""
        if row < 0 or self._last_batch_results is None:
            # Re-render batch overlay without arrow
            if self._last_batch_results is not None:
                self._on_batch_inspect_done(self._last_batch_results)
            return
        if row >= len(self._last_batch_results):
            return

        dist, pair, pt_a, pt_b, det_a, det_b = self._last_batch_results[row]
        if dist is None or pt_a is None or pt_b is None:
            return

        # Re-render the full batch overlay first
        from detect_lines import ARC_PALETTE
        self._on_batch_inspect_done(self._last_batch_results)

        # Now draw the red arrow perpendicular to the measurement line at midpoint
        if self._proc_state and self._proc_state.image_bgr is not None:
            canvas = self._proc_state.image_bgr.copy()
            # Re-render batch overlay onto this fresh canvas
            from detect_lines import (render_measurement_overlay, LineResult, ArcResult)
            for idx, (d, p, pa, pb, da, db) in enumerate(self._last_batch_results):
                pc = ARC_PALETTE[idx % len(ARC_PALETTE)]
                if d is not None and not (p.lower_mm == 0 and p.upper_mm == 0):
                    if not (p.lower_mm <= d <= p.upper_mm):
                        pc = (min(255, pc[0] + 100), max(0, pc[1] - 80), max(0, pc[2] - 80))
                for det in (da, db):
                    if det is None:
                        continue
                    if isinstance(det, LineResult):
                        cv2.line(canvas, tuple(det.start_px.astype(int)),
                                 tuple(det.end_px.astype(int)), pc, 2, cv2.LINE_AA)
                    elif isinstance(det, ArcResult):
                        cv2.circle(canvas, (int(det.center_px[0]), int(det.center_px[1])),
                                   int(det.radius_px), pc, 2, cv2.LINE_AA)
                if d is not None and pa is not None and pb is not None:
                    render_measurement_overlay(canvas, pa, pb, d, pc)

            # Draw red arrow at midpoint perpendicular to measurement line
            pt_a_int = pt_a.astype(int)
            pt_b_int = pt_b.astype(int)
            mid = ((pt_a_int + pt_b_int) // 2).astype(int)
            direction = (pt_b_int - pt_a_int).astype(float)
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
            perp = np.array([-direction[1], direction[0]])
            arrow_start = (mid + (perp * 40)).astype(int)
            cv2.arrowedLine(canvas, tuple(arrow_start), tuple(mid),
                            (0, 0, 255), 3, tipLength=0.4)
            self._show_composited(canvas)

    # ── Alignment playground ────────────────────────────────────────────

    @Slot()
    def _on_alignment_clicked(self):
        """Show alignment playground window."""
        # Check if we have both template and current features
        has_template = self._template_lines is not None or self._template_arcs is not None
        has_current = self._last_arclines_result is not None
        if has_template and has_current:
            self._alignment_window.set_status(
                f"Template: {len(self._template_lines or [])} lines, "
                f"{len(self._template_arcs or [])} arcs. "
                f"Current: {len(self._last_arclines_result.lines)} lines, "
                f"{len(self._last_arclines_result.arcs)} arcs.")
        elif has_template and not has_current:
            self._alignment_window.set_status(
                "Template features stored. Run Arclines on current frame first.")
        elif not has_template and has_current:
            self._alignment_window.set_status(
                "Current features detected. Need template — "
                "run Arclines once to store template, then Redetect for current.")
        else:
            self._alignment_window.set_status(
                "Need both template and detected features. "
                "Run Arclines, then Redetect to get a second frame.")
        self._alignment_window.show()

    @Slot()
    def _on_alignment_run(self):
        """Run RANSAC alignment between template and current features."""
        if self._template_lines is None and self._template_arcs is None:
            self._alignment_window.set_status("No template features stored.")
            self._alignment_window.reset_button()
            return
        if self._last_arclines_result is None:
            self._alignment_window.set_status("No current detection results.")
            self._alignment_window.reset_button()
            return

        params = self._alignment_window.get_params()
        worker = AlignmentWorker(
            self._template_lines or [],
            self._template_arcs or [],
            self._last_arclines_result.lines,
            self._last_arclines_result.arcs,
            self._processor.pixel_size,
            params['n_iterations'],
            params['inlier_threshold_mm'],
        )
        worker.done.connect(self._on_alignment_done)
        worker.error.connect(self._on_alignment_error)
        self._active_worker = worker
        worker.start()

    def _on_alignment_done(self, result: AlignmentResult):
        """Alignment completed — show results and render overlay."""
        self._active_worker = None
        self._alignment_window.show_results(result)

        if self._proc_state and self._proc_state.image_bgr is not None:
            overlay = render_alignment_overlay(
                self._proc_state.image_bgr, result,
                self._template_lines or [], self._template_arcs or [],
                self._last_arclines_result.lines,
                self._last_arclines_result.arcs,
                self._processor.pixel_size,
            )
            self._show_composited(overlay)

    def _on_alignment_error(self, msg: str):
        """Alignment failed."""
        self._active_worker = None
        self._alignment_window.set_status(f"Error: {msg}")
        self._alignment_window.reset_button()

    @Slot(str, str, str, str)
    def _on_pair_added(self, type_a: str, id_a: str, type_b: str, id_b: str):
        """Two features Ctrl-selected — compute distance and add to pair list."""
        result = self._last_arclines_result
        if result is None:
            return
        dist = compute_feature_distance(
            type_a, id_a, type_b, id_b,
            result.lines, result.arcs, self._processor.pixel_size)
        if dist is None:
            self._status_label.setText(
                f"Cannot locate features {id_a} / {id_b}")
            return
        pair = FeaturePair(
            type_a=type_a, id_a=id_a,
            type_b=type_b, id_b=id_b,
            distance_mm=dist,
        )
        self._pair_list_window.add_pair(pair)
        self._pair_list_window.show()
        self._status_label.setText(
            f"Pair added: {id_a} ↔ {id_b} = {dist:.3f} mm")

    def _on_pair_config_confirmed(self, configs: list):
        """User confirmed pair configuration — save all configs to config.yaml."""
        self._config['feature_pairs'] = configs
        config_path = os.path.join(_app_dir(), 'config.yaml')
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False,
                          sort_keys=False)
            self._status_label.setText(
                f"Saved {len(configs)} configuration(s)")
        except Exception as e:
            self._status_label.setText(f"Save failed: {e}")

    @Slot(int)
    def _on_pair_row_selected(self, row: int):
        """Pair table row selected — render measurement overlay."""
        if row < 0:
            self._render_arclines()
            return

        # Use fuzzy match results if available (preferred over exact lookup)
        matches = self._pair_list_window._last_matches
        if matches is not None and 0 <= row < len(matches):
            dist, pair, pt_a, pt_b, det_a, det_b = matches[row]
            if dist is not None and pt_a is not None and pt_b is not None:
                self._render_arclines(
                    measurement_points=(dist, pt_a, pt_b))
                return

        # Fallback to exact lookup (for pairs added before any detection run)
        pair = self._pair_list_window.get_pair(row)
        if pair is None or self._last_arclines_result is None:
            self._render_arclines()
            return

        result = self._last_arclines_result
        pts = compute_feature_pair_points(
            pair.type_a, pair.id_a, pair.type_b, pair.id_b,
            result.lines, result.arcs, self._processor.pixel_size)
        if pts is None:
            self._render_arclines()
            return

        self._render_arclines(measurement_points=pts)

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

    def _render_arclines(self, highlight_type="", highlight_id="",
                         measurement_points=None):
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
            highlight_type=highlight_type, highlight_id=highlight_id,
            measurement_points=measurement_points)
        self._show_composited(annotated)

    @Slot(int)
    def _on_grid_check_changed(self, state: int):
        """Toggle grid overlay on top of current composited image."""
        self._show_grid_overlay = bool(state)
        if self._last_composited is not None:
            self._show_composited(self._last_composited)

    @Slot(int)
    def _on_segments_check_changed(self, state: int):
        """Toggle segments overlay on top of current composited image."""
        self._show_segments_overlay = bool(state)
        if self._last_composited is not None:
            self._show_composited(self._last_composited)

    # ── Mode transitions ────────────────────────────────────────────

    def _switch_to_live(self):
        """Switch to continuous capture mode."""
        self._current_mode = "live"
        self._grab_btn.setEnabled(False)
        self._reset_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._calib_btn.setEnabled(False)
        self._arclines_btn.setEnabled(False)
        self._batch_btn.setEnabled(False)
        self._alignment_btn.setEnabled(False)
        self._grid_check.setEnabled(False)
        self._segments_check.setEnabled(False)
        self._image_label.setCursor(Qt.ArrowCursor)
        self._proc_state = None
        self._last_composited = None
        self._last_arclines_result = None
        self._last_batch_results = None
        self._template_arc_grid = None
        self._template_line_grid = None
        self._template_lines = None
        self._template_arcs = None
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
        self._batch_btn.setEnabled(True)
        self._alignment_btn.setEnabled(True)
        self._grid_check.setEnabled(True)
        self._segments_check.setEnabled(True)
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

        # Old watershed object-picking replaced by arclines-based measurement.
        # Clicks are no longer used for distance picking.

    def _on_image_mouse_move(self, event):
        """Handle mouse hover — show grid cell / edge segment info in status bar."""
        if self._current_mode != "processing":
            return
        if self._proc_state is None or self._proc_state.image_bgr is None:
            return

        ix, iy = self._label_to_image(event.position().toPoint())
        if ix is None or iy is None:
            return

        img_h, img_w = self._proc_state.image_bgr.shape[:2]
        det_cfg = self._config.get('detection', {})
        pixel_size_mm = self._processor.pixel_size

        parts = []

        # Grid cell
        grid_size_mm = det_cfg.get('grid_size_mm', 5.0)
        row, col = compute_grid_cell(ix, iy, pixel_size_mm, grid_size_mm)
        parts.append(f"Cell: C{row}_{col}")

        # Edge segments (always show when in processing mode)
        segment_mm = det_cfg.get('edge_segment_mm', 10.0)
        segs = compute_edge_segments(ix, iy, img_w, img_h, pixel_size_mm, segment_mm)
        parts.append(f"Edges: Up{segs['Up']}/Lo{segs['Lo']}/Le{segs['Le']}/Ri{segs['Ri']}")

        # Pixel position in mm
        x_mm = ix * pixel_size_mm
        y_mm = iy * pixel_size_mm
        parts.append(f"({x_mm:.2f}, {y_mm:.2f}) mm")

        self._status_label.setText("  |  ".join(parts))

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

    def _show_composited(self, bgr_frame: np.ndarray):
        """Store composited image (without debug overlays) and display it.

        All rendering paths that produce a composited result (watershed,
        arclines, batch inspect, measurement overlay) should call this instead
        of _display_image directly.  Debug overlays (grid/segments) are applied
        on top when enabled, without modifying the stored composited image.
        """
        self._last_composited = bgr_frame.copy()

        if self._show_grid_overlay or self._show_segments_overlay:
            canvas = bgr_frame.copy()
            det_cfg = self._config.get('detection', {})
            pixel_size_mm = self._processor.pixel_size
            if self._show_grid_overlay:
                canvas = draw_grid_overlay(
                    canvas, pixel_size_mm, det_cfg.get('grid_size_mm', 5.0))
            if self._show_segments_overlay:
                canvas = draw_edge_segments(
                    canvas, pixel_size_mm, det_cfg.get('edge_segment_mm', 10.0))
            self._display_image(canvas)
        else:
            self._display_image(bgr_frame)

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
        self._show_composited(composited)

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

        # Save feature pair configuration
        self._config['feature_pairs'] = self._pair_list_window.get_config_list()

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

    # Load stored feature pair configuration
    window._pair_list_window.load_config(
        config.get('feature_pairs', []))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
