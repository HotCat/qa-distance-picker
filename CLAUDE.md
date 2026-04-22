# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine-vision quality assurance application for measuring distances between objects on manufactured workpieces. Uses DIPlib watershed segmentation and a MindVision industrial camera. Built with PySide6 for the UI, with a decoupled architecture separating camera control, image processing, and presentation.

## Running the Application

```bash
python app.py
```

Requires: `PySide6`, `diplib`, `opencv-python`, `numpy`, `pyyaml`, `pillow`, `scipy`. The camera SDK (`driver/mvsdk.py`) is bundled — no separate install needed. A MindVision USB/GigE camera must be connected for live view. Requires `libxcb-cursor0` system package on Linux.

## Architecture

The application has three decoupled modules with clear boundaries:

### `app.py` — PySide6 UI + main loop

- `MainWindow`: toolbar (Live View / Image Processing combo, Grab, Reset, Save, Calibration, Arclines), central `QLabel` for image display, status bar
- `CameraSettingsWindow`: floating widget (`Qt.Tool | Qt.WindowStaysOnTopHint`) with sliders for exposure, gamma, contrast, analog gain, auto-exposure toggle, and ReverseX/ReverseY mirror checkboxes
- `ProcessingState`: mutable state for the current processing image (BGR/RGB arrays, segmentation result, selected object IDs, click positions, distance result)
- Mode toggle: **Live View** (continuous capture, `trigger=0`) ↔ **Image Processing** (software trigger, `trigger=1`)
- Object picking state machine: `PICK_OBJECT1` → `PICK_OBJECT2` → `SHOW_RESULT`
- Coordinate mapping: `_label_to_image()` reverse-maps QLabel click coordinates to original image pixels accounting for aspect-ratio scaling and centering
- Config persistence: `_app_dir()` resolves the base directory for config.yaml (exe dir when frozen, source dir otherwise). On first run from a frozen build, the bundled default is extracted from `sys._MEIPASS`. Settings are saved to config.yaml on exit via `closeEvent()`
- Calibration: `_calib_pending` flag intercepts `grab_done` signal to route grabbed frames to `CalibrationWorker` instead of normal processing. On success, updates `WatershedProcessor.pixel_size` and writes new value to config
- Arclines: runs `LinesArcsWorker` on the current processing image asynchronously, displays annotated BGR result. Line IDs use edge-intersection-based assignment. Arc IDs use grid-cell-based assignment. Shows results in `ResultsWindow` floating tree view
- `ResultsWindow`: floating `QTreeWidget` with Lines/Curves top-level items. ExtendedSelection mode — single-click highlights a feature, Ctrl+click two features adds a pair. Filter bar with QDoubleSpinBox controls for line length range (min/max mm) and arc radius range (min/max mm). Emits `item_selected(str, str)` for single highlight, `pair_added(str, str, str, str)` when 2 features are Ctrl-selected. Filter values saved to `config.yaml` under `detection` section. Feature A/B columns use `Interactive` resize mode so users can drag to adjust widths
- `PairListWindow`: floating window for batch metrology. Shows `QTableWidget` with columns: sequence #, Feature A, Feature B, Distance (mm), Lower bound (mm), Upper bound (mm). Config name `QLineEdit` and Confirm button to save pair configuration to `config.yaml` under `feature_pairs` section. Distances recomputed on each Arclines run using `compute_feature_distance`. Lower/upper bound cells are user-editable. Delete Row button removes selected pair. Emits `pair_row_selected(int)` when a row is clicked, triggering measurement overlay rendering on the current frame (red line connecting the two features with mm label)
- Image click handling: when arclines results are displayed (`_last_arclines_result is not None`), clicking the image is disabled to prevent interference with the old object-picking state machine

### `camera.py` — Camera abstraction (MindVision SDK wrapper)

- `MindVisionCamera`: owns SDK handle (`hCamera`), frame buffer, and live view thread
- `_LiveViewThread(QThread)`: polls `CameraGetImageBuffer` in a loop, emits `frame_ready` signal with numpy BGR frames
- Mode switching: `set_live_mode()` (trigger=0, starts thread), `set_trigger_mode()` (stops thread, trigger=1)
- `software_trigger()`: fires `CameraSoftTrigger` + `CameraGetImageBuffer`, emits `grab_done` signal
- Settings: `apply_settings(CameraSettings)` / `get_current_settings()` / `get_setting_ranges()` — includes mirror via `CameraSetMirror(hCamera, dir, enable)` where dir=0 is horizontal, dir=1 is vertical
- Signal emitter pattern: `CameraSignalEmitter(QObject)` owns `frame_ready`, `grab_done`, `error` signals — camera class itself is not a QObject
- Key SDK sequence for mode switch: `CameraStop` → `CameraSetTriggerMode` → `CameraPlay`

### `processing.py` — Watershed segmentation + distance computation (no Qt dependency)

- `WatershedProcessor`:
  - `segment(image_bgr)`: DIPlib pipeline — `Gauss(0.4)` → `Convert('grey')` → `GradientMagnitude` → `Norm` → `Closing(3)/Opening(3)` → `Watershed(connectivity=1, maxDepth=3)` → label array + region sizes dict
  - `compute_distance(image_rgb, labels, dip_grey, id1, id2)`: closest boundary points (primary) + DIPlib distance transform (cross-check) + 1D gradient profile with sub-pixel peak refinement (cross-check)
  - All DIPlib images created from numpy must call `SetColorSpace('sRGB')` before `ColorSpaceManager.Convert(img, 'grey')`
- `OverlayRenderer.render()`: composites mask tinting, contour drawing, click markers, measurement line, distance text with perpendicular offset, HUD bar

### `config.yaml` — Parameters

- `camera`: exposure, gamma, contrast, analog gain, AE state, reverse_x (horizontal mirror), reverse_y (vertical mirror) — saved on exit, loaded on startup
- `processing`: pixel_size (updated by calibration), gauss_sigma, morph_radius, min_region_size (500 px), watershed connectivity/max_depth
- `calibration`: board_cols (inner corners horizontal), board_rows (inner corners vertical), grid_size_mm (physical grid square size)
- `detection`: line_min_mm, line_max_mm (line length filter range), arc_min_mm, arc_max_mm (arc radius filter range), grid_size_mm (grid cell size for arc ID assignment), edge_segment_mm (segment length along image edges for line ID assignment)
- `feature_pairs`: list of saved configurations, each with `name` and `pairs` (list of dicts with type_a, id_a, type_b, id_b, distance_mm, lower_mm, upper_mm)

### `calibration.py` — Chessboard pixel-size calibration

- `calibrate_pixel_size(image_bgr, board_cols, board_rows, grid_size_mm)`: single-call chessboard detection using configured dimensions (not brute-force over all sizes). Uses `cv2.findChessboardCorners` + `cornerSubPix`. Returns `CalibrationResult` with `pixel_size_mm`, or `None` if pattern not found
- `CalibrationWorker(QThread)`: runs `calibrate_pixel_size` in background thread, emits `done(CalibrationResult)` or `error(str)`
- **Why not brute-force**: The reference `calibrate_pixel_size` iterated cols=3..19, rows=3..19 (289 expensive calls). Using the configured board size directly reduces to 1 call — faster and validates that the physical chessboard matches config

### `detect_lines.py` — Line and arc detection with stable ID mapping

- `detect_lines_and_arcs(image_bgr, pixel_size_mm, gauss_sigma, morph_radius, grid_size_mm, template_arc_grid, edge_segment_mm, template_line_grid)`: full pipeline — DIPlib watershed → LSD line detection (`detect_lines`) + collinear merge → curvature-based arc detection (`detect_arcs_for_object`) per watershed object → `deduplicate_cross_object_arcs` → convert to mm → stable ID assignment → annotated BGR image
- Line ID matching: `_assign_line_ids_by_edges` uses edge-intersection-based assignment. Each line is extended to intersect two image frame edges. The ID encodes which edges and which segments it crosses plus the line length: `L_{edge1}{seg1}_{edge2}{seg2}_{angle}_{length_mm}` (e.g. `L_Up23_Le12_5.3_86.07`). Edges ordered as `['Up', 'Lo', 'Le', 'Ri']`. Segment length configurable via `edge_segment_mm`. Length included in grid key so parallel lines with different lengths get unique IDs. On subsequent runs, template matching inherits IDs when same (edge1, seg1, edge2, seg2, length_bucket) key matches with angle within ±15°.
- Arc ID matching: `_assign_arc_ids_by_grid` uses grid-cell-based assignment. Key = (grid_row, grid_col, int(radius_mm)). Arcs in the same cell with similar radius on subsequent runs inherit the template ID. Configurable grid size (default 5mm). Arcs get IDs like `C5_22_2`
- Helper functions: `detect_lines` (LSD + merge), `merge_collinear_lines`, `detect_arcs_for_object` (contour → curvature → RANSAC), `deduplicate_cross_object_arcs`, `compute_curvature`, `extract_arc_region`, `ransac_fit_circle`, `solve_circle_from_3_points`, `_extend_line_to_edges`, `_segment_number`, `_assign_line_ids_by_edges`, `_estimate_alignment`, `match_features`
- Distance computation: `distance_line_to_line` (perpendicular from center of line A to infinite extension of line B), `distance_arc_to_arc` (center-to-center), `distance_line_to_arc` (perpendicular from arc center A to infinite extension of line B), `compute_feature_distance` (dispatch by feature type, returns float only), `compute_feature_pair_points` (dispatch by feature type, returns distance + geometry points for rendering). Helper `_perpendicular_foot_to_line` projects a point onto an infinite line through two endpoints (not clamped to segment)
- Measurement overlay: `render_measurement_overlay` draws a red line between two pixel points with filled circle endpoints and a mm label offset perpendicular to the line. Called from `render_annotations` when `measurement_points` parameter is provided
- Data types: `LineResult` (id, category H/V/D/O, endpoints, length_mm, angle_deg, centroid_mm), `ArcResult` (id, center, radius, centroid_mm), `LinesArcsResult` (includes `arc_grid` and `line_grid` dicts for template), `FeatureDescriptor`, `FeaturePair` (type_a, id_a, type_b, id_b, distance_mm, lower_mm, upper_mm)
- `LinesArcsWorker(QThread)`: runs `detect_lines_and_arcs` in background thread, emits `done(LinesArcsResult)` or `error(str)`

### `driver/` — MindVision camera SDK

- `mvsdk.py`: full Python ctypes bindings for the MindVision SDK. On Linux loads `libMVSDK.so` via `ctypes.cdll.LoadLibrary`; when running from a PyInstaller bundle, resolves the library from `sys._MEIPASS` instead of relying on system `LD_LIBRARY_PATH`
- `cv_grab.py`: reference sample showing continuous capture with OpenCV display

### PyInstaller packaging

- `qa-distance-picker.spec`: PyInstaller spec file for one-directory build. Bundles `libMVSDK.so` (from `/usr/lib/`), `config.yaml` (as default), diplib native `.so` files. Excludes torch/sam2/open3d/matplotlib. Hidden imports for `mvsdk`, `diplib`, `scipy.*`, PySide6 submodules
- `build.sh`: reusable build script — installs PyInstaller, cleans previous build, runs `pyinstaller qa-distance-picker.spec --noconfirm`
- Output: `dist/qa-distance-picker/` (~580 MB) — copy the entire folder to any Linux machine and run `./qa-distance-picker`
- `camera.py` skips `sys.path.insert` for `driver/` when frozen (mvsdk found via hidden-import); `mvsdk.py` resolves `libMVSDK.so` from `sys._MEIPASS` when frozen; `app.py` uses `_app_dir()` to find config.yaml next to the executable

## Key Design Decisions

- **DIPlib color space**: Images created from numpy arrays must have `SetColorSpace('sRGB')` set — `dip.ImageRead` does this automatically, but `dip.Image(numpy_array)` does not. Without it, `ColorSpaceManager.Convert(img, 'grey')` fails or produces different results.
- **No SAM2**: SAM2 was removed because it over-segments, extending masks into gaps between objects, making boundary-based distance measurements unreliable (0.117 mm instead of ~6 mm). Watershed boundaries are precise and non-overlapping.
- **Cris Luengo method rejected**: The StackOverflow method (Gravity/GreyMajorAxes) measures perpendicular distance between entire edge regions, not facing edges — gives 25–35 mm instead of 6–8 mm. Closest boundary points is the primary method instead.
- **Live view thread**: Subclasses `QThread` directly (not worker+moveToThread pattern) because `run_loop` is a blocking poll loop that doesn't use the Qt event loop.
- **Exposure time units**: `CameraGetExposureTimeRange` and `CameraGetExposureTime` return **microseconds** directly (not seconds as the name might suggest).
- **Camera mirror**: `CameraSetMirror(hCamera, direction, enable)` where direction=0 is `MIRROR_DIRECTION_HORIZONTAL` (reverse_x) and direction=1 is `MIRROR_DIRECTION_VERTICAL` (reverse_y). Read back with `CameraGetMirror(hCamera, direction)`.
- **PyInstaller one-directory build**: A one-file build (`-F`) would decompress 500+ MB to a temp dir on every launch — slow startup and no writable config.yaml location. One-directory build allows config.yaml to live alongside the executable and be read/written persistently.
- **Frozen detection**: `getattr(sys, 'frozen', False)` is True inside a PyInstaller bundle. `sys._MEIPASS` points to the `_internal/` directory containing bundled data/binaries. `_app_dir()` returns the executable's directory (for writable config.yaml), not `_MEIPASS` (which is read-only inside the bundle).
- **Calibration validates board dimensions**: `calibrate_pixel_size` uses the configured `board_cols`×`board_rows` directly — a single `findChessboardCorners` call. If the chessboard doesn't match, it fails explicitly rather than silently detecting a different size. This prevents accidental miscalibration from a wrong board.
- **Async workers for long computations**: `CalibrationWorker` and `LinesArcsWorker` subclass `QThread` and emit `done`/`error` signals, keeping the UI responsive during watershed (~300ms) and line/arc detection (~2s).

## Distance Measurement Accuracy

Verified on `screenshot2.png` with known object pairs:
- ID 2773 ↔ 2677: **7.958 mm** (expected ~8.00 mm)
- ID 9359 ↔ 2677: **5.970 mm** (expected ~6.00 mm)
