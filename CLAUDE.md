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

- `MainWindow`: toolbar (Live View / Image Processing combo, Grab, Reset, Save), central `QLabel` for image display, status bar
- `CameraSettingsWindow`: floating widget (`Qt.Tool | Qt.WindowStaysOnTopHint`) with sliders for exposure, gamma, contrast, analog gain, and auto-exposure toggle
- `ProcessingState`: mutable state for the current processing image (BGR/RGB arrays, segmentation result, selected object IDs, click positions, distance result)
- Mode toggle: **Live View** (continuous capture, `trigger=0`) ↔ **Image Processing** (software trigger, `trigger=1`)
- Object picking state machine: `PICK_OBJECT1` → `PICK_OBJECT2` → `SHOW_RESULT`
- Coordinate mapping: `_label_to_image()` reverse-maps QLabel click coordinates to original image pixels accounting for aspect-ratio scaling and centering

### `camera.py` — Camera abstraction (MindVision SDK wrapper)

- `MindVisionCamera`: owns SDK handle (`hCamera`), frame buffer, and live view thread
- `_LiveViewThread(QThread)`: polls `CameraGetImageBuffer` in a loop, emits `frame_ready` signal with numpy BGR frames
- Mode switching: `set_live_mode()` (trigger=0, starts thread), `set_trigger_mode()` (stops thread, trigger=1)
- `software_trigger()`: fires `CameraSoftTrigger` + `CameraGetImageBuffer`, emits `grab_done` signal
- Settings: `apply_settings(CameraSettings)` / `get_current_settings()` / `get_setting_ranges()`
- Signal emitter pattern: `CameraSignalEmitter(QObject)` owns `frame_ready`, `grab_done`, `error` signals — camera class itself is not a QObject
- Key SDK sequence for mode switch: `CameraStop` → `CameraSetTriggerMode` → `CameraPlay`

### `processing.py` — Watershed segmentation + distance computation (no Qt dependency)

- `WatershedProcessor`:
  - `segment(image_bgr)`: DIPlib pipeline — `Gauss(0.4)` → `Convert('grey')` → `GradientMagnitude` → `Norm` → `Closing(3)/Opening(3)` → `Watershed(connectivity=1, maxDepth=3)` → label array + region sizes dict
  - `compute_distance(image_rgb, labels, dip_grey, id1, id2)`: closest boundary points (primary) + DIPlib distance transform (cross-check) + 1D gradient profile with sub-pixel peak refinement (cross-check)
  - All DIPlib images created from numpy must call `SetColorSpace('sRGB')` before `ColorSpaceManager.Convert(img, 'grey')`
- `OverlayRenderer.render()`: composites mask tinting, contour drawing, click markers, measurement line, distance text with perpendicular offset, HUD bar

### `config.yaml` — Parameters

- `camera`: default exposure (30000 us), gamma, contrast, analog gain, AE state
- `processing`: pixel_size (0.117027 mm/px), gauss_sigma, morph_radius, min_region_size (500 px), watershed connectivity/max_depth

### `driver/` — MindVision camera SDK

- `mvsdk.py`: full Python ctypes bindings for the MindVision SDK (`libMVSDK.so` on Linux)
- `cv_grab.py`: reference sample showing continuous capture with OpenCV display

## Key Design Decisions

- **DIPlib color space**: Images created from numpy arrays must have `SetColorSpace('sRGB')` set — `dip.ImageRead` does this automatically, but `dip.Image(numpy_array)` does not. Without it, `ColorSpaceManager.Convert(img, 'grey')` fails or produces different results.
- **No SAM2**: SAM2 was removed because it over-segments, extending masks into gaps between objects, making boundary-based distance measurements unreliable (0.117 mm instead of ~6 mm). Watershed boundaries are precise and non-overlapping.
- **Cris Luengo method rejected**: The StackOverflow method (Gravity/GreyMajorAxes) measures perpendicular distance between entire edge regions, not facing edges — gives 25–35 mm instead of 6–8 mm. Closest boundary points is the primary method instead.
- **Live view thread**: Subclasses `QThread` directly (not worker+moveToThread pattern) because `run_loop` is a blocking poll loop that doesn't use the Qt event loop.
- **Exposure time units**: `CameraGetExposureTimeRange` and `CameraGetExposureTime` return **microseconds** directly (not seconds as the name might suggest).

## Distance Measurement Accuracy

Verified on `screenshot2.png` with known object pairs:
- ID 2773 ↔ 2677: **7.958 mm** (expected ~8.00 mm)
- ID 9359 ↔ 2677: **5.970 mm** (expected ~6.00 mm)
