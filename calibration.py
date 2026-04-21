"""
Chessboard-based pixel-size calibration.

Detects a chessboard pattern in a camera frame using OpenCV's
findChessboardCorners + cornerSubPix, then computes the mm/px scale
factor from the known physical grid size.

No Qt dependency — pure numpy / OpenCV. The CalibrationWorker QThread
subclass is provided for async use from the UI.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from PySide6.QtCore import QThread, Signal


# ── Data types ──────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    pixel_size_mm: float       # mm per pixel (for SetPixelSize)
    board_cols: int            # inner corner count (horizontal)
    board_rows: int            # inner corner count (vertical)
    grid_size_mm: float        # physical grid square size
    mean_spacing_px: float     # average corner-to-corner distance in pixels
    corners: np.ndarray        # (N, 1, 2) subpixel corner positions


# ── Core calibration function ───────────────────────────────────────────

def calibrate_pixel_size(
    image_bgr: np.ndarray,
    board_cols: int,
    board_rows: int,
    grid_size_mm: float,
) -> CalibrationResult | None:
    """Detect chessboard and compute mm/px scale factor.

    Unlike the brute-force approach that tries every board size from 3x3
    to 19x19 (289 calls to findChessboardCorners), this uses the
    configured dimensions directly — a single call. Fails explicitly if
    the pattern doesn't match, which is the desired validation behavior.

    Args:
        image_bgr: BGR or grayscale image containing a chessboard pattern.
        board_cols: Number of inner corners (horizontal).
        board_rows: Number of inner corners (vertical).
        grid_size_mm: Physical size of one grid square in mm.

    Returns:
        CalibrationResult on success, None if chessboard not detected.
    """
    if image_bgr.ndim == 2:
        gray = image_bgr
    elif image_bgr.shape[2] == 1:
        gray = image_bgr[:, :, 0]
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, (board_cols, board_rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not found:
        return None

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Measure spacing between adjacent corners
    c = corners.squeeze()  # (N, 2)
    dists_h, dists_v = [], []
    for r in range(board_rows):
        for col in range(board_cols - 1):
            d = np.linalg.norm(c[r * board_cols + col] - c[r * board_cols + col + 1])
            dists_h.append(d)
    for r in range(board_rows - 1):
        for col in range(board_cols):
            d = np.linalg.norm(c[r * board_cols + col] - c[(r + 1) * board_cols + col])
            dists_v.append(d)

    mean_spacing = (np.mean(dists_h) + np.mean(dists_v)) / 2.0
    pixel_size_mm = grid_size_mm / mean_spacing

    return CalibrationResult(
        pixel_size_mm=pixel_size_mm,
        board_cols=board_cols,
        board_rows=board_rows,
        grid_size_mm=grid_size_mm,
        mean_spacing_px=mean_spacing,
        corners=corners,
    )


# ── Async worker for UI ─────────────────────────────────────────────────

class CalibrationWorker(QThread):
    """Run calibration in a background thread."""

    done = Signal(object)   # CalibrationResult on success
    error = Signal(str)

    def __init__(self, image_bgr: np.ndarray,
                 board_cols: int, board_rows: int,
                 grid_size_mm: float):
        super().__init__()
        self._image = image_bgr.copy()
        self._board_cols = board_cols
        self._board_rows = board_rows
        self._grid_size_mm = grid_size_mm

    def run(self):
        result = calibrate_pixel_size(
            self._image, self._board_cols, self._board_rows, self._grid_size_mm)
        if result is not None:
            self.done.emit(result)
        else:
            self.error.emit(
                f"Chessboard pattern ({self._board_cols}×{self._board_rows}) "
                f"not detected in captured frame")
