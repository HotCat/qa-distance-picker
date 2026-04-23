"""
Debug overlay rendering for grid cells and edge segments.

Provides visual annotations to help users understand how arc IDs (grid-based)
and line IDs (edge-segment-based) are assigned. Intended as a debugging aid
toggled via checkboxes in the toolbar.
"""

import cv2
import numpy as np


# ── Grid overlay for arc cell visualization ────────────────────────────────

_GRID_LINE_COLOR = (180, 180, 180)   # bright gray — visible on dark & light
_GRID_LINE_WIDTH = 1
_GRID_LABEL_COLOR = (0, 255, 255)    # yellow — high contrast
_GRID_LABEL_BG = (0, 0, 0)           # black background for labels
_GRID_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GRID_FONT_SCALE = 0.38
_GRID_FONT_THICK = 1


def draw_grid_overlay(
    image_bgr: np.ndarray,
    pixel_size_mm: float,
    grid_size_mm: float,
) -> np.ndarray:
    """Draw grid lines and cell labels for arc ID visualization.

    Labels each cell with "C{row}_{col}" at the top-left corner, matching the
    arc ID format used in detection (e.g. "C4_19_4" → row=4, col=19).

    Returns:
        BGR image with grid overlay drawn directly (no alpha blending).
    """
    canvas = image_bgr.copy()
    img_h, img_w = canvas.shape[:2]

    if pixel_size_mm <= 0 or grid_size_mm <= 0:
        return canvas

    grid_px = int(round(grid_size_mm / pixel_size_mm))
    if grid_px < 2:
        return canvas

    # Vertical grid lines
    x = grid_px
    while x < img_w:
        cv2.line(canvas, (x, 0), (x, img_h - 1),
                 _GRID_LINE_COLOR, _GRID_LINE_WIDTH, cv2.LINE_AA)
        x += grid_px

    # Horizontal grid lines
    y = grid_px
    while y < img_h:
        cv2.line(canvas, (0, y), (img_w - 1, y),
                 _GRID_LINE_COLOR, _GRID_LINE_WIDTH, cv2.LINE_AA)
        y += grid_px

    # Cell labels at each grid intersection
    col = 0
    x = 0
    while x < img_w:
        row = 0
        y = 0
        while y < img_h:
            label = f"C{row}_{col}"
            (tw, th), _ = cv2.getTextSize(label, _GRID_FONT,
                                           _GRID_FONT_SCALE, _GRID_FONT_THICK)
            tx = x + 4
            ty = y + th + 4
            if tx + tw + 4 > img_w:
                tx = max(0, x - tw - 4)
            if ty + 4 > img_h:
                ty = img_h - 4
            if tx >= 0 and ty - th - 2 >= 0:
                cv2.rectangle(canvas, (tx - 2, ty - th - 2),
                              (tx + tw + 2, ty + 3),
                              _GRID_LABEL_BG, -1)
                cv2.putText(canvas, label, (tx, ty), _GRID_FONT,
                            _GRID_FONT_SCALE, _GRID_LABEL_COLOR,
                            _GRID_FONT_THICK, cv2.LINE_AA)
            row += 1
            y += grid_px
        col += 1
        x += grid_px

    return canvas


def compute_grid_cell(
    ix: int, iy: int,
    pixel_size_mm: float,
    grid_size_mm: float,
) -> tuple[int, int]:
    """Compute grid cell (row, col) from image pixel coordinates."""
    col = int(ix * pixel_size_mm // grid_size_mm)
    row = int(iy * pixel_size_mm // grid_size_mm)
    return row, col


# ── Edge segment overlay for line ID visualization ──────────────────────────

_EDGE_COLORS = {
    'Up': (0, 220, 220),    # yellow — top edge
    'Lo': (80, 80, 255),    # red — bottom edge
    'Le': (220, 220, 0),    # cyan — left edge
    'Ri': (220, 0, 220),    # magenta — right edge
}
_EDGE_BG = (0, 0, 0)
_EDGE_FONT = cv2.FONT_HERSHEY_SIMPLEX
_EDGE_FONT_SCALE = 0.38
_EDGE_FONT_THICK = 1
_EDGE_TICK_LEN = 30        # tick mark length in pixels
_EDGE_TICK_WIDTH = 2        # tick mark thickness


def _segment_number(pos_mm: float, segment_mm: float) -> int:
    """1-indexed segment number for a position along an edge (in mm)."""
    if segment_mm <= 0:
        return 1
    return max(1, int(pos_mm / segment_mm) + (1 if pos_mm % segment_mm > 0 else 0))


def draw_edge_segments(
    image_bgr: np.ndarray,
    pixel_size_mm: float,
    segment_mm: float,
) -> np.ndarray:
    """Draw edge segment tick marks and labels for line ID visualization.

    Edge names: Up (top, y=0), Lo (bottom, y=H-1), Le (left, x=0), Ri (right, x=W-1).
    Each segment is labeled with its edge name + 1-indexed number (e.g. "Up3", "Le1").
    """
    canvas = image_bgr.copy()
    img_h, img_w = canvas.shape[:2]

    if pixel_size_mm <= 0 or segment_mm <= 0:
        return canvas

    seg_px = int(round(segment_mm / pixel_size_mm))
    if seg_px < 2:
        return canvas

    tick = _EDGE_TICK_LEN
    w = _EDGE_TICK_WIDTH

    # Up edge (y = 0): segments along x-axis
    seg = 1
    x = 0
    while x < img_w:
        cv2.line(canvas, (x, 0), (x, min(tick, img_h - 1)),
                 _EDGE_COLORS['Up'], w, cv2.LINE_AA)
        _draw_seg_label(canvas, f"Up{seg}", x + 3, th_offset=4,
                        color=_EDGE_COLORS['Up'], img_w=img_w, img_h=img_h)
        seg += 1
        x += seg_px

    # Lo edge (y = img_h - 1): segments along x-axis
    seg = 1
    x = 0
    yb = img_h - 1
    while x < img_w:
        cv2.line(canvas, (x, max(0, yb - tick)), (x, yb),
                 _EDGE_COLORS['Lo'], w, cv2.LINE_AA)
        _draw_seg_label(canvas, f"Lo{seg}", x + 3, bottom=yb - 3,
                        color=_EDGE_COLORS['Lo'], img_w=img_w, img_h=img_h)
        seg += 1
        x += seg_px

    # Le edge (x = 0): segments along y-axis
    seg = 1
    y = 0
    while y < img_h:
        cv2.line(canvas, (0, y), (min(tick, img_w - 1), y),
                 _EDGE_COLORS['Le'], w, cv2.LINE_AA)
        _draw_seg_label(canvas, f"Le{seg}", tx_start=3, ty_mid=y + 4,
                        color=_EDGE_COLORS['Le'], img_w=img_w, img_h=img_h)
        seg += 1
        y += seg_px

    # Ri edge (x = img_w - 1): segments along y-axis
    seg = 1
    y = 0
    xr = img_w - 1
    while y < img_h:
        cv2.line(canvas, (max(0, xr - tick), y), (xr, y),
                 _EDGE_COLORS['Ri'], w, cv2.LINE_AA)
        _draw_seg_label(canvas, f"Ri{seg}", right=xr, ty_mid=y + 4,
                        color=_EDGE_COLORS['Ri'], img_w=img_w, img_h=img_h)
        seg += 1
        y += seg_px

    return canvas


def _draw_seg_label(canvas, label, tx_start=None, th_offset=None,
                    bottom=None, ty_mid=None, right=None,
                    color=(255, 255, 255), img_w=1920, img_h=1080):
    """Draw a segment label with black background, clamped to image bounds."""
    (tw, th), _ = cv2.getTextSize(label, _EDGE_FONT, _EDGE_FONT_SCALE,
                                   _EDGE_FONT_THICK)

    # Compute (tx, ty) where tx is left edge of text and ty is baseline
    if tx_start is not None and th_offset is not None:
        # Top-edge labels: anchored near top
        tx, ty = tx_start, th_offset + th
    elif tx_start is not None and bottom is not None:
        # Bottom-edge labels: anchored near bottom
        tx, ty = tx_start, bottom
    elif right is not None and ty_mid is not None:
        # Right-edge labels: right-aligned
        tx = right - tw - 3
        ty = ty_mid + th
    elif ty_mid is not None:
        # Left-edge labels: left-aligned
        tx = 3
        ty = ty_mid + th
    else:
        return

    # Clamp to image bounds
    tx = max(2, min(tx, img_w - tw - 4))
    ty = max(th + 3, min(ty, img_h - 3))

    cv2.rectangle(canvas, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 3),
                  _EDGE_BG, -1)
    cv2.putText(canvas, label, (tx, ty), _EDGE_FONT, _EDGE_FONT_SCALE,
                color, _EDGE_FONT_THICK, cv2.LINE_AA)


def compute_edge_segments(
    ix: int, iy: int,
    img_w: int, img_h: int,
    pixel_size_mm: float,
    segment_mm: float,
) -> dict[str, int]:
    """Compute which edge segments a point falls into.

    Returns dict mapping edge name to 1-indexed segment number.
    """
    x_mm = ix * pixel_size_mm
    y_mm = iy * pixel_size_mm
    return {
        'Up': _segment_number(x_mm, segment_mm),
        'Lo': _segment_number(x_mm, segment_mm),
        'Le': _segment_number(y_mm, segment_mm),
        'Ri': _segment_number(y_mm, segment_mm),
    }
