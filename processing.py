"""
Watershed segmentation + distance measurement + overlay rendering.

No Qt dependency — pure numpy / DIPlib / OpenCV.
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import diplib as dip
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.signal import find_peaks


# ── Data types ──────────────────────────────────────────────────────────

PICK_OBJECT1 = 0
PICK_OBJECT2 = 1
SHOW_RESULT = 2


@dataclass
class SegmentationResult:
    labels: np.ndarray                        # 2D int, 0=boundary, >0=region ID
    region_sizes: dict[int, int]              # label -> pixel count


@dataclass
class DistanceResult:
    distance_mm: float
    method: str = "boundary-pixel"
    pt1: tuple[int, int] = (0, 0)             # closest boundary point on obj1
    pt2: tuple[int, int] = (0, 0)             # closest boundary point on obj2
    dt_mm: Optional[float] = None             # DIPlib DT cross-check
    profile_mm: Optional[float] = None        # 1D gradient profile cross-check
    obj1_id: int = 0
    obj2_id: int = 0


# ── Overlay colors (BGR) ───────────────────────────────────────────────

COL_MASK1    = (0, 180, 255)     # orange
COL_MASK2    = (255, 100, 0)     # blue
COL_CONTOUR1 = (0, 220, 255)     # yellow
COL_CONTOUR2 = (255, 180, 0)     # cyan
COL_MEASURE  = (0, 255, 0)       # green measurement line
COL_TEXT     = (0, 255, 0)       # green text
COL_POINT    = (0, 0, 255)       # red click markers


# ════════════════════════════════════════════════════════════════════════
#  WatershedProcessor
# ════════════════════════════════════════════════════════════════════════

class WatershedProcessor:
    """Watershed segmentation and distance computation."""

    def __init__(
        self,
        pixel_size: float = 0.117027,
        gauss_sigma: float = 0.4,
        morph_radius: int = 3,
        min_region_size: int = 500,
        watershed_connectivity: int = 1,
        watershed_max_depth: int = 3,
    ):
        self.pixel_size = pixel_size
        self.gauss_sigma = gauss_sigma
        self.morph_radius = morph_radius
        self.min_region_size = min_region_size
        self.watershed_connectivity = watershed_connectivity
        self.watershed_max_depth = watershed_max_depth

    # ── Segmentation ────────────────────────────────────────────────

    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Run watershed pipeline on a BGR image.

        Pipeline: DIPlib load → Gauss → grey → GradientMagnitude → Norm
                  → Closing/Opening → Watershed → numpy label array
        """
        # BGR→RGB, then tell DIPlib this is an RGB image
        rgb = image_bgr[:, :, ::-1].copy()
        dip_img = dip.Image(rgb)
        dip_img.SetColorSpace('sRGB')
        dip_img.SetPixelSize([self.pixel_size, self.pixel_size], "mm")

        img = dip.Gauss(dip_img, self.gauss_sigma)
        img2 = dip.ColorSpaceManager.Convert(img, 'grey')

        gm = dip.Norm(dip.GradientMagnitude(img2))
        gm = dip.Opening(dip.Closing(gm, self.morph_radius), self.morph_radius)

        wlab = dip.Watershed(
            gm,
            connectivity=self.watershed_connectivity,
            maxDepth=self.watershed_max_depth,
            flags={'correct', 'labels'},
        )

        labels = np.array(wlab)
        unique_ids, counts = np.unique(labels, return_counts=True)
        region_sizes = dict(zip(unique_ids.tolist(), counts.tolist()))

        return SegmentationResult(labels=labels, region_sizes=region_sizes)

    def make_dip_grey(self, image_bgr: np.ndarray) -> dip.Image:
        """Create DIPlib grey image with pixel size set (for DT)."""
        rgb = image_bgr[:, :, ::-1].copy()
        dip_img = dip.Image(rgb)
        dip_img.SetColorSpace('sRGB')
        dip_img.SetPixelSize([self.pixel_size, self.pixel_size], "mm")
        img = dip.Gauss(dip_img, self.gauss_sigma)
        grey = dip.ColorSpaceManager.Convert(img, 'grey')
        grey.SetPixelSize([self.pixel_size, self.pixel_size], "mm")
        return grey

    # ── Label lookup ────────────────────────────────────────────────

    def get_label_at(
        self,
        labels: np.ndarray,
        ix: int,
        iy: int,
        region_sizes: dict[int, int],
    ) -> Optional[int]:
        """Read watershed label at (ix, iy). Returns None if boundary(0) or too small."""
        h, w = labels.shape
        if iy < 0 or iy >= h or ix < 0 or ix >= w:
            return None
        label = int(labels[iy, ix])
        if label == 0:
            return None
        size = region_sizes.get(label, 0)
        if size < self.min_region_size:
            return None
        return label

    # ── Distance computation ────────────────────────────────────────

    @staticmethod
    def find_closest_boundary_points(
        ws_labels: np.ndarray,
        id1: int,
        id2: int,
    ) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        """Find closest pair of boundary pixels between two watershed objects."""
        mask1 = (ws_labels == id1).astype(np.uint8) * 255
        mask2 = (ws_labels == id2).astype(np.uint8) * 255

        cnts1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not cnts1 or not cnts2:
            return None, None

        c1 = max(cnts1, key=cv2.contourArea)[:, 0, :]
        c2 = max(cnts2, key=cv2.contourArea)[:, 0, :]

        best_dist = np.inf
        best_i, best_j = 0, 0
        chunk = 2000
        for i0 in range(0, len(c1), chunk):
            diff = (c1[i0:i0+chunk, None, :].astype(np.float64)
                    - c2[None, :, :].astype(np.float64))
            d2 = np.sum(diff * diff, axis=2)
            idx = np.unravel_index(np.argmin(d2), d2.shape)
            d = d2[idx]
            if d < best_dist:
                best_dist = d
                best_i = i0 + idx[0]
                best_j = idx[1]

        pt1 = (int(c1[best_i][0]), int(c1[best_i][1]))
        pt2 = (int(c2[best_j][0]), int(c2[best_j][1]))
        return pt1, pt2

    def _compute_distance_dt(
        self,
        ws_labels: np.ndarray,
        id1: int,
        id2: int,
    ) -> Optional[float]:
        """DIPlib distance transform cross-check (mm)."""
        mask1 = (ws_labels == id1).astype(bool)
        mask2 = (ws_labels == id2).astype(bool)

        inv_m1 = np.ascontiguousarray(~mask1)
        try:
            inv_m1_dip = dip.Image(inv_m1)
            inv_m1_dip.SetPixelSize([self.pixel_size, self.pixel_size], "mm")
            vdt = dip.VectorDistanceTransform(inv_m1_dip)
            dt_dip = dip.Norm(vdt)
            dt_np = np.array(dt_dip)
            m2_u8 = mask2.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            boundary2 = (cv2.dilate(m2_u8, kernel) - cv2.erode(m2_u8, kernel)) > 0
            if not boundary2.any():
                return None
            return float(dt_np[boundary2].min())
        except Exception:
            return None

    def _compute_distance_profile(
        self,
        image_rgb: np.ndarray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
    ) -> Optional[float]:
        """1D gradient profile along closest-point line (sub-pixel, mm)."""
        cx1, cy1 = pt1
        cx2, cy2 = pt2
        dist_px = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        if dist_px < 2:
            return None

        num_samples = max(int(dist_px * 3), 300)
        t = np.linspace(0, 1, num_samples)
        sample_y = cy1 + t * (cy2 - cy1)
        sample_x = cx1 + t * (cx2 - cx1)

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        profile = map_coordinates(gray, [sample_y, sample_x], order=1, mode='nearest')
        profile_smooth = gaussian_filter1d(profile, sigma=3)
        grad = np.abs(np.gradient(profile_smooth))

        min_sep = max(len(grad) // 8, 5)
        peaks, _ = find_peaks(grad, distance=min_sep)

        if len(peaks) >= 2:
            sorted_by_height = sorted(peaks, key=lambda p: grad[p], reverse=True)
            p1_local, p2_local = sorted_by_height[0], sorted_by_height[1]
        else:
            half = len(grad) // 2
            p1_local = int(np.argmax(grad[:half]))
            p2_local = half + int(np.argmax(grad[half:]))

        if p1_local > p2_local:
            p1_local, p2_local = p2_local, p1_local

        # Sub-pixel peak refinement
        def refine_peak(arr, idx):
            if 1 <= idx < len(arr) - 1:
                y0, y1, y2 = arr[idx-1], arr[idx], arr[idx+1]
                denom = 2.0 * (2*y1 - y0 - y2)
                if abs(denom) > 1e-10:
                    return idx + (y0 - y2) / denom
            return float(idx)

        p1_ref = refine_peak(grad, p1_local)
        p2_ref = refine_peak(grad, p2_local)

        t1 = p1_ref / (num_samples - 1)
        t2 = p2_ref / (num_samples - 1)
        edge_x1 = cx1 + t1 * (cx2 - cx1)
        edge_y1 = cy1 + t1 * (cy2 - cy1)
        edge_x2 = cx1 + t2 * (cx2 - cx1)
        edge_y2 = cy1 + t2 * (cy2 - cy1)

        distance_px = np.sqrt((edge_x2 - edge_x1)**2 + (edge_y2 - edge_y1)**2)
        return distance_px * self.pixel_size

    def compute_distance(
        self,
        image_rgb: np.ndarray,
        labels: np.ndarray,
        dip_grey: dip.Image,
        id1: int,
        id2: int,
    ) -> Optional[DistanceResult]:
        """Compute distance between two watershed objects.

        Primary: closest boundary points (pixel dist × pixel_size).
        Cross-checks: DIPlib DT and 1D gradient profile.
        """
        pt1, pt2 = self.find_closest_boundary_points(labels, id1, id2)
        if pt1 is None or pt2 is None:
            return None

        pixel_dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
        simple_mm = pixel_dist * self.pixel_size

        dt_mm = self._compute_distance_dt(labels, id1, id2)
        profile_mm = self._compute_distance_profile(image_rgb, pt1, pt2)

        return DistanceResult(
            distance_mm=simple_mm,
            method="boundary-pixel",
            pt1=pt1,
            pt2=pt2,
            dt_mm=dt_mm,
            profile_mm=profile_mm,
            obj1_id=id1,
            obj2_id=id2,
        )


# ════════════════════════════════════════════════════════════════════════
#  OverlayRenderer
# ════════════════════════════════════════════════════════════════════════

class OverlayRenderer:
    """Pure numpy/OpenCV renderer — produces BGR display images."""

    @staticmethod
    def render(
        canvas: np.ndarray,
        labels: np.ndarray,
        obj1_id: Optional[int],
        obj2_id: Optional[int],
        click1: Optional[tuple[int, int]],
        click2: Optional[tuple[int, int]],
        result: Optional[DistanceResult],
        overlay_alpha: float = 0.35,
    ) -> np.ndarray:
        """Render overlays onto a copy of canvas. Returns composited BGR image."""
        out = canvas.copy()
        img_h, img_w = out.shape[:2]

        # Region overlays
        if obj1_id is not None:
            region1 = (labels == obj1_id)
            overlay = out.copy()
            overlay[region1] = COL_MASK1
            cv2.addWeighted(overlay, overlay_alpha, out, 1 - overlay_alpha, 0, dst=out)
            mask_u8 = region1.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cnts, -1, COL_CONTOUR1, 2)

        if obj2_id is not None:
            region2 = (labels == obj2_id)
            overlay = out.copy()
            overlay[region2] = COL_MASK2
            cv2.addWeighted(overlay, overlay_alpha, out, 1 - overlay_alpha, 0, dst=out)
            mask_u8 = region2.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cnts, -1, COL_CONTOUR2, 2)

        # Click markers
        if click1 is not None:
            ix, iy = click1
            cv2.circle(out, (ix, iy), 6, COL_POINT, -1)
            label_text = f"1: ID={obj1_id}" if obj1_id else "1"
            cv2.putText(out, label_text, (ix + 8, iy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_POINT, 2)
        if click2 is not None:
            ix, iy = click2
            cv2.circle(out, (ix, iy), 6, COL_POINT, -1)
            label_text = f"2: ID={obj2_id}" if obj2_id else "2"
            cv2.putText(out, label_text, (ix + 8, iy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_POINT, 2)

        # Measurement line
        if result is not None:
            cv2.line(out, result.pt1, result.pt2, COL_MEASURE, 2)
            cv2.circle(out, result.pt1, 4, COL_MEASURE, -1)
            cv2.circle(out, result.pt2, 4, COL_MEASURE, -1)

        # Distance text
        if result is not None:
            mx = (result.pt1[0] + result.pt2[0]) / 2.0
            my = (result.pt1[1] + result.pt2[1]) / 2.0
            dx = result.pt2[0] - result.pt1[0]
            dy = result.pt2[1] - result.pt1[1]
            length = np.sqrt(dx * dx + dy * dy)
            if length > 1:
                perp_x = -dy / length
                perp_y = dx / length
                mx += perp_x * 20
                my += perp_y * 20
            mx, my = int(mx), int(my)

            text = f"{result.distance_mm:.3f} mm"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(out, (mx - 4, my - th - 8), (mx + tw + 4, my + 4),
                          (0, 0, 0), -1)
            cv2.putText(out, text, (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_TEXT, 2, cv2.LINE_AA)

        # HUD bar at bottom
        bar_h = 26
        hud = out[img_h - bar_h:img_h, :img_w].copy()
        cv2.rectangle(hud, (0, 0), (img_w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(hud, 0.65, out[img_h - bar_h:img_h, :img_w], 0.35, 0,
                        dst=out[img_h - bar_h:img_h, :img_w])

        pixel_size = 0.117027  # read from result or hardcoded
        hud_text = f"Scale: {pixel_size} mm/px"
        if result is not None:
            hud_text += f"  |  Distance: {result.distance_mm:.3f} mm"
        cv2.putText(out, hud_text, (8, img_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return out
