"""
Line and arc detection for metrology.

Detects straight lines (LSD + collinear merge) and circular arcs
(curvature peaks + RANSAC circle fit) in an image. Returns structured
results with stable ID assignment across repeated measurements of the
same workpiece (tolerates ±10mm offset, ±15° rotation).
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import diplib as dip
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

from PySide6.QtCore import QThread, Signal


# ── Data types ──────────────────────────────────────────────────────────

@dataclass
class LineResult:
    id: str                    # "L1", "L2", ...  (stable across runs)
    category: str              # "H", "V", "D", "O"
    start_px: np.ndarray       # (2,) start endpoint in pixels
    end_px: np.ndarray         # (2,) end endpoint in pixels
    length_mm: float
    angle_deg: float
    centroid_mm: np.ndarray    # (2,) midpoint in mm — for ID matching


@dataclass
class ArcResult:
    id: str                    # "C1", "C2", ...  (stable across runs)
    center_px: np.ndarray      # (2,) circle center in pixels
    radius_px: float
    radius_mm: float
    centroid_mm: np.ndarray    # (2,) center in mm — for ID matching


@dataclass
class LinesArcsResult:
    lines: list[LineResult] = field(default_factory=list)
    arcs: list[ArcResult] = field(default_factory=list)
    arc_grid: dict = field(default_factory=dict)
    line_grid: dict = field(default_factory=dict)


@dataclass
class FeaturePair:
    """A pair of features (line/arc) for metrology measurement."""
    type_a: str          # "line" or "arc"
    id_a: str            # feature ID from detection
    type_b: str
    id_b: str
    distance_mm: float = 0.0
    lower_mm: float = 0.0
    upper_mm: float = 0.0


@dataclass
class FeatureDescriptor:
    """Compact descriptor used for stable ID matching."""
    id: str
    centroid_mm: np.ndarray   # (2,)
    primary: float            # angle_deg for lines, radius_mm for arcs
    secondary: float          # length_mm for lines, n_inliers for arcs


# ── Line detection: LSD + merge ─────────────────────────────────────────

def line_angle(a, b):
    return np.degrees(np.arctan2(-a, b)) % 180


def angle_diff(a1, a2):
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def merge_collinear_lines(lines, angle_threshold=3.0, dist_threshold=4.0,
                          overlap_threshold=0.3):
    if not lines:
        return lines
    merged = []
    used = [False] * len(lines)
    for i in range(len(lines)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        a1, b1, c1 = lines[i]['line']
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            a2, b2, c2 = lines[j]['line']
            if angle_diff(lines[i]['angle'], lines[j]['angle']) > angle_threshold:
                continue
            mid_j = (lines[j]['start'] + lines[j]['end']) / 2
            dist = abs(a1 * mid_j[0] + b1 * mid_j[1] + c1)
            if dist >= dist_threshold:
                continue
            direction = np.array([b1, -a1])
            proj_i_s = np.dot(lines[i]['start'], direction)
            proj_i_e = np.dot(lines[i]['end'], direction)
            proj_j_s = np.dot(lines[j]['start'], direction)
            proj_j_e = np.dot(lines[j]['end'], direction)
            i_lo, i_hi = min(proj_i_s, proj_i_e), max(proj_i_s, proj_i_e)
            j_lo, j_hi = min(proj_j_s, proj_j_e), max(proj_j_s, proj_j_e)
            overlap_len = max(0, min(i_hi, j_hi) - max(i_lo, j_lo))
            j_len = j_hi - j_lo
            if (j_len > 0 and overlap_len / j_len > overlap_threshold) or dist < dist_threshold:
                group.append(j)
                used[j] = True
        all_pts = []
        for idx in group:
            all_pts.extend([lines[idx]['start'], lines[idx]['end']])
        all_pts = np.array(all_pts)
        a, b, c = lines[group[0]]['line']
        direction = np.array([-b, a])
        centroid = all_pts.mean(axis=0)
        projs = [(p - centroid) @ direction for p in all_pts]
        p_start = centroid + min(projs) * direction
        p_end = centroid + max(projs) * direction
        merged.append({
            'line': (a, b, c),
            'start': p_start,
            'end': p_end,
            'length': np.linalg.norm(p_end - p_start),
            'n_inliers': sum(lines[idx].get('n_inliers', 0) for idx in group),
            'angle': line_angle(a, b),
        })
    return merged


def detect_lines(gray):
    lsd = cv2.createLineSegmentDetector()
    lsd_result = lsd.detect(gray)
    if lsd_result[0] is None:
        return []
    segments = []
    for l in lsd_result[0]:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 15:
            continue
        dx, dy = x2 - x1, y2 - y1
        ln = np.hypot(dx, dy)
        a, b = -dy / ln, dx / ln
        c = -(a * x1 + b * y1)
        if a < 0 or (a == 0 and b < 0):
            a, b, c = -a, -b, -c
        segments.append({
            'line': (a, b, c),
            'start': np.array([x1, y1]),
            'end': np.array([x2, y2]),
            'length': length,
            'n_inliers': 0,
            'angle': line_angle(a, b),
        })
    merged = merge_collinear_lines(segments, angle_threshold=3.0,
                                   dist_threshold=4.0, overlap_threshold=0.3)
    merged = merge_collinear_lines(merged, angle_threshold=3.0,
                                   dist_threshold=5.0, overlap_threshold=0.3)
    merged = [l for l in merged if l['length'] > 30]
    merged.sort(key=lambda l: -l['length'])
    return merged


# ── Arc detection: curvature + RANSAC circle fit ────────────────────────

def solve_circle_from_3_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-10:
        return None
    Ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) +
          (x3**2 + y3**2) * (y1 - y2)) / D
    Uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) +
          (x3**2 + y3**2) * (x2 - x1)) / D
    center = np.array([Ux, Uy])
    radius = np.sqrt((x1 - Ux)**2 + (y1 - Uy)**2)
    return center, radius


def ransac_fit_circle(points, n_iterations=2000, threshold=2.0, min_inliers=10):
    points = np.array(points, dtype=np.float64)
    if len(points) < 3:
        return None, None, 0
    best_inlier_count = 0
    best_center = None
    best_radius = None
    best_inlier_mask = None
    for _ in range(n_iterations):
        indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[indices]
        result = solve_circle_from_3_points(p1, p2, p3)
        if result is None:
            continue
        center, radius = result
        if radius > 300:
            continue
        distances = np.sqrt(np.sum((points - center)**2, axis=1))
        inlier_mask = np.abs(distances - radius) < threshold
        n_inliers = np.sum(inlier_mask)
        if n_inliers > best_inlier_count:
            best_inlier_count = n_inliers
            best_center = center
            best_radius = radius
            best_inlier_mask = inlier_mask
    if best_inlier_mask is not None and best_inlier_count >= min_inliers:
        inlier_points = points[best_inlier_mask]
        X = inlier_points[:, 0]
        Y = inlier_points[:, 1]
        Z = X**2 + Y**2
        A_mat = np.column_stack([X, Y, np.ones_like(X)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, Z, rcond=None)
            A, B, C = coeffs
            xc = A / 2
            yc = B / 2
            r = np.sqrt(C + xc**2 + yc**2)
            if r < 300:
                return np.array([xc, yc]), r, best_inlier_count
        except np.linalg.LinAlgError:
            pass
        return best_center, best_radius, best_inlier_count
    return None, None, 0


def compute_curvature(contour, window=15):
    n = len(contour)
    curvatures = np.zeros(n)
    for i in range(n):
        i_prev = (i - window) % n
        i_next = (i + window) % n
        v1 = contour[i] - contour[i_prev]
        v2 = contour[i_next] - contour[i]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-6 or len2 < 1e-6:
            curvatures[i] = 0
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
        angle = np.arccos(cos_angle)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        curvatures[i] = angle if cross > 0 else -angle
    return curvatures


def extract_arc_region(contour, abs_curv, peak_idx, drop_ratio=0.3):
    n = len(contour)
    peak_curv = abs_curv[peak_idx]
    threshold = peak_curv * drop_ratio
    end_fwd = peak_idx
    for step in range(1, n // 4):
        idx = (peak_idx + step) % n
        if abs_curv[idx] < threshold:
            end_fwd = idx
            break
    else:
        end_fwd = (peak_idx + n // 8) % n
    end_bwd = peak_idx
    for step in range(1, n // 4):
        idx = (peak_idx - step) % n
        if abs_curv[idx] < threshold:
            end_bwd = idx
            break
    else:
        end_bwd = (peak_idx - n // 8) % n
    arc_pts = []
    idx = end_bwd
    while True:
        arc_pts.append(contour[idx])
        if idx == end_fwd:
            break
        idx = (idx + 1) % n
        if len(arc_pts) > n // 2:
            break
    return np.array(arc_pts)


def detect_arcs_for_object(obj_id, wlab, px_size_mm):
    obj_mask = (wlab == obj_id)
    obj_array = np.array(obj_mask).astype(np.uint8)
    if obj_array.sum() == 0:
        return []

    contours, _ = cv2.findContours(obj_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour_raw = max(contours, key=cv2.contourArea)
    contour = contour_raw.squeeze(1).astype(np.float64)
    if len(contour) < 50:
        return []

    contour_smooth_x = gaussian_filter1d(contour[:, 0], sigma=5, mode='wrap')
    contour_smooth_y = gaussian_filter1d(contour[:, 1], sigma=5, mode='wrap')
    contour_smooth = np.column_stack([contour_smooth_x, contour_smooth_y])

    curvatures = compute_curvature(contour_smooth, window=20)
    abs_curv = np.abs(curvatures)

    min_peak_dist = max(50, len(contour) // 20)
    peaks, _ = find_peaks(abs_curv, height=0.1, distance=min_peak_dist, prominence=0.05)
    if len(peaks) < 2:
        peaks, _ = find_peaks(abs_curv, height=0.03, distance=min_peak_dist, prominence=0.02)
    if len(peaks) < 2:
        peaks, _ = find_peaks(abs_curv, height=0.01, distance=min_peak_dist // 2, prominence=0.005)

    arcs = []
    arc_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                  [255, 0, 255], [0, 255, 255], [128, 255, 0], [255, 128, 0]]

    for i, pk in enumerate(peaks):
        arc_pts = extract_arc_region(contour, abs_curv, pk, drop_ratio=0.3)
        center, radius, num_inliers = ransac_fit_circle(
            arc_pts, n_iterations=2000, threshold=2.0, min_inliers=5
        )
        if center is not None and 3 < radius < 300:
            arcs.append({
                'center_px': center,
                'radius_px': radius,
                'radius_mm': radius * px_size_mm,
                'n_inliers': num_inliers,
                'color': arc_colors[i % len(arc_colors)],
            })

    arcs = [a for a in arcs if a['radius_mm'] >= 2.0]
    return arcs


def deduplicate_cross_object_arcs(all_arcs, center_dist_threshold=25.0,
                                  radius_diff_threshold=5.0):
    """Merge overlapping arcs detected on shared edges between adjacent objects."""
    flat_arcs = []
    for obj_id, arcs in all_arcs.items():
        for i, a in enumerate(arcs):
            flat_arcs.append({
                'obj_id': obj_id,
                'arc_idx': i,
                'center': a['center_px'],
                'radius_px': a['radius_px'],
                'radius_mm': a['radius_mm'],
                'n_inliers': a['n_inliers'],
                'color': a['color'],
            })

    used = [False] * len(flat_arcs)
    merged_arcs = []
    for i in range(len(flat_arcs)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(flat_arcs)):
            if used[j]:
                continue
            if flat_arcs[i]['obj_id'] == flat_arcs[j]['obj_id']:
                continue
            d = np.linalg.norm(flat_arcs[i]['center'] - flat_arcs[j]['center'])
            r_diff = abs(flat_arcs[i]['radius_px'] - flat_arcs[j]['radius_px'])
            if d < center_dist_threshold and r_diff < radius_diff_threshold:
                group.append(j)
                used[j] = True

        if len(group) == 1:
            merged_arcs.append(flat_arcs[i])
        else:
            best_idx = max(group, key=lambda k: flat_arcs[k]['n_inliers'])
            merged_arcs.append(flat_arcs[best_idx])

    return merged_arcs


# ── Stable ID matching ─────────────────────────────────────────────────

def _classify_line(angle):
    if angle < 10 or angle > 170:
        return "H"
    if 80 < angle < 100:
        return "V"
    if (40 < angle < 60) or (120 < angle < 140):
        return "D"
    return "O"


# ── Edge-intersection line ID assignment ─────────────────────────────────

_EDGE_ORDER = ['Up', 'Lo', 'Le', 'Ri']


def _extend_line_to_edges(p1, p2, img_w, img_h):
    """Find the two points where the line through p1-p2 meets the image frame.

    Returns a list of (edge_name, x, y) tuples for the two valid intersections.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1

    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return []

    hits = []
    eps = 1e-6

    # Upper edge: y = 0
    if abs(dy) > eps:
        t = -y1 / dy
        x = x1 + t * dx
        if -eps <= x <= img_w - 1 + eps:
            hits.append(('Up', x, 0.0))

    # Lower edge: y = img_h - 1
    if abs(dy) > eps:
        t = (img_h - 1 - y1) / dy
        x = x1 + t * dx
        if -eps <= x <= img_w - 1 + eps:
            hits.append(('Lo', x, float(img_h - 1)))

    # Left edge: x = 0
    if abs(dx) > eps:
        t = -x1 / dx
        y = y1 + t * dy
        if -eps <= y <= img_h - 1 + eps:
            hits.append(('Le', 0.0, y))

    # Right edge: x = img_w - 1
    if abs(dx) > eps:
        t = (img_w - 1 - x1) / dx
        y = y1 + t * dy
        if -eps <= y <= img_h - 1 + eps:
            hits.append(('Ri', float(img_w - 1), y))

    # Deduplicate: a line through a corner can produce two hits at the same point
    seen = set()
    unique = []
    for name, x, y in hits:
        key = (name, round(x, 1), round(y, 1))
        if key not in seen:
            seen.add(key)
            unique.append((name, x, y))

    # Sort by canonical edge order
    unique.sort(key=lambda h: _EDGE_ORDER.index(h[0]))

    return unique[:2]


def _segment_number(pos_mm, segment_mm):
    """1-indexed segment number for a position along an edge (in mm)."""
    if segment_mm <= 0:
        return 1
    return max(1, int(pos_mm / segment_mm) + (1 if pos_mm % segment_mm > 0 else 0))


def _assign_line_ids_by_edges(lines, img_w, img_h, pixel_size_mm,
                               segment_mm, template_line_grid=None):
    """Assign line IDs based on which image-edge segments the line crosses.

    Key includes length bucket so parallel lines at different positions
    get unique IDs.

    Returns:
        line_grid: dict mapping (edge1, seg1, edge2, seg2, len_bucket) -> (id, angle_deg)
    """
    line_grid = {}

    for lr in lines:
        hits = _extend_line_to_edges(lr.start_px, lr.end_px, img_w, img_h)
        if len(hits) < 2:
            lr.id = f"L_unk_{lr.angle_deg:.1f}_{lr.length_mm:.2f}"
            continue

        e1_name, e1_x, e1_y = hits[0]
        e2_name, e2_x, e2_y = hits[1]

        # Segment number: Up/Lo use x_mm, Le/Ri use y_mm
        if e1_name in ('Up', 'Lo'):
            s1 = _segment_number(e1_x * pixel_size_mm, segment_mm)
        else:
            s1 = _segment_number(e1_y * pixel_size_mm, segment_mm)

        if e2_name in ('Up', 'Lo'):
            s2 = _segment_number(e2_x * pixel_size_mm, segment_mm)
        else:
            s2 = _segment_number(e2_y * pixel_size_mm, segment_mm)

        len_bucket = round(lr.length_mm, 1)
        key = (e1_name, s1, e2_name, s2, len_bucket)
        angle = lr.angle_deg

        # Try template match
        if template_line_grid and key in template_line_grid:
            t_id, t_angle = template_line_grid[key]
            if abs(angle - t_angle) < 15.0 or abs(angle - t_angle) > 165.0:
                lr.id = t_id
                line_grid[key] = (t_id, angle)
                continue

        lr.id = f"L_{e1_name}{s1}_{e2_name}{s2}_{angle:.1f}_{lr.length_mm:.2f}"
        line_grid[key] = (lr.id, angle)

    return line_grid


def _estimate_alignment(template_pts, detection_pts):
    """Estimate translation to align detection centroids to template space.

    Uses mean-centroid offset since we don't have correspondences yet
    (Procrustes/SVD requires paired points). Returns (R, t) where R is
    always identity and t is the centroid offset, or (None, None) if
    either set is empty.
    """
    if len(template_pts) < 1 or len(detection_pts) < 1:
        return None, None
    t_c = np.mean(template_pts, axis=0)
    d_c = np.mean(detection_pts, axis=0)
    R = np.eye(2)
    t = t_c - d_c
    return R, t


def match_features(
    template: list[FeatureDescriptor],
    detections: list[FeatureDescriptor],
    spatial_tol_mm: float = 10.0,
    angle_tol_deg: float = 15.0,
    radius_tol_ratio: float = 0.3,
) -> dict[str, str]:
    """Match detected features to template IDs using Hungarian assignment.

    Handles small translation (±spatial_tol_mm) and rotation (±angle_tol_deg).

    Args:
        template: Feature descriptors from the template (first) run.
        detections: Feature descriptors from the current run.
        spatial_tol_mm: Max spatial distance for a valid match (mm).
        angle_tol_deg: Max angle difference for line matching (degrees).
        radius_tol_ratio: Max radius ratio difference for arc matching.

    Returns:
        Dict mapping detection list index to template ID.
        Unmatched detections get None.
    """
    if not template or not detections:
        return {}

    # Try to estimate rigid transform between centroids
    t_centroids = np.array([f.centroid_mm for f in template])
    d_centroids = np.array([f.centroid_mm for f in detections])

    R, t = _estimate_alignment(t_centroids, d_centroids)

    # Align detection centroids to template space
    if R is not None:
        aligned = d_centroids + t
    else:
        aligned = d_centroids

    n_t = len(template)
    n_d = len(detections)

    # Build cost matrix
    cost = np.full((n_d, n_t), 1e9)
    for i in range(n_d):
        for j in range(n_t):
            spatial_dist = np.linalg.norm(aligned[i] - t_centroids[j])
            if spatial_dist > spatial_tol_mm:
                continue

            # Check feature-type-specific tolerance
            t_id = template[j].id
            if t_id.startswith('L'):
                # Line: check angle difference
                angle_d = angle_diff(detections[i].primary, template[j].primary)
                if angle_d > angle_tol_deg:
                    continue
                feat_cost = angle_d / angle_tol_deg
            else:
                # Arc: check radius similarity
                r_ratio = abs(detections[i].primary - template[j].primary)
                r_ref = max(abs(template[j].primary), 1.0)
                if r_ratio / r_ref > radius_tol_ratio:
                    continue
                feat_cost = r_ratio / r_ref

            # Combined cost: spatial dominates, feature type refines
            cost[i, j] = spatial_dist + feat_cost * spatial_tol_mm

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    result: dict[int, Optional[str]] = {i: None for i in range(n_d)}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e8:
            result[r] = template[c].id

    return result


def _assign_arc_ids_by_grid(arcs, grid_size_mm, template_grid=None):
    """Assign stable arc IDs based on grid-cell + radius bucket.

    Key = (grid_row, grid_col, radius_bucket) where radius_bucket =
    int(radius_mm).  This handles multiple arcs per cell with different radii.

    If template_grid is provided (from first run), arcs in the same
    (row, col, radius_bucket) inherit the template ID.

    Returns:
        grid_map: dict mapping (row, col, radius_bucket) -> (id, radius_mm)
    """
    grid_map = {}

    for ar in arcs:
        gx = int(ar.centroid_mm[0] // grid_size_mm)
        gy = int(ar.centroid_mm[1] // grid_size_mm)
        rb = int(ar.radius_mm)

        key = (gy, gx, rb)

        if template_grid and key in template_grid:
            t_id, t_radius = template_grid[key]
            if abs(ar.radius_mm - t_radius) / max(t_radius, 1.0) < 0.3:
                ar.id = t_id

        if not ar.id:
            ar.id = f"C{gy}_{gx}_{rb}"

        grid_map[key] = (ar.id, ar.radius_mm)

    return grid_map


# ── Distance computation for feature pairs ────────────────────────────────

def _perpendicular_foot_to_line(point, line_start, line_end):
    """Project a point onto the infinite line through line_start→line_end.

    Returns the foot of the perpendicular (on the extended line, not clamped).
    """
    d = line_end - line_start
    length_sq = np.dot(d, d)
    if length_sq < 1e-10:
        return line_start.copy()
    t = np.dot(point - line_start, d) / length_sq
    return line_start + t * d


def distance_line_to_line(line1: LineResult, line2: LineResult,
                          pixel_size_mm: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Perpendicular distance from center of line1 to line2 (extended if needed).

    Finds the midpoint of line1, draws a perpendicular through it, and finds
    the intersection with line2 (extended beyond its endpoints if necessary).
    Distance is from the midpoint of line1 to that intersection point.

    Returns:
        (distance_mm, midpoint_of_line1, intersection_on_line2) in pixel coordinates.
    """
    mid = (line1.start_px + line1.end_px).astype(float) / 2.0
    intersect = _perpendicular_foot_to_line(mid, line2.start_px.astype(float),
                                            line2.end_px.astype(float))
    return float(np.linalg.norm(mid - intersect) * pixel_size_mm), mid, intersect


def distance_arc_to_arc(arc1: ArcResult, arc2: ArcResult,
                        pixel_size_mm: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Euclidean distance between circle centers in mm.

    Returns:
        (distance_mm, center1_px, center2_px).
    """
    return (float(np.linalg.norm(arc1.center_px - arc2.center_px) * pixel_size_mm),
            arc1.center_px.copy(), arc2.center_px.copy())


def distance_line_to_arc(line: LineResult, arc: ArcResult,
                         pixel_size_mm: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Perpendicular distance from arc center to the line (extended if needed).

    Drops a perpendicular from the arc center to the infinite line through
    the line segment endpoints (extended beyond endpoints if necessary).

    Returns:
        (distance_mm, foot_of_perpendicular_px, arc_center_px).
    """
    foot = _perpendicular_foot_to_line(arc.center_px.astype(float),
                                       line.start_px.astype(float),
                                       line.end_px.astype(float))
    return float(np.linalg.norm(arc.center_px - foot) * pixel_size_mm), foot, arc.center_px.copy()


def compute_feature_distance(
    type_a: str, id_a: str, type_b: str, id_b: str,
    lines: list[LineResult], arcs: list[ArcResult],
    pixel_size_mm: float,
) -> float | None:
    """Compute distance between two features by ID."""
    result = compute_feature_pair_points(
        type_a, id_a, type_b, id_b, lines, arcs, pixel_size_mm)
    return result[0] if result is not None else None


def compute_feature_pair_points(
    type_a: str, id_a: str, type_b: str, id_b: str,
    lines: list[LineResult], arcs: list[ArcResult],
    pixel_size_mm: float,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Compute distance and geometry points between two features by ID.

    Returns:
        (distance_mm, point_a_px, point_b_px) or None if features not found.
    """
    line_map = {lr.id: lr for lr in lines}
    arc_map = {ar.id: ar for ar in arcs}

    def _lookup(ftype, fid):
        if ftype == "line":
            return line_map.get(fid)
        return arc_map.get(fid)

    obj_a = _lookup(type_a, id_a)
    obj_b = _lookup(type_b, id_b)
    if obj_a is None or obj_b is None:
        return None

    if type_a == "line" and type_b == "line":
        return distance_line_to_line(obj_a, obj_b, pixel_size_mm)
    if type_a == "arc" and type_b == "arc":
        return distance_arc_to_arc(obj_a, obj_b, pixel_size_mm)
    # Mixed: one line, one arc
    line = obj_a if type_a == "line" else obj_b
    arc = obj_b if type_b == "arc" else obj_a
    return distance_line_to_arc(line, arc, pixel_size_mm)


# ── Fuzzy ID matching for batch inspection ────────────────────────────────

import re

# Regex patterns for parsing IDs
_LINE_ID_RE = re.compile(
    r'^L_([A-Z][a-z])(\d+)_([A-Z][a-z])(\d+)_(\d+\.?\d*)_(\d+\.?\d*)$')
_ARC_ID_RE = re.compile(r'^C(\d+)_(\d+)_(\d+)$')


def parse_line_id(line_id: str) -> dict | None:
    """Parse a line ID string into components.

    Args:
        line_id: e.g. "L_Up8_Lo7_90.4_76.60"

    Returns:
        dict with keys: edge1, seg1, edge2, seg2, angle, length
        or None if parse fails.
    """
    m = _LINE_ID_RE.match(line_id)
    if not m:
        return None
    return {
        'edge1': m.group(1),
        'seg1': int(m.group(2)),
        'edge2': m.group(3),
        'seg2': int(m.group(4)),
        'angle': float(m.group(5)),
        'length': float(m.group(6)),
    }


def parse_arc_id(arc_id: str) -> dict | None:
    """Parse an arc ID string into components.

    Args:
        arc_id: e.g. "C4_13_4"

    Returns:
        dict with keys: row, col, radius_bucket
        or None if parse fails.
    """
    m = _ARC_ID_RE.match(arc_id)
    if not m:
        return None
    return {
        'row': int(m.group(1)),
        'col': int(m.group(2)),
        'radius_bucket': int(m.group(3)),
    }


def _score_line_match(template: dict, detected: LineResult,
                      img_w: int, img_h: int, pixel_size_mm: float,
                      segment_mm: float) -> float:
    """Score how well a detected line matches a parsed template.

    Lower score = better match. Returns infinity for impossible matches.

    Scoring weights:
    - Edge pair must match exactly (otherwise infinity)
    - Segment distance: weight 10 per segment difference
    - Angle difference: weight 2 per degree (tolerance ~5°)
    - Length ratio: weight 50 per 100% difference (tolerance ~15%)
    """
    # Compute detected line's edge intersections
    hits = _extend_line_to_edges(detected.start_px, detected.end_px, img_w, img_h)
    if len(hits) < 2:
        return float('inf')

    # Sort detected edges by canonical order for comparison
    det_edges = sorted([hits[0][0], hits[1][0]], key=lambda e: _EDGE_ORDER.index(e))
    tmpl_edges = sorted([template['edge1'], template['edge2']], key=lambda e: _EDGE_ORDER.index(e))

    # Edge pair must match
    if det_edges != tmpl_edges:
        return float('inf')

    # Compute detected segment numbers
    def seg_num(hit):
        name, x, y = hit
        if name in ('Up', 'Lo'):
            return _segment_number(x * pixel_size_mm, segment_mm)
        return _segment_number(y * pixel_size_mm, segment_mm)

    det_seg1 = seg_num(hits[0])
    det_seg2 = seg_num(hits[1])

    # Map detected segments to template segments by edge name
    tmpl_seg_map = {template['edge1']: template['seg1'], template['edge2']: template['seg2']}
    det_seg_map = {hits[0][0]: det_seg1, hits[1][0]: det_seg2}

    seg_dist = 0
    for edge in tmpl_seg_map:
        if edge in det_seg_map:
            seg_dist += abs(det_seg_map[edge] - tmpl_seg_map[edge])

    # Angle difference (handle wrap-around)
    angle_d = abs(detected.angle_deg - template['angle']) % 180
    angle_d = min(angle_d, 180 - angle_d)

    # Length ratio
    if template['length'] > 1:
        length_ratio = abs(detected.length_mm - template['length']) / template['length']
    else:
        length_ratio = abs(detected.length_mm - template['length'])

    # Combined score
    score = seg_dist * 10 + angle_d * 2 + length_ratio * 50
    return score


def _score_arc_match(template: dict, detected: ArcResult,
                     grid_size_mm: float) -> float:
    """Score how well a detected arc matches a parsed template.

    Scoring weights:
    - Grid cell distance: weight 10 per cell
    - Radius bucket difference: weight 5 per bucket
    """
    # Compute detected arc's grid cell
    det_row = int(detected.centroid_mm[1] // grid_size_mm)
    det_col = int(detected.centroid_mm[0] // grid_size_mm)
    det_bucket = int(detected.radius_mm)

    grid_dist = abs(det_row - template['row']) + abs(det_col - template['col'])
    radius_diff = abs(det_bucket - template['radius_bucket'])

    score = grid_dist * 10 + radius_diff * 5
    return score


def fuzzy_match_template_pairs(
    template_pairs: list[FeaturePair],
    detected_lines: list[LineResult],
    detected_arcs: list[ArcResult],
    img_w: int, img_h: int,
    pixel_size_mm: float,
    segment_mm: float,
    grid_size_mm: float,
    line_min_mm: float = 0.0,
    line_max_mm: float = float('inf'),
    arc_min_mm: float = 0.0,
    arc_max_mm: float = float('inf'),
) -> list[tuple[float | None, FeaturePair]]:
    """Match template pair IDs to detected features using fuzzy matching.

    Uses Hungarian assignment for 1-to-1 matching between template features
    and detected features, ensuring each detected feature matches at most one
    template feature. Detected features are pre-filtered by line length and
    arc radius criteria before matching.

    Args:
        template_pairs: List of FeaturePair from a saved template.
        detected_lines: List of LineResult from detection.
        detected_arcs: List of ArcResult from detection.
        img_w, img_h: Image dimensions.
        pixel_size_mm: mm per pixel.
        segment_mm: Edge segment size in mm.
        grid_size_mm: Grid cell size in mm.

    Returns:
        List of (distance_mm_or_None, updated_pair, pt_a_px_or_None, pt_b_px_or_None,
                  det_a_or_None, det_b_or_None) for each template pair.
        det_a/det_b are the matched LineResult or ArcResult objects.
        updated_pair has distance_mm filled in; None if matching failed.
    """
    if not template_pairs:
        return []

    # Collect all unique template feature IDs with their types
    template_features = {}  # (type, id) -> parsed dict or None
    for pair in template_pairs:
        key_a = (pair.type_a, pair.id_a)
        key_b = (pair.type_b, pair.id_b)
        if key_a not in template_features:
            if pair.type_a == 'line':
                template_features[key_a] = parse_line_id(pair.id_a)
            else:
                template_features[key_a] = parse_arc_id(pair.id_a)
        if key_b not in template_features:
            if pair.type_b == 'line':
                template_features[key_b] = parse_line_id(pair.id_b)
            else:
                template_features[key_b] = parse_arc_id(pair.id_b)

    # Separate lines and arcs
    tmpl_lines = [(k, v) for k, v in template_features.items() if k[0] == 'line']
    tmpl_arcs = [(k, v) for k, v in template_features.items() if k[0] == 'arc']

    # Pre-filter detected features by length/radius criteria
    filtered_lines = [l for l in detected_lines
                     if line_min_mm <= l.length_mm <= line_max_mm]
    filtered_arcs = [a for a in detected_arcs
                    if arc_min_mm <= a.radius_mm <= arc_max_mm]

    # Build cost matrix for lines
    line_assignment = {}
    if tmpl_lines and filtered_lines:
        n_tmpl = len(tmpl_lines)
        n_det = len(filtered_lines)
        cost = np.full((n_tmpl, n_det), 1e9)

        for i, (key, parsed) in enumerate(tmpl_lines):
            if parsed is None:
                continue
            for j, det in enumerate(filtered_lines):
                score = _score_line_match(parsed, det, img_w, img_h,
                                          pixel_size_mm, segment_mm)
                cost[i, j] = score

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e8:  # reasonable threshold
                line_assignment[tmpl_lines[r][0]] = filtered_lines[c]

    # Build cost matrix for arcs
    arc_assignment = {}
    if tmpl_arcs and filtered_arcs:
        n_tmpl = len(tmpl_arcs)
        n_det = len(filtered_arcs)
        cost = np.full((n_tmpl, n_det), 1e9)

        for i, (key, parsed) in enumerate(tmpl_arcs):
            if parsed is None:
                continue
            for j, det in enumerate(filtered_arcs):
                score = _score_arc_match(parsed, det, grid_size_mm)
                cost[i, j] = score

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e8:
                arc_assignment[tmpl_arcs[r][0]] = filtered_arcs[c]

    # Compute distances for each template pair
    results = []
    for pair in template_pairs:
        key_a = (pair.type_a, pair.id_a)
        key_b = (pair.type_b, pair.id_b)

        det_a = line_assignment.get(key_a) or arc_assignment.get(key_a)
        det_b = line_assignment.get(key_b) or arc_assignment.get(key_b)

        if det_a is None or det_b is None:
            results.append((None, pair, None, None, None, None))
            continue

        # Compute distance using the appropriate function
        if pair.type_a == 'line' and pair.type_b == 'line':
            dist, pt_a, pt_b = distance_line_to_line(det_a, det_b, pixel_size_mm)
        elif pair.type_a == 'arc' and pair.type_b == 'arc':
            dist, pt_a, pt_b = distance_arc_to_arc(det_a, det_b, pixel_size_mm)
        else:
            # Mixed
            line = det_a if pair.type_a == 'line' else det_b
            arc = det_b if pair.type_b == 'arc' else det_a
            dist, pt_line, pt_arc = distance_line_to_arc(line, arc, pixel_size_mm)
            # Map points back to pair order
            if pair.type_a == 'line':
                pt_a, pt_b = pt_line, pt_arc
            else:
                pt_a, pt_b = pt_arc, pt_line

        # Update the pair with computed distance
        updated = FeaturePair(
            type_a=pair.type_a, id_a=pair.id_a,
            type_b=pair.type_b, id_b=pair.id_b,
            distance_mm=dist,
            lower_mm=pair.lower_mm,
            upper_mm=pair.upper_mm,
        )
        results.append((dist, updated, pt_a, pt_b, det_a, det_b))

    return results


# ── Main detection pipeline ────────────────────────────────────────────

def detect_lines_and_arcs(
    image_bgr: np.ndarray,
    pixel_size_mm: float,
    gauss_sigma: float = 0.4,
    morph_radius: int = 3,
    grid_size_mm: float = 5.0,
    template_arc_grid: Optional[dict] = None,
    edge_segment_mm: float = 10.0,
    template_line_grid: Optional[dict] = None,
) -> LinesArcsResult:
    """Detect lines and arcs in an image.

    Args:
        image_bgr: BGR or grayscale image.
        pixel_size_mm: mm per pixel (from calibration or config).
        gauss_sigma: Gaussian smoothing before gradient.
        morph_radius: Closing/Opening radius before watershed.
        grid_size_mm: Grid cell size in mm for arc ID assignment.
        template_arc_grid: If provided, match ARC IDs to template grid map.
        edge_segment_mm: Segment length along image edges in mm for line IDs.
        template_line_grid: If provided, match LINE IDs to template grid map.

    Returns:
        LinesArcsResult with detected lines, arcs, and annotated image.
    """
    # Ensure 3-channel BGR
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.shape[2] == 1:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    # DIPlib watershed
    rgb = image_bgr[:, :, ::-1].copy()
    dip_img = dip.Image(rgb)
    dip_img.SetColorSpace('sRGB')
    dip_img.SetPixelSize([pixel_size_mm, pixel_size_mm], "mm")

    img = dip.Gauss(dip_img, gauss_sigma)
    img2 = dip.ColorSpaceManager.Convert(img, 'grey')
    gm = dip.Norm(dip.GradientMagnitude(img2))
    gm = dip.Opening(dip.Closing(gm, morph_radius), morph_radius)
    wlab = dip.Watershed(gm, connectivity=1, maxDepth=3,
                         flags={'correct', 'labels'})

    # Detect lines
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    raw_lines = detect_lines(gray)

    line_results = []
    for l in raw_lines:
        cat = _classify_line(l['angle'])
        mid_px = (l['start'] + l['end']) / 2
        line_results.append(LineResult(
            id="",  # assigned later by matching
            category=cat,
            start_px=l['start'],
            end_px=l['end'],
            length_mm=l['length'] * pixel_size_mm,
            angle_deg=l['angle'],
            centroid_mm=mid_px * pixel_size_mm,
        ))

    # Detect arcs for all objects
    msr = dip.MeasurementTool.Measure(wlab, wlab, ['Size'])
    size = msr['Size']
    all_obj_ids = [o for o in msr.Objects() if o != 0]
    all_obj_ids.sort(key=lambda o: -size[o][0])

    all_arcs = {}
    for obj_id in all_obj_ids:
        arcs = detect_arcs_for_object(obj_id, wlab, pixel_size_mm)
        if arcs:
            all_arcs[obj_id] = arcs

    merged_arcs = deduplicate_cross_object_arcs(all_arcs)

    arc_results = []
    for a in merged_arcs:
        arc_results.append(ArcResult(
            id="",  # assigned later by matching
            center_px=a['center'],
            radius_px=a['radius_px'],
            radius_mm=a['radius_mm'],
            centroid_mm=a['center'] * pixel_size_mm,
        ))

    # Assign line IDs via edge intersections
    img_h, img_w = image_bgr.shape[:2]
    line_grid = _assign_line_ids_by_edges(
        line_results, img_w, img_h, pixel_size_mm,
        edge_segment_mm, template_line_grid)

    # Assign arc IDs via grid cells
    arc_grid = _assign_arc_ids_by_grid(arc_results, grid_size_mm, template_arc_grid)

    return LinesArcsResult(
        lines=line_results,
        arcs=arc_results,
        arc_grid=arc_grid,
        line_grid=line_grid,
    )


# ── Annotation rendering ───────────────────────────────────────────────

LINE_COLORS = {"H": (0, 255, 0), "V": (0, 255, 255),
               "D": (0, 165, 255), "O": (255, 100, 100)}

ARC_PALETTE = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
               (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]


def render_measurement_overlay(
    canvas: np.ndarray,
    pt1: np.ndarray,
    pt2: np.ndarray,
    distance_mm: float,
    color: tuple = (0, 0, 255),
) -> None:
    """Draw a measurement line between two points with mm label.

    Args:
        canvas: BGR image to draw on (modified in-place).
        pt1: First point in pixel coordinates.
        pt2: Second point in pixel coordinates.
        distance_mm: Distance to display.
        color: Line color (default red).
    """
    pt1_int = pt1.astype(int)
    pt2_int = pt2.astype(int)

    cv2.line(canvas, tuple(pt1_int), tuple(pt2_int), color, 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pt1_int), 5, color, -1)
    cv2.circle(canvas, tuple(pt2_int), 5, color, -1)

    mid = ((pt1_int + pt2_int) // 2).astype(int)
    direction = (pt2_int - pt1_int).astype(float)
    length = np.linalg.norm(direction)
    if length > 0:
        direction = direction / length
    perp = np.array([-direction[1], direction[0]])

    label = f"{distance_mm:.3f} mm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)

    offset = (perp * 15).astype(int)
    tx, ty = mid[0] + offset[0], mid[1] + offset[1]

    if tx - tw // 2 < 0:
        tx = tw // 2 + 2
    elif tx + tw // 2 > canvas.shape[1]:
        tx = canvas.shape[1] - tw // 2 - 2
    if ty - th < 0:
        ty = th + 10
    elif ty > canvas.shape[0]:
        ty = canvas.shape[0] - 5

    cv2.rectangle(canvas, (tx - tw // 2 - 3, ty - th - 3),
                  (tx + tw // 2 + 3, ty + 5), (0, 0, 0), -1)
    cv2.putText(canvas, label, (tx - tw // 2, ty),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def render_annotations(
    image_bgr: np.ndarray,
    lines: list[LineResult],
    arcs: list[ArcResult],
    highlight_type: str = "",
    highlight_id: str = "",
    measurement_points: tuple | None = None,
) -> np.ndarray:
    """Draw line and arc annotations onto a copy of image_bgr.

    Args:
        image_bgr: Original BGR image.
        lines: Line results to draw.
        arcs: Arc results to draw.
        highlight_type: "line" or "arc" to highlight one feature.
        highlight_id: ID of the feature to highlight.
        measurement_points: Optional (distance_mm, pt_a_px, pt_b_px) for
            measurement overlay.

    Returns:
        Annotated BGR image.
    """
    canvas = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for lr in lines:
        color = LINE_COLORS[lr.category]
        p1 = lr.start_px.astype(int)
        p2 = lr.end_px.astype(int)
        cv2.line(canvas, tuple(p1), tuple(p2), color, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(p1), 4, color, -1)
        cv2.circle(canvas, tuple(p2), 4, color, -1)
        mid = ((p1 + p2) // 2)
        label = f"{lr.id}({lr.category}): {lr.length_mm:.1f}mm {lr.angle_deg:.1f}deg"
        (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
        cv2.rectangle(canvas, (mid[0] - 2, mid[1] - th - 4),
                       (mid[0] + tw + 2, mid[1] + 4), (0, 0, 0), -1)
        cv2.putText(canvas, label, (mid[0], mid[1]),
                    font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    for i, ar in enumerate(arcs):
        cx, cy = int(ar.center_px[0]), int(ar.center_px[1])
        r = int(ar.radius_px)
        color = ARC_PALETTE[i % len(ARC_PALETTE)]
        cv2.circle(canvas, (cx, cy), r, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, color, -1)
        label = f"{ar.id}: r={ar.radius_mm:.2f}mm"
        (tw, th), _ = cv2.getTextSize(label, font, 0.38, 1)
        tx, ty = cx + 8, cy - 8
        if tx + tw + 4 > canvas.shape[1]:
            tx = cx - tw - 12
        if ty - th - 4 < 0:
            ty = cy + 20
        cv2.rectangle(canvas, (tx - 2, ty - th - 4),
                       (tx + tw + 2, ty + 4), (0, 0, 0), -1)
        cv2.putText(canvas, label, (tx, ty),
                    font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Highlight overlay
    if highlight_type == "arc" and highlight_id:
        for i, ar in enumerate(arcs):
            if ar.id == highlight_id:
                cx, cy = int(ar.center_px[0]), int(ar.center_px[1])
                r = int(ar.radius_px)
                color = ARC_PALETTE[i % len(ARC_PALETTE)]
                overlay = canvas.copy()
                cv2.circle(overlay, (cx, cy), r, color, -1)
                cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, dst=canvas)
                break

    elif highlight_type == "line" and highlight_id:
        for lr in lines:
            if lr.id == highlight_id:
                color = LINE_COLORS.get(lr.category, (255, 255, 255))
                p1 = lr.start_px.astype(int)
                p2 = lr.end_px.astype(int)
                mid = ((p1 + p2) // 2)
                direction = (p2 - p1).astype(float)
                length = np.linalg.norm(direction)
                if length > 0:
                    direction /= length
                perp = np.array([-direction[1], direction[0]])
                arrow_start = (mid + (perp * 30)).astype(int)
                cv2.arrowedLine(canvas, tuple(arrow_start), tuple(mid),
                                color, 3, tipLength=0.4)
                cv2.line(canvas, tuple(p1), tuple(p2), color, 3, cv2.LINE_AA)
                break

    # Measurement overlay
    if measurement_points is not None:
        dist_mm, pt_a, pt_b = measurement_points
        render_measurement_overlay(canvas, pt_a, pt_b, dist_mm)

    return canvas


# ── Async worker for UI ─────────────────────────────────────────────────

class LinesArcsWorker(QThread):
    """Run line/arc detection in a background thread."""

    done = Signal(object)   # LinesArcsResult on success
    error = Signal(str)

    def __init__(self, image_bgr: np.ndarray, pixel_size_mm: float,
                 gauss_sigma: float = 0.4, morph_radius: int = 3,
                 grid_size_mm: float = 5.0,
                 template_arc_grid: Optional[dict] = None,
                 edge_segment_mm: float = 10.0,
                 template_line_grid: Optional[dict] = None):
        super().__init__()
        self._image = image_bgr.copy()
        self._pixel_size_mm = pixel_size_mm
        self._gauss_sigma = gauss_sigma
        self._morph_radius = morph_radius
        self._grid_size_mm = grid_size_mm
        self._template_arc_grid = template_arc_grid
        self._edge_segment_mm = edge_segment_mm
        self._template_line_grid = template_line_grid

    def run(self):
        try:
            result = detect_lines_and_arcs(
                self._image, self._pixel_size_mm,
                gauss_sigma=self._gauss_sigma,
                morph_radius=self._morph_radius,
                grid_size_mm=self._grid_size_mm,
                template_arc_grid=self._template_arc_grid,
                edge_segment_mm=self._edge_segment_mm,
                template_line_grid=self._template_line_grid,
            )
            self.done.emit(result)
        except Exception as e:
            self.error.emit(f"Line/arc detection failed: {e}")
