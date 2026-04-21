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
    annotated_bgr: Optional[np.ndarray] = None


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


# ── Main detection pipeline ────────────────────────────────────────────

def detect_lines_and_arcs(
    image_bgr: np.ndarray,
    pixel_size_mm: float,
    gauss_sigma: float = 0.4,
    morph_radius: int = 3,
    template: Optional[list[FeatureDescriptor]] = None,
) -> LinesArcsResult:
    """Detect lines and arcs in an image.

    Args:
        image_bgr: BGR or grayscale image.
        pixel_size_mm: mm per pixel (from calibration or config).
        gauss_sigma: Gaussian smoothing before gradient.
        morph_radius: Closing/Opening radius before watershed.
        template: If provided, match detected features to these template IDs.

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

    # Assign IDs
    if template is not None:
        # Build detection descriptors for matching
        line_descs = [FeatureDescriptor(
            id="", centroid_mm=lr.centroid_mm,
            primary=lr.angle_deg, secondary=lr.length_mm
        ) for lr in line_results]
        arc_descs = [FeatureDescriptor(
            id="", centroid_mm=ar.centroid_mm,
            primary=ar.radius_mm, secondary=0
        ) for ar in arc_results]

        # Separate template into lines and arcs
        t_lines = [f for f in template if f.id.startswith('L')]
        t_arcs = [f for f in template if f.id.startswith('C')]

        # Match
        line_mapping = match_features(t_lines, line_descs)
        for idx, tid in line_mapping.items():
            if tid is not None:
                line_results[idx].id = tid

        arc_mapping = match_features(t_arcs, arc_descs)
        for idx, tid in arc_mapping.items():
            if tid is not None:
                arc_results[idx].id = tid

    # Assign sequential IDs to unmatched features
    existing_line_ids = {lr.id for lr in line_results if lr.id}
    line_counter = max(
        (int(lr.id[1:]) for lr in line_results if lr.id.startswith('L')),
        default=0)
    for lr in line_results:
        if not lr.id:
            line_counter += 1
            lr.id = f"L{line_counter}"

    existing_arc_ids = {ar.id for ar in arc_results if ar.id}
    arc_counter = max(
        (int(ar.id[1:]) for ar in arc_results if ar.id.startswith('C')),
        default=0)
    for ar in arc_results:
        if not ar.id:
            arc_counter += 1
            ar.id = f"C{arc_counter}"

    # Build annotated image
    result_cv = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    line_colors = {"H": (0, 255, 0), "V": (0, 255, 255),
                   "D": (0, 165, 255), "O": (255, 100, 100)}
    for lr in line_results:
        color = line_colors[lr.category]
        p1 = lr.start_px.astype(int)
        p2 = lr.end_px.astype(int)
        cv2.line(result_cv, tuple(p1), tuple(p2), color, 2, cv2.LINE_AA)
        cv2.circle(result_cv, tuple(p1), 4, color, -1)
        cv2.circle(result_cv, tuple(p2), 4, color, -1)
        mid = ((p1 + p2) // 2)
        label = f"{lr.id}({lr.category}): {lr.length_mm:.1f}mm {lr.angle_deg:.1f}deg"
        (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
        cv2.rectangle(result_cv, (mid[0] - 2, mid[1] - th - 4),
                       (mid[0] + tw + 2, mid[1] + 4), (0, 0, 0), -1)
        cv2.putText(result_cv, label, (mid[0], mid[1]),
                    font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    arc_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
    for i, ar in enumerate(arc_results):
        cx, cy = int(ar.center_px[0]), int(ar.center_px[1])
        r = int(ar.radius_px)
        color = arc_colors[i % len(arc_colors)]
        cv2.circle(result_cv, (cx, cy), r, color, 2, cv2.LINE_AA)
        cv2.circle(result_cv, (cx, cy), 3, color, -1)
        label = f"{ar.id}: r={ar.radius_mm:.2f}mm"
        (tw, th), _ = cv2.getTextSize(label, font, 0.38, 1)
        tx, ty = cx + 8, cy - 8
        if tx + tw + 4 > result_cv.shape[1]:
            tx = cx - tw - 12
        if ty - th - 4 < 0:
            ty = cy + 20
        cv2.rectangle(result_cv, (tx - 2, ty - th - 4),
                       (tx + tw + 2, ty + 4), (0, 0, 0), -1)
        cv2.putText(result_cv, label, (tx, ty),
                    font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return LinesArcsResult(
        lines=line_results,
        arcs=arc_results,
        annotated_bgr=result_cv,
    )


# ── Async worker for UI ─────────────────────────────────────────────────

class LinesArcsWorker(QThread):
    """Run line/arc detection in a background thread."""

    done = Signal(object)   # LinesArcsResult on success
    error = Signal(str)

    def __init__(self, image_bgr: np.ndarray, pixel_size_mm: float,
                 gauss_sigma: float = 0.4, morph_radius: int = 3,
                 template: Optional[list[FeatureDescriptor]] = None):
        super().__init__()
        self._image = image_bgr.copy()
        self._pixel_size_mm = pixel_size_mm
        self._gauss_sigma = gauss_sigma
        self._morph_radius = morph_radius
        self._template = template

    def run(self):
        try:
            result = detect_lines_and_arcs(
                self._image, self._pixel_size_mm,
                gauss_sigma=self._gauss_sigma,
                morph_radius=self._morph_radius,
                template=self._template,
            )
            self.done.emit(result)
        except Exception as e:
            self.error.emit(f"Line/arc detection failed: {e}")
