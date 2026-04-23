"""
RANSAC-based rigid image registration using detected lines and arcs.

Provides an independent alignment playground for evaluating how well a
product image can be registered to a template image using geometric features.
Does not modify any detection results, metrology pairs, or ID assignments.

Rigid transform model: 2D rotation + translation (no scaling or shear).
"""

import random
import cv2
import numpy as np
from dataclasses import dataclass, field


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RigidTransform:
    """2D rigid transformation: rotation + translation in mm space."""
    rotation_deg: float
    translation_mm: np.ndarray  # (2,)


@dataclass
class AlignmentResult:
    """Result of RANSAC rigid registration."""
    transform: RigidTransform
    inlier_count: int
    total_features: int
    residual_rms_mm: float
    inlier_pairs: list = field(default_factory=list)
    # Each inlier_pair: (tmpl_centroid_mm, det_centroid_mm, feature_type_str)


# ── Transform helpers ───────────────────────────────────────────────────────

def apply_transform(points_mm: np.ndarray, transform: RigidTransform) -> np.ndarray:
    """Apply rigid transform to an array of points (N, 2) in mm."""
    angle_rad = np.radians(transform.rotation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (R @ points_mm.T).T + transform.translation_mm


def _apply_transform_single(point_mm: np.ndarray, transform: RigidTransform) -> np.ndarray:
    """Apply rigid transform to a single point (2,) in mm."""
    angle_rad = np.radians(transform.rotation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return R @ point_mm + transform.translation_mm


# ── Candidate generation ───────────────────────────────────────────────────

def _candidate_from_line_pair(
    tmpl_angle_deg: float, tmpl_centroid_mm: np.ndarray,
    det_angle_deg: float, det_centroid_mm: np.ndarray,
) -> RigidTransform:
    """Generate candidate transform from a line–line correspondence.

    Rotation = angle difference (handle wrap-around for unsigned line angles).
    Translation = template centroid minus rotated detected centroid.
    """
    # Line angles are unsigned (0–180). Use cross-product sign from endpoints
    # to determine signed rotation direction. For simplicity, try both
    # directions and let scoring pick the better one.
    angle_diff = (det_angle_deg - tmpl_angle_deg) % 180
    # Try the smaller rotation
    if angle_diff > 90:
        angle_diff -= 180
    rotation_deg = -angle_diff
    t = RigidTransform(
        rotation_deg=rotation_deg,
        translation_mm=np.array([0.0, 0.0]),
    )
    rotated_det = _apply_transform_single(det_centroid_mm, t)
    t.translation_mm = tmpl_centroid_mm - rotated_det
    return t


def _candidate_from_arc_pair(
    tmpl_centroid_mm: np.ndarray,
    det_centroid_mm: np.ndarray,
) -> RigidTransform:
    """Generate candidate transform from an arc–arc correspondence.

    Arcs alone don't provide rotation information. This produces a pure
    translation from center offset.
    """
    return RigidTransform(
        rotation_deg=0.0,
        translation_mm=tmpl_centroid_mm - det_centroid_mm,
    )


# ── Scoring ─────────────────────────────────────────────────────────────────

def _score_transform(
    transform: RigidTransform,
    tmpl_centroids_mm: list[tuple[np.ndarray, str, float]],
    det_centroids_mm: list[tuple[np.ndarray, str, float]],
    inlier_threshold_mm: float = 5.0,
) -> tuple[int, float, list]:
    """Score a candidate transform by counting inlier pairs.

    Args:
        tmpl_centroids_mm: [(centroid_mm, type_str, extra_val), ...]
            For lines: extra_val = angle_deg
            For arcs:  extra_val = radius_mm
        det_centroids_mm: same format

    Returns:
        (inlier_count, residual_rms, inlier_pairs)
    """
    inlier_pairs = []
    residuals = []
    used_tmpl = set()

    for d_idx, (det_c, det_type, det_val) in enumerate(det_centroids_mm):
        transformed_det = _apply_transform_single(det_c, transform)
        best_dist = float('inf')
        best_t_idx = -1

        for t_idx, (tmpl_c, tmpl_type, tmpl_val) in enumerate(tmpl_centroids_mm):
            if t_idx in used_tmpl:
                continue
            if tmpl_type != det_type:
                continue
            dist = np.linalg.norm(transformed_det - tmpl_c)
            if dist < best_dist:
                best_dist = dist
                best_t_idx = t_idx

        if best_dist < inlier_threshold_mm and best_t_idx >= 0:
            used_tmpl.add(best_t_idx)
            inlier_pairs.append((
                tmpl_centroids_mm[best_t_idx][0],  # tmpl centroid
                det_c,                               # det centroid
                det_type,
            ))
            residuals.append(best_dist)

    rms = float(np.sqrt(np.mean(np.array(residuals) ** 2))) if residuals else float('inf')
    return len(inlier_pairs), rms, inlier_pairs


# ── Least-squares refinement ────────────────────────────────────────────────

def _refine_transform(
    inlier_pairs: list,
    initial: RigidTransform,
) -> RigidTransform:
    """Refine rigid transform using SVD-based Procrustes on inlier pairs.

    Given matched point correspondences, finds the optimal rotation and
    translation that minimizes sum of squared distances.
    """
    if len(inlier_pairs) < 2:
        return initial

    tmpl_pts = np.array([p[0] for p in inlier_pairs])
    det_pts = np.array([p[1] for p in inlier_pairs])

    # Center
    tmpl_center = tmpl_pts.mean(axis=0)
    det_center = det_pts.mean(axis=0)
    tmpl_c = tmpl_pts - tmpl_center
    det_c = det_pts - det_center

    # SVD for optimal rotation
    H = det_c.T @ tmpl_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    translation_mm = tmpl_center - R @ det_center

    return RigidTransform(rotation_deg=rotation_deg, translation_mm=translation_mm)


# ── Main RANSAC loop ────────────────────────────────────────────────────────

def ransac_rigid_registration(
    tmpl_lines, tmpl_arcs,
    det_lines, det_arcs,
    pixel_size_mm: float,
    n_iterations: int = 200,
    inlier_threshold_mm: float = 5.0,
    min_inliers: int = 3,
    radius_tolerance_ratio: float = 0.15,
    angle_tolerance_deg: float = 20.0,
) -> AlignmentResult | None:
    """RANSAC-based rigid registration using detected lines and arcs.

    Generates candidate transforms from feature pairs, scores each by
    counting inliers, picks the best, and refines with least-squares.

    Args:
        tmpl_lines: Template LineResult list.
        tmpl_arcs: Template ArcResult list.
        det_lines: Detected LineResult list.
        det_arcs: Detected ArcResult list.
        pixel_size_mm: mm per pixel.
        n_iterations: Number of RANSAC iterations.
        inlier_threshold_mm: Max distance for inlier classification.
        min_inliers: Minimum inliers for a valid result.
        radius_tolerance_ratio: Max radius difference ratio for arc pairing.
        angle_tolerance_deg: Max angle difference for line pairing.

    Returns:
        AlignmentResult or None if registration fails.
    """
    # Build centroid lists with metadata
    tmpl_centroids = []
    for lr in tmpl_lines:
        tmpl_centroids.append((lr.centroid_mm.copy(), 'line', lr.angle_deg))
    for ar in tmpl_arcs:
        tmpl_centroids.append((ar.centroid_mm.copy(), 'arc', ar.radius_mm))

    det_centroids = []
    for lr in det_lines:
        det_centroids.append((lr.centroid_mm.copy(), 'line', lr.angle_deg))
    for ar in det_arcs:
        det_centroids.append((ar.centroid_mm.copy(), 'arc', ar.radius_mm))

    if not tmpl_centroids or not det_centroids:
        return None

    best_count = 0
    best_rms = float('inf')
    best_transform = None
    best_pairs = []

    # Filter candidate pairs by angle/radius tolerance
    valid_line_pairs = []
    for t_idx, tl in enumerate(tmpl_lines):
        for d_idx, dl in enumerate(det_lines):
            ad = abs(tl.angle_deg - dl.angle_deg) % 180
            ad = min(ad, 180 - ad)
            if ad < angle_tolerance_deg:
                valid_line_pairs.append((t_idx, d_idx))

    valid_arc_pairs = []
    for t_idx, ta in enumerate(tmpl_arcs):
        for d_idx, da in enumerate(det_arcs):
            r_max = max(ta.radius_mm, da.radius_mm, 0.1)
            if abs(ta.radius_mm - da.radius_mm) / r_max < radius_tolerance_ratio:
                valid_arc_pairs.append((t_idx, d_idx))

    all_candidates = (
        [('line', t, d) for t, d in valid_line_pairs] +
        [('arc', t, d) for t, d in valid_arc_pairs]
    )

    if not all_candidates:
        return None

    for _ in range(n_iterations):
        feat_type, t_idx, d_idx = random.choice(all_candidates)

        if feat_type == 'line':
            tl = tmpl_lines[t_idx]
            dl = det_lines[d_idx]
            candidate = _candidate_from_line_pair(
                tl.angle_deg, tl.centroid_mm,
                dl.angle_deg, dl.centroid_mm,
            )
        else:
            ta = tmpl_arcs[t_idx]
            da = det_arcs[d_idx]
            candidate = _candidate_from_arc_pair(
                ta.centroid_mm, da.centroid_mm,
            )

        count, rms, pairs = _score_transform(
            candidate, tmpl_centroids, det_centroids,
            inlier_threshold_mm,
        )

        if count > best_count or (count == best_count and rms < best_rms):
            best_count = count
            best_rms = rms
            best_transform = candidate
            best_pairs = pairs

    if best_transform is None or best_count < min_inliers:
        return None

    # Refine with least-squares on inliers
    refined = _refine_transform(best_pairs, best_transform)

    # Re-score with refined transform
    final_count, final_rms, final_pairs = _score_transform(
        refined, tmpl_centroids, det_centroids, inlier_threshold_mm,
    )

    return AlignmentResult(
        transform=refined,
        inlier_count=final_count,
        total_features=len(tmpl_centroids) + len(det_centroids),
        residual_rms_mm=round(final_rms, 4),
        inlier_pairs=final_pairs,
    )


# ── Rendering ───────────────────────────────────────────────────────────────

def render_alignment_overlay(
    image_bgr: np.ndarray,
    result: AlignmentResult,
    tmpl_lines, tmpl_arcs,
    det_lines, det_arcs,
    pixel_size_mm: float,
) -> np.ndarray:
    """Draw alignment visualization on image.

    Green: template features.  Blue: detected features.
    Red: detected features after computed transform.
    Yellow lines: inlier correspondences.
    """
    canvas = image_bgr.copy()
    px = pixel_size_mm

    # Draw template features (green)
    for lr in tmpl_lines:
        p1 = (lr.start_px / px * px).astype(int)  # already in px
        p2 = (lr.end_px / px * px).astype(int)
        cv2.line(canvas, tuple(lr.start_px.astype(int)),
                 tuple(lr.end_px.astype(int)), (0, 200, 0), 2, cv2.LINE_AA)
    for ar in tmpl_arcs:
        cx, cy = int(ar.center_px[0]), int(ar.center_px[1])
        cv2.circle(canvas, (cx, cy), int(ar.radius_px), (0, 200, 0), 2, cv2.LINE_AA)

    # Draw detected features (blue)
    for lr in det_lines:
        cv2.line(canvas, tuple(lr.start_px.astype(int)),
                 tuple(lr.end_px.astype(int)), (200, 100, 0), 2, cv2.LINE_AA)
    for ar in det_arcs:
        cx, cy = int(ar.center_px[0]), int(ar.center_px[1])
        cv2.circle(canvas, (cx, cy), int(ar.radius_px), (200, 100, 0), 2, cv2.LINE_AA)

    # Draw transformed detected features (red)
    for lr in det_lines:
        s_mm = lr.start_px * px
        e_mm = lr.end_px * px
        s_t = _apply_transform_single(s_mm, result.transform)
        e_t = _apply_transform_single(e_mm, result.transform)
        cv2.line(canvas, tuple((s_t / px).astype(int)),
                 tuple((e_t / px).astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    for ar in det_arcs:
        c_mm = ar.center_px * px
        c_t = _apply_transform_single(c_mm, result.transform)
        cv2.circle(canvas, tuple((c_t / px).astype(int)),
                   int(ar.radius_px), (0, 0, 255), 2, cv2.LINE_AA)

    # Draw inlier correspondence lines (yellow)
    for tmpl_c, det_c, _ in result.inlier_pairs:
        det_t = _apply_transform_single(det_c, result.transform)
        pt1 = tuple((tmpl_c / px).astype(int))
        pt2 = tuple((det_t / px).astype(int))
        cv2.line(canvas, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

    # Info text box
    t = result.transform
    lines = [
        f"Rotation: {t.rotation_deg:.2f} deg",
        f"Translation: ({t.translation_mm[0]:.2f}, {t.translation_mm[1]:.2f}) mm",
        f"Inliers: {result.inlier_count}",
        f"Residual RMS: {result.residual_rms_mm:.3f} mm",
    ]
    y = 30
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (10, y - th - 4), (10 + tw + 8, y + 4), (0, 0, 0), -1)
        cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += th + 10

    # Legend
    img_h, img_w = canvas.shape[:2]
    legend_y = img_h - 20
    for label, color in [("Template", (0, 200, 0)), ("Detected", (200, 100, 0)),
                          ("Transformed", (0, 0, 255)), ("Inlier link", (0, 255, 255))]:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(canvas, (10, legend_y - th - 2), (10 + tw + 6, legend_y + 3),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (12, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
        legend_y -= th + 8

    return canvas
