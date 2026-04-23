# Fuzzy Feature Matching for Template-Based Metrology

## Purpose

In production quality assurance, manufactured workpieces are measured repeatedly
against a saved template. Each measurement captures a new camera frame, detects
geometric features (lines and arcs), and compares distances between matched
features to the template's nominal values.

The challenge: the workpiece shifts and rotates slightly between placements.
Feature IDs change across runs because they encode spatial position. The **fuzzy
matching algorithm** bridges this gap by matching template feature IDs to newly
detected features using spatial proximity and geometric similarity, without
requiring exact ID equality.

---

## Feature ID Systems

Two complementary ID systems provide the spatial anchors that make fuzzy matching
possible.

### Arc IDs: Grid-Cell Assignment

Arcs (curves detected via curvature analysis and RANSAC circle fitting) are
assigned IDs based on a uniform grid overlaid on the image in millimeter space.

**ID format:** `C{row}_{col}_{radius_bucket}`

| Component       | Meaning                                       |
|-----------------|-----------------------------------------------|
| `row`           | Grid row index: `y_mm // grid_size_mm`        |
| `col`           | Grid column index: `x_mm // grid_size_mm`     |
| `radius_bucket` | Integer part of radius in mm (`int(radius_mm)`) |

**Example:** `C4_13_4` means the arc centroid is in grid row 4, column 13, with
radius approximately 4 mm.

**Configurable parameter:** `grid_size_mm` (default 5.0 mm) controls the grid
cell size. A larger grid tolerates more positional offset but risks merging
nearby arcs into the same cell.

**How grid cells tolerate offset:**

If a workpiece shifts 3 mm horizontally, an arc at column 13 might move to
column 12 or 14. The fuzzy matcher accepts a ±1 cell difference (configurable
via the scoring weight), so the arc still matches its template entry. With a
5 mm grid, the system tolerates roughly ±5 mm of positional shift before a
feature migrates two cells away.

### Line IDs: Edge-Segment Assignment

Lines (detected via LSD and collinear merging) are assigned IDs based on where
they intersect the image frame edges.

**ID format:** `L_{edge1}{seg1}_{edge2}{seg2}_{angle}_{length_mm}`

| Component | Meaning                                                    |
|-----------|------------------------------------------------------------|
| `edge1`   | First intersected edge name: `Up`, `Lo`, `Le`, or `Ri`    |
| `seg1`    | 1-indexed segment number along that edge                   |
| `edge2`   | Second intersected edge name                               |
| `seg2`    | 1-indexed segment number along that edge                   |
| `angle`   | Line angle in degrees (0–180)                              |
| `length`  | Line length in mm (rounded to 2 decimal places)            |

**Example:** `L_Up8_Lo7_90.4_76.60` is a vertical line crossing through segment
8 of the top edge and segment 7 of the bottom edge, at 90.4° and 76.60 mm long.

**Configurable parameter:** `edge_segment_mm` (default 10.0 mm) controls how
finely each edge is divided into numbered segments.

**How segments tolerate offset:**

If a line crosses the top edge at segment 8 on the template and segment 7 on
the product (due to a small horizontal shift), the fuzzy matcher's segment
distance score adds a penalty but does not reject the match. The segment
scoring weight (10 per segment difference) allows ±1–2 segment differences
while still preferring closer matches.

---

## Scoring Functions

The fuzzy matcher builds a **cost matrix** where each cell contains a score
representing how well a detected feature matches a template feature. Lower
scores mean better matches.

### Line Matching Score

`_score_line_match(template_parsed, detected_line)` computes:

| Criterion         | Weight | Tolerance               | Effect                          |
|-------------------|--------|-------------------------|---------------------------------|
| Edge pair match   | —      | Must match exactly      | Reject if different edge names  |
| Segment distance  | 10     | Per segment difference  | Allows ±1–2 segment drift      |
| Angle difference  | 2      | Per degree              | Allows ~5° orientation change   |
| Length ratio      | 50     | Per 100% deviation      | Allows ~15% length variation    |

**Formula:** `score = seg_dist * 10 + angle_diff * 2 + length_ratio * 50`

A short noisy line crossing different edges returns infinity (no match). A
genuine line that shifted slightly gets a small penalty proportional to the
segment drift.

### Arc Matching Score

`_score_arc_match(template_parsed, detected_arc)` computes:

| Criterion           | Weight | Tolerance              | Effect                          |
|---------------------|--------|------------------------|---------------------------------|
| Grid cell distance  | 10     | Manhattan distance     | Allows ±1–2 cell drift          |
| Radius bucket diff  | 5      | Per integer mm         | Allows ~1–2 mm radius variation |

**Formula:** `score = grid_dist * 10 + radius_diff * 5`

---

## Hungarian Assignment

After scoring all template–detected pairs, the algorithm uses the **Hungarian
method** (`scipy.optimize.linear_sum_assignment`) to find the optimal 1-to-1
assignment that minimizes total cost.

**Why Hungarian?** Without it, two template features near the same detected
feature could both claim it. The Hungarian method ensures each detected feature
is assigned to at most one template feature, and vice versa, producing a
globally optimal matching.

**Process:**
1. Build cost matrix: template features (rows) × detected features (columns)
2. Set impossible matches to 1e9 (infinity)
3. `linear_sum_assignment(cost)` returns the optimal row-column assignment
4. Discard assignments with cost ≥ 1e8 (unreasonable matches)

---

## Feature Pair Workflow

### 1. Template Creation

The user runs detection on a reference workpiece. In the Detection Results
window, they Ctrl-click two features (one line or arc each) to create a
**Feature Pair** — a named measurement relationship:

```
FeaturePair:
  type_a = "arc",   id_a = "C4_14_4"
  type_b = "arc",   id_b = "C5_5_4"
  distance_mm = 45.95    # measured distance between features
  lower_mm = 0.0         # tolerance lower bound
  upper_mm = 0.0         # tolerance upper bound
```

Multiple pairs can be saved as a named configuration (product template).

### 2. Production Matching

When a new product is placed on the fixture:

1. **Detect** lines and arcs on the new frame
2. **Filter** by line length and arc radius criteria (from Detection Results page)
3. **Match** template feature IDs to detected features using fuzzy scoring + Hungarian assignment
4. **Measure** distances between matched pairs
5. **Evaluate** pass/fail against tolerance bounds

### 3. Filter Criteria Integration

The line length (`line_min_mm`–`line_max_mm`) and arc radius
(`arc_min_mm`–`arc_max_mm`) filter criteria from the Detection Results page are
applied **before** fuzzy matching. Short noisy lines and tiny arc fragments are
excluded from the cost matrix, preventing false matches.

---

## Data Flow Diagram

```
Camera Frame
     │
     ▼
Watershed Segmentation → Object Boundaries
     │
     ├── LSD Line Detection → Merge → LineResult list
     │                              (start_px, end_px, length_mm, angle_deg, centroid_mm)
     │
     └── Curvature Analysis → RANSAC Circle Fit → ArcResult list
                                      (center_px, radius_px, radius_mm, centroid_mm)
     │
     ▼
ID Assignment (Grid + Edge-Segments)
     │
     ├── Lines: _assign_line_ids_by_edges → "L_Up8_Lo7_90.4_76.60"
     └── Arcs:  _assign_arc_ids_by_grid   → "C4_13_4"
     │
     ▼
Filter by line_min_mm/line_max_mm and arc_min_mm/arc_max_mm
     │
     ▼
Fuzzy Matching (Hungarian Assignment)
     │
     ├── _score_line_match: edge pair + segment + angle + length
     └── _score_arc_match:  grid cell + radius bucket
     │
     ▼
Distance Measurement
     │
     ├── distance_line_to_line: perpendicular distance
     ├── distance_arc_to_arc:   center-to-center distance
     └── distance_line_to_arc:  perpendicular from arc center to line
     │
     ▼
Pass/Fail Evaluation (lower_mm ≤ distance ≤ upper_mm)
```

---

## Key Design Decisions

- **Grid and segments are tolerance mechanisms**: They quantize continuous
  positions into discrete bins, providing natural tolerance windows. A 5 mm grid
  means ±2.5 mm of positional shift keeps a feature in the same cell.

- **Hungarian prevents double-booking**: Without 1-to-1 constraints, the
  closest detected feature would be claimed by multiple template features,
  producing incorrect distance measurements.

- **Filter criteria gate the matcher**: Short lines from noise or edge
  fragments never enter the cost matrix. This is more robust than relying on
  scoring penalties alone, since a short line in the right grid cell could
  otherwise outscore a genuine long line.

- **ID inheritance across runs**: On subsequent detections, the system first
  tries to match features to the template grid/segment map. If a feature falls
  in the same cell with similar geometry, it inherits the template ID. This
  makes IDs stable across runs, enabling the template matching workflow.
