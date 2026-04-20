# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine-vision quality assurance project for detecting and analyzing a groove feature on a manufactured workpiece. Two standalone Python scripts implement alternative segmentation approaches, both operating on `screenshot.png` as input.

## Running the Scripts

```bash
# SAM2 pipeline (requires CUDA-capable GPU for reasonable speed)
python sam2_autoprompt2.py

# Watershed pipeline (CPU only, uses DIPlib)
python watershed2.py
```

No build step, test suite, linter, or package manager is configured. Dependencies must be installed manually: `sam2`, `torch`, `opencv-python`, `open3d`, `diplib`, `matplotlib`, `pillow`, `numpy`.

## Architecture

### `sam2_autoprompt2.py` — SAM2-based groove segmentation

Pipeline: Canny edge detection → coarse bounding box (`auto_object_bbox`) → adaptive-threshold tightening (`tighten_bbox`) → 10% shrink (`shrink_bbox_percent`) → SAM2 inference with box prompt → mask extraction → point cloud conversion → ICP template matching.

Key functions:
- `auto_object_bbox` / `tighten_bbox` — two-stage bounding box detection (coarse via Canny, tight via adaptive threshold)
- `groove_prompt_points` — generates positive/negative point prompts for the groove (computed but currently unused; SAM2 is called with `point_coords=None`)
- `mask_to_pointcloud` — samples 500 contour points, lifts to 3D (z=0), saves as PLY
- `icp_distance` / `match_template` — point-to-point ICP registration for comparing segmented mask against a template

External dependencies:
- SAM2 checkpoint: `/home/hotcat/Downloads/sam2/checkpoints/sam2.1_hiera_large.pt`
- SAM2 config: `configs/sam2.1/sam2.1_hiera_l.yaml` (relative to SAM2 repo)
- Template mask: `template/mask_A.png` (does not exist in this directory)
- Mask output: `/home/hotcat/Downloads/albumentation/mask/img_0001.png`
- Point cloud output: `mask_pc.ply`

### `watershed2.py` — Watershed-based circular object detection

Pipeline: Gaussian smoothing → gradient magnitude → morphological close+open → watershed → small/edge object removal → measure roundness → select object with roundness > 0.95 → draw circle overlay.

Uses DIPlib exclusively (no OpenCV). Displays results via `dip.viewer.Show` / `dip.viewer.Spin` (requires a graphical display).

## Notes

- Both scripts run top-level code with no `if __name__ == "__main__"` guard — importing them executes the full pipeline.
- The SAM2 script's prompt points are generated but not passed to the predictor (lines 317-318: `point_coords=None, point_labels=None`). The segmentation relies solely on the bounding box prompt.
- `mask_pc.ply` is a generated output artifact, not source code.
