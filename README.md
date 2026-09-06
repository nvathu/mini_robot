# Robot Camera Projection Test

Script to test the 3D→2D projection (`project_to_pixel`) used in the RGB + MoCap room point-cloud pipeline.

## Context

During review, `project_to_pixel()` was found to use the wrong sign compared to the standard pinhole formula (`u = fx*Xc/Zc + cx`), and was inconsistent with the inverse direction (`unproject`) used throughout the point-cloud pipeline. It has been fixed to match the standard formula. `test.py` is a visual sanity check on real images: it projects the known MoCap position of robot S1 onto the image and checks whether the green dot lands on the real robot.

## How to run

```bash
pip install -r requirements.txt
python test.py
```

Output is saved in `test_projection_output/`, one image per sample with a green dot at the predicted position of robot S1.

## Interpreting the results

Open a few images in `test_projection_output/` and compare the green dot to the robot's actual position:

- Dot lands on the robot → the fixed formula is correct.
- Dot is offset (especially left/right) → the formula is still wrong and needs another look.

Prioritize images with large `lateral` values (robot clearly off to one side) — a left/right sign bug shows up most clearly there; when `lateral` is near 0 (robot near image center) it's hard to tell correct from incorrect.

## Folder structure

```
.
├── test.py
├── requirements.txt
├── dataset/
│   ├── calib/intrinsics.json
│   ├── poses/records_20250813-1535-S3.csv
│   └── rgb/sample_session/     # a few dozen sample RGB images
└── README.md
```
## Part 2: full pipeline test (`test.py`)

Runs the complete flow on a handful of frames: RGB → depth prediction (ResNet) → metric
scale → mask dynamic robots (using the fixed `project_to_pixel`) → unproject → transform
to world coordinates using MoCap self-pose.

Requires `model.py`, `weights/best_model.pth`, and `dataset/calib/global_scale.json` in
addition to the files used by `test.py`.

```bash
python test.py
```

Output: `test_pipeline_output/pipeline_test_pointcloud.npz` (raw points) and
`pipeline_test_topdown.png` (quick top-down scatter for visual inspection).
