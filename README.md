# Record3D to Open3D 3D Reconstruction Pipeline

This project reconstructs a 3D scene from **Record3D** captures using synchronized depth and RGB data.
It performs point cloud generation, surface normal estimation, and volumetric fusion using **VDBFusion** and **Open3D**.

---

## Overview

The script processes frame sequences exported from the **Record3D** app.
It reads per-frame camera intrinsics and poses, reconstructs per-pixel 3D points in world coordinates,
and fuses them into a single 3D model.

It provides:

* Intrinsic scaling for resized frames
* Camera pose handling and coordinate correction
* Point cloud generation in camera and world coordinates
* Normal estimation using Scharr filters
* Volumetric fusion and mesh extraction using **VDBFusion**
* Optional **space carving** for improved geometric consistency

---

## Requirements

This project was tested with the following setup:

| Library   | Version |
| --------- | ------- |
| Python    | 3.10+   |
| Open3D    | 0.19.0  |
| OpenCV    | 4.12.0  |
| NumPy     | 1.22.4  |
| SciPy     | 1.12.0  |
| tqdm      | 4.66.4  |
| vdbfusion | 0.1.6   |

To install dependencies, run:

pip install open3d opencv-python numpy scipy tqdm vdbfusion

---

## Dataset Structure

This project expects data exported from the **Record3D** iOS app.
The directory should contain three main components: RGB frames, depth maps, and metadata.

Example directory layout:
```bash
Your_Record3D_Capture/
├── rgb/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── depth/
│   ├── 0.exr
│   ├── 1.exr
│   ├── 2.exr
│   └── ...
└── metadata.json
```

### Metadata contents

The file `metadata.json` should include:

* `poses`: list of camera poses as `[qx, qy, qz, qw, tx, ty, tz]` for each frame
* `perFrameIntrinsicCoeffs`: per-frame intrinsics `[fx, fy, cx, cy]`

These are generated automatically when exporting from **Record3D**.

Record3D official website: [https://record3d.app](https://record3d.app)

---

## Usage

### 1. Basic command

Provide your own dataset path using the `--data_path` argument:

python record3d_to_open3d.py --data_path "/path/to/your/Record3D_capture/"

### 3. Enable space carving

Space carving can remove uncertain or empty regions during volumetric fusion:

python record3d_to_open3d.py --data_path "/path/to/your/Record3D_capture/" --space_carving

---

## Command Line Arguments

| Argument          | Description                     | Default                                                                        |
| ----------------- | ------------------------------- | ------------------------------------------------------------------------------ |
| `--data_path`     | Path to Record3D dataset folder | `/home/hannes/Downloads/Robot_Record3D-20251004T153553Z-1-001/Robot_Record3D/` |
| `--width`         | Target processing width         | `192`                                                                          |
| `--height`        | Target processing height        | `256`                                                                          |
| `--max_depth`     | Maximum valid depth in meters   | `1.5`                                                                          |
| `--min_depth`     | Minimum valid depth in meters   | `0.25`                                                                         |
| `--voxel_size`    | Voxel resolution for VDB fusion | `0.005`                                                                        |
| `--sdf_trunc`     | TSDF truncation distance        | `0.8`                                                                          |
| `--space_carving` | Enable space carving            | `False`                                                                        |

---

## Output

* Live visualization of color frames and normals
* Incremental point cloud integration
* Final fused **Open3D mesh** visualized interactively

Press **q** during processing to preview partial reconstructions.

---

## References

* Record3D: [https://record3d.app](https://record3d.app)
* Open3D: [https://www.open3d.org](https://www.open3d.org)
* VDBFusion: [https://github.com/PRBonn/vdbfusion](https://github.com/PRBonn/vdbfusion)

---

## Notes

* The Z-axis is flipped to align camera and world conventions.
* Intrinsics are scaled per frame using `perFrameIntrinsicCoeffs`.
* Normals are computed using Scharr filters for local gradient stability.

---

## Author

Hannes Reichert
3D Reconstruction / Robotics Research

## Licence

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit.
- NonCommercial — You may not use the material for commercial purposes.
- No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

Full license text: https://creativecommons.org/licenses/by-nc/4.0/legalcode


