<p align="center">
  <img src="media/logo.png" alt="lensboy" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/lensboy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lensboy"></a>
  <a href="https://pypi.org/project/lensboy/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/lensboy"></a>
  <a href="https://github.com/Robertleoj/lensboy/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Robertleoj/lensboy"></a>
</p>

Camera calibration for vision engineers. Maximally powerful, minimally complex.

One job: fit camera models and verify the results. OpenCV models when they work, spline-based distortion when they don't.

> Many of the techniques in this library were originally developed in [mrcal](https://mrcal.secretsauce.net/).

## Why lensboy

Even for standard OpenCV models, lensboy gives you better calibrations than raw `cv2.calibrateCamera` (see [model comparison notebook](examples/model_comparison.ipynb)):

- **Automatic outlier filtering** removes bad detections
- **Target warp estimation** compensates for non-flat calibration boards

For cheap or wide-angle lenses where OpenCV's distortion model isn't enough, lensboy offers spline-based models that can capture arbitrary distortion patterns.

Lensboy also offers **analysis tools** to verify your calibration is actually good.

## Quick example

```python
import lensboy as lb

target_points, frames, image_indices = lb.extract_frames_from_charuco(board, imgs)

result = lb.calibrate_camera(
    target_points, frames,
    camera_model_config=lb.OpenCVConfig(
        image_height=h, image_width=w,
    ),
)

result.camera_model.save("camera.json")
```

Swap the config for a spline model — same API, more flexible:

```python
result = lb.calibrate_camera(
    target_points, frames,
    camera_model_config=lb.PinholeSplinedConfig(
        image_height=h, image_width=w,
    ),
)
```

## Getting started

Read the **[calibration guide](https://robertleoj.github.io/lensboy/calibration_guide.html)** for a full walkthrough - calibrating a camera, verifying the results, and exporting for runtime use.

If you just want to see `lensboy` in action, see [quickstart notebook](examples/quickstart.ipynb).

## Analysis tools

Plots for residuals, distortion, detection coverage, and [model differencing](examples/model_differencing.ipynb). See the [example notebooks](examples/).

<p align="center">
  <img src="media/showcase_3.png" width="700"><br>
  <img src="media/showcase_4.png" width="345"> <img src="media/showcase_1.png" width="345"><br>
  <img src="media/showcase_2.png" width="345"> <img src="media/showcase_5.png" width="345">
</p>

## Install

Full install, with analysis and plotting:

```bash
pip install lensboy[analysis]
```

Minimal install, for loading and using models:

```bash
pip install lensboy
```

## Spline models

Spline models use B-spline grids instead of polynomial coefficients, so they can fit lenses that OpenCV's model can't. This approach is inspired by [mrcal](https://mrcal.secretsauce.net/).

The calibrated model converts to a pinhole model with undistortion maps, so you can use it with any standard pinhole pipeline.
