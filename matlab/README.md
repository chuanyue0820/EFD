# Historical MATLAB metrics

Generate numbered SR PNGs using `inference.py`. Open MATLAB in the repository root:

```matlab
addpath('matlab');
T = evaluate_matlab('results/CelebA_x4_png', ...
    'datasets/CelebA/test/HR', 4, 200, 'results/CelebA_x4_matlab_200.csv');
```

Use `2000` instead of `200` to evaluate a complete 2,000-image folder. State the count with the result. Use the correct scale and paired folder for every experiment. Images are read as `1.png` through `N.png`. Existing output CSVs are not overwritten.

The metric convention matches the historical IQA script:

1. Select the R channel and crop dimensions to a multiple of scale.
2. Shave `scale` pixels from every border.
3. Calculate FSIM on that region.
4. Shave **another 4 pixels** for PSNR and SSIM, as in `compute_psnr(...,4)`.

MATLAB's `fspecial`, `filter2` and the reference metric functions must be available; the Image Processing Toolbox is required. This is a different convention from the Python Y-channel PSNR/SSIM test configs.

The included `FeatureSIM.m` and `ssim_index.m` are unchanged third-party reference implementations with educational/research-use-only permissions. Their original notices are retained. They are not covered by the main code's Apache license; see `third_party/matlab_metrics/NOTICE.md`.
