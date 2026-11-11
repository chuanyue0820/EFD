# Reproduction details

## Model and training

The released EFD class preserves the historical forward computations and parameter names. The public class/file name changes from the internal FreqFormer name to EFD. The BasicSR SRModel training implementation and mean L1 objective are retained. The six portable configurations copy the saved crop, batch, augmentation, optimizer, scheduler and dataset-enlargement settings. They do not contain private machine paths or old resume-state filenames.

The published progressive configurations explicitly initialize ×4/×8 from the selected lower-scale files. The original CelebA ×8 saved configuration contains a resume state; that machine-specific resume is not reproduced as a new-run default. The lower-scale transfer uses the framework's existing non-strict loading rule for a different upsampling head. Same-scale pretrained evaluation uses strict loading after discarding only non-network THOP bookkeeping keys.

Network logging no longer runs THOP profiling, timing or random forward passes. This removes profiling buffers and random-number consumption from `repr(model)`. Consequently a new training trajectory is not claimed to be bitwise identical to a historical run that executed those logging side effects. Explicit benchmarking operates on a disposable model copy.

## Data and metrics

Reproducing the original table requires the author's processed datasets and supplied checkpoint files, not just a raw CelebA download. The historical 128×128 HR alignment script and complete original-JPEG mapping have not been recovered. This repository does not substitute the separate supplementary alignment procedure for the original preprocessing. The author-provided cloud bundles are linked in the README; use those processed pairs and model files for comparison.

Python validation uses uint8 RGB reconstructions converted to Y, then crops `scale` pixels before PSNR/SSIM. The historical MATLAB script selects the R channel, modcrops to `scale`, shaves `scale` pixels, computes FSIM on that region, and crops another **4 pixels** for PSNR/SSIM. It is not equivalent to Python Y-channel evaluation. Image count must be stated explicitly: the historical local MATLAB scripts evaluated the first 200 numbered images, while Python validation and full-folder reconstruction can cover the complete folder.

## Runtime environment

The existing verification environment uses Python 3.8.20, torch 2.2.2+cu121, torchvision 0.17.2+cu121, NumPy 1.24.3, OpenCV 4.13.0, Pillow 10.2.0, pytorch-wavelets 1.3.0, timm 1.0.24 and einops 0.8.1. The install manifest pins OpenCV 4.10.0 for NumPy-1.x package compatibility. A clean dependency installation and full 500k-step retraining are separate from the bounded runtime verification; neither is implied by a successful smoke test.

Multi-GPU training can preserve the global batch by dividing the per-GPU batch by world size, but the sample ordering and floating-point reductions can still differ. Seed 10 fixes initialization and loader RNGs for a given execution configuration; exact cross-hardware equality is not promised.
