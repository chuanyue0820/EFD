# Release verification

The code-only package was checked on 2026-11-11 using the existing Linux
environment: Python 3.8.20 and PyTorch 2.2.2+cu121. No model binaries or dataset
images are included in the repository, and no GPU training was launched.

Passed checks:

- Four CPU unit tests covering ×2/×4/×8 output sizes and parameter counts,
  side-effect-free logging, backpropagation, strict checkpoint loading,
  THOP metadata handling, CLI seed overrides and paired-file validation.
- All six historical CelebA/Helen checkpoints loaded into the released model.
  On the same real LR input, the maximum absolute CPU output difference from
  the historical implementation was **0** for every checkpoint.
- The actual training CLI ran two optimizer steps, saved a checkpoint and
  training state, and resumed on CPU to step three.
- The paired test CLI and LR-only inference CLI processed two synthetic pairs;
  their reconstructed PNG pixels matched exactly.
- LR generation, dataset checking and the benchmark CLI completed.
- MATLAB R-channel evaluation was rerun on 200 real pairs. Maximum differences
  from the historical CSV were below 5e-14 for PSNR and 5e-16 for SSIM/FSIM.
- Python syntax, code-only file selection and private-path scans passed.

These are bounded runtime and equivalence checks. They do not constitute a
fresh environment installation or full 500,000-step convergence reproduction.
Use the processed dataset and pretrained files linked in the README for the
original experiments.
