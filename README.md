## Installation

The reference experiment environment is:

| Component | Version |
|:--|:--|
| Operating system | Linux |
| GPU | NVIDIA RTX 4090D, 24 GB |
| Python | 3.8.20 |
| PyTorch | 2.2.2, CUDA 12.1 build |
| TorchVision | 0.17.2 |
| NumPy | 1.24.3 |
| Pillow | 10.2.0 |
| pytorch-wavelets | 1.3.0 |
| timm / einops | 1.0.24 / 0.8.1 |

Run all commands below from the repository root. No pretrained models or dataset images are stored in this repository.

```bash
git clone https://github.com/chuanyue0820/EFD.git
cd EFD
conda create -n efd python=3.8 -y
conda activate efd
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The NVIDIA driver must support the CUDA build of PyTorch. A separate CUDA toolkit compilation is not needed. OpenCV 4.10.0 is pinned for an installable NumPy-1.x environment; the existing experiment runtime uses OpenCV 4.13.0. See [reproduction details](docs/reproduction.md) for the exact verification scope.

## Datasets

Download the author-provided, processed CelebA and Helen datasets from [Google Drive](https://drive.google.com/drive/folders/1kEi1WrSiOtaD2fb7fjfRnSnbMSuIhqNC?usp=drive_link). Extract them and arrange the folders as follows:

```text
datasets/
├── CelebA/
│   ├── train/
│   │   ├── HR/             # 128×128 RGB PNG images
│   │   └── LR/
│   │       ├── x2/         # 64×64
│   │       ├── x4/         # 32×32
│   │       └── x8/         # 16×16
│   └── test/
│       ├── HR/
│       └── LR/{x2,x4,x8}/
└── Helen/
    ├── train/
    │   ├── HR/
    │   └── LR/{x2,x4,x8}/
    └── test/
        ├── HR/
        └── LR/{x2,x4,x8}/
```

Each HR/LR pair must have the same filename, for example `HR/1.png` and `LR/x4/1.png`. The historical CelebA split contains 20,000 training images and 2,000 test images. Local numeric names are processed-image names, not a recovered complete mapping to raw CelebA JPEGs. Use the author's processed split for comparisons with the reported experiments; do not substitute a newly aligned or randomly split dataset.

If only the processed HR images are provided, create the LR images using Pillow bicubic:

```bash
python -m tools.prepare_lr --hr datasets/CelebA/train/HR --output datasets/CelebA/train/LR
python -m tools.prepare_lr --hr datasets/CelebA/test/HR  --output datasets/CelebA/test/LR
python -m tools.prepare_lr --hr datasets/Helen/train/HR --output datasets/Helen/train/LR
python -m tools.prepare_lr --hr datasets/Helen/test/HR  --output datasets/Helen/test/LR

python -m tools.check_dataset --root datasets/CelebA
python -m tools.check_dataset --root datasets/Helen
```

This tool downsamples existing HR images; it does not implement the unrecovered historical raw-image alignment pipeline.

## Pretrained Models

Download the six pretrained models from [Google Drive](https://drive.google.com/drive/folders/1tEDt-aw5ZJq2hmuIsvw5HlgEG_ssRIaN?usp=drive_link) and put them directly in `pretrained/`:

| Dataset | Scale | Filename |
|:--|:--:|:--|
| CelebA | ×2 | `EFD_CelebA_x2_iter455000.pth` |
| CelebA | ×4 | `EFD_CelebA_x4_iter320000.pth` |
| CelebA | ×8 | `EFD_CelebA_x8_iter255000.pth` |
| Helen | ×2 | `EFD_Helen_x2_iter120000.pth` |
| Helen | ×4 | `EFD_Helen_x4_iter15000.pth` |
| Helen | ×8 | `EFD_Helen_x8_iter5000.pth` |

SHA-256 checksums are in [pretrained/manifest.json](pretrained/manifest.json). They refer to the unchanged historical files. If the author republishes cleaned or renamed checkpoints, update this manifest accordingly.

```bash
python -m tools.verify_weights --directory pretrained
```

## Training

The six configurations use the full EFD network with internal LR-derived edge and semantic branches. No external segmentation weights or HR-derived priors are required. Training uses FP32, Adam, mean L1 loss and seed **10**. No gradient accumulation is enabled.

| Scale | HR crop | Batch/GPU | Initial learning rate | Iterations | LR milestones | Dataset enlargement |
|:--:|:--:|:--:|:--:|:--:|:--|:--:|
| ×2 | 96×96 | 32 | 2e-4 | 500,000 | 250k, 400k, 450k, 475k | 100 |
| ×4 | 128×128 | 32 | 2e-4 | 500,000 | 250k, 400k, 450k, 475k | 100 |
| ×8 | 128×128 | 16 | 2e-4 | 500,000 | 250k, 400k, 450k, 475k | 1 |

The Adam betas are `(0.9, 0.99)`, weight decay is zero, and the milestone multiplier is 0.5. Horizontal flipping, vertical flipping and transpose augmentation follow the original loader. Validation and checkpoint saving occur every 5,000 optimizer steps. The commands below use one GPU:

```bash
# CelebA
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_CelebA_x2.yml
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_CelebA_x4.yml
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_CelebA_x8.yml

# Helen
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_Helen_x2.yml
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_Helen_x4.yml
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_Helen_x8.yml
```

The ×2 configuration starts from random initialization. The ×4 and ×8 configurations explicitly initialize from the lower-scale checkpoint listed above; only shape-compatible parameters are transferred, and the new upsampling head keeps its initialization. Place the supplied checkpoints in `pretrained/`, or edit `path.pretrain_network_g` to point to your own lower-scale run. Stages are separate commands; completing ×2 does not automatically launch ×4.

To train ×4 directly from random initialization:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_CelebA_x4.yml \
  --force_yml name=EFD_CelebA_x4_scratch_seed10 path:pretrain_network_g=~
```

To use two GPUs while retaining a global batch of 32 for ×2/×4:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 train.py \
  -opt options/train/EFD_CelebA_x4.yml --launcher pytorch \
  --force_yml num_gpu=2 datasets:train:batch_size_per_gpu=16
```

For ×8, use a batch of 8 per GPU to retain its reference global batch of 16. The effective batch is `batch_size_per_gpu × world_size`; keeping 32 per GPU on multiple GPUs changes the reference setting. Results are saved under `experiments/<name>/`. To resume an interrupted run, use the same configuration and `--auto_resume`, or set `path.resume_state` to the corresponding `.state` file. Use a new experiment name for a separate run.

## Testing

### PSNR and SSIM in Python

After downloading the processed test set and pretrained model, run:

```bash
# CelebA
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/EFD_CelebA_x2.yml
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/EFD_CelebA_x4.yml
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/EFD_CelebA_x8.yml

# Helen
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/EFD_Helen_x2.yml
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/EFD_Helen_x4.yml
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/EFD_Helen_x8.yml
```

These configurations evaluate **all paired test images**, compute PSNR/SSIM on the Y channel after a scale-pixel border crop, and save logs and reconstructions under `results/<name>/`. CPU testing is available by adding `--force_yml num_gpu=0`.

### Reconstruct images for MATLAB

The following command accepts only LR images and preserves their stems: `1.png` is reconstructed as `1.png`.

```bash
python inference.py --scale 4 \
  --weights pretrained/EFD_CelebA_x4_iter320000.pth \
  --input datasets/CelebA/test/LR/x4 --output results/CelebA_x4_png --device cuda:0
```

Change the scale, model and input folder for the other five combinations. For CPU inference use `--device cpu`. The script refuses to overwrite existing output images.

MATLAB evaluation follows the historical R-channel protocol and is separate from Python Y-channel validation. See [matlab/README.md](matlab/README.md) for the exact crop, image-count and FSIM conventions. Do not directly compare Python Y-channel scores with MATLAB R-channel scores.

## Runtime and Complexity

```bash
python -m tools.benchmark --scale 4 --height 32 --width 32 --batch-size 1 \
  --warmup 10 --repeats 100 --device cuda:0
```

The JSON output records batch size, LR/HR resolution, dtype, warmups, repeats, device, timer and profiler. Latency uses synchronized CUDA events on GPU and `time.perf_counter` on CPU. Complexity is reported as **THOP MACs** and, separately, the `2 × MACs` FLOP convention. Unsupported custom wavelet/attention operations may not be counted by THOP. These measurements must use identical settings for every compared method.

## Verification

```bash
python -m unittest discover -s tests -v
```

The checks cover all three output scales, loss/backpropagation, checkpoint handling, CLI seed overrides and dataset pairing. See [docs/reproduction.md](docs/reproduction.md) and [docs/verification.md](docs/verification.md) for release checks and remaining data-distribution requirements.

## Acknowledgements and License

The implementation builds on [FreqFormer](https://github.com/JPWang-CS/FreqFormer) and [BasicSR](https://github.com/XPixelGroup/BasicSR). Their Apache-2.0 license and attribution are retained. The third-party MATLAB FSIM/SSIM reference implementations retain their own educational/research-use notices and are **not** relicensed under Apache-2.0; see `third_party/matlab_metrics/NOTICE.md`.
