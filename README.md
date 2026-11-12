
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

### Identity-Disjoint CelebA Test Set

We additionally provide a CelebA test set for evaluating face super-resolution on unseen identities. It contains **2,000 images from 1,340 identities** and was constructed with **random seed 10**. The test images follow the same preprocessing and bicubic degradation protocol as the original CelebA test set. This supplementary set is provided separately from the original benchmark split.

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1R64aUW52bP7pDccPUrn3dJDhMiCdqYIM?usp=drive_link). The exact image filenames and corresponding identity IDs are listed in `data_splits/identity_disjoint_test.txt`.

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

## Training

Run the following scripts:

```bash
# CelebA
python train.py -opt options/train/EFD_CelebA_x2.yml
python train.py -opt options/train/EFD_CelebA_x4.yml
python train.py -opt options/train/EFD_CelebA_x8.yml

# Helen
python train.py -opt options/train/EFD_Helen_x2.yml
python train.py -opt options/train/EFD_Helen_x4.yml
python train.py -opt options/train/EFD_Helen_x8.yml
```

## Testing

After downloading the processed test set and pretrained model, run:

```bash
# CelebA
python test.py -opt options/test/EFD_CelebA_x2.yml
python test.py -opt options/test/EFD_CelebA_x4.yml
python test.py -opt options/test/EFD_CelebA_x8.yml

# Helen
python test.py -opt options/test/EFD_Helen_x2.yml
python test.py -opt options/test/EFD_Helen_x4.yml
python test.py -opt options/test/EFD_Helen_x8.yml
```

## Acknowledgements

This codebase is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
