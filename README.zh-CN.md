## Installation

完整命令见 [英文 README](README.md)。实验参考环境：Linux、RTX 4090D 24GB、Python 3.8.20、PyTorch 2.2.2（CUDA 12.1）、TorchVision 0.17.2、NumPy 1.24.3、Pillow 10.2.0、pytorch-wavelets 1.3.0。

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

处理好的数据集：[Google Drive](https://drive.google.com/drive/folders/1kEi1WrSiOtaD2fb7fjfRnSnbMSuIhqNC?usp=drive_link)。下载处理后的 CelebA 和 Helen，分别放到 `datasets/CelebA`、`datasets/Helen`，各自包含 `train/HR`、`test/HR` 和同名配对的 `train/LR/x2`、`x4`、`x8` 及测试 LR 目录。HR 为 128×128 RGB。可运行 `python -m tools.check_dataset --root datasets/CelebA` 检查配对。

原始 HR 的完整生成代码尚未找回，因此请使用作者提供的处理后数据。不要把重新裁剪、重新对齐或重新划分的图像当作完全相同的原实验数据。

## Pretrained Models

六份预训练模型：[Google Drive](https://drive.google.com/drive/folders/1tEDt-aw5ZJq2hmuIsvw5HlgEG_ssRIaN?usp=drive_link)。文件名和 SHA-256 见 `pretrained/manifest.json`，放在 `pretrained/` 下即可。仓库不包含模型和数据图片。

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_CelebA_x2.yml
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_CelebA_x4.yml
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/EFD_CelebA_x8.yml
```

Helen 将配置名中的 `CelebA` 换成 `Helen`。×2 随机初始化，×4 和 ×8 默认分别使用给定的低倍率模型初始化。若使用自己训练出的权重，修改配置中的 `path.pretrain_network_g`。三个阶段分别执行，不会自动接跑。

统一使用 seed=10、FP32、Adam、L1，初始学习率 2e-4，每 5,000 步验证与存权重。×2/×4 单卡 batch=32，×8 单卡 batch=16。完整参数表及多卡、从头训练、恢复命令见英文 README。

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/EFD_CelebA_x4.yml

python inference.py --scale 4 --weights pretrained/EFD_CelebA_x4_iter320000.pth \
  --input datasets/CelebA/test/LR/x4 --output results/CelebA_x4_png --device cuda:0
```

测试配置共六份，覆盖 CelebA/Helen ×2/×4/×8。`test.py` 计算完整测试集的 Y 通道 PSNR/SSIM；`inference.py` 只接收 LR、保持文件编号不变，适合后续 MATLAB 评测。CPU 测试在 `test.py` 后加 `--force_yml num_gpu=0`，CPU 推理用 `--device cpu`。

MATLAB 的原 R 通道 PSNR/SSIM/FSIM 流程见 `matlab/README.md`，与 Python 的 Y 通道验证指标分开使用。已有实验中使用过前 200 张的评测，完整测试时需明确设置数量，不能混用。
