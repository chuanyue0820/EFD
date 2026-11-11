"""LR-only EFD inference. File stems are preserved for MATLAB evaluation."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from basicsr.models import build_model
from basicsr.utils import tensor2img


def load_model(weights, scale, device):
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Pass --device cpu.')
        torch.cuda.set_device(device)
    config = Path(__file__).resolve().parent / 'options/test' / ('EFD_CelebA_x%d.yml' % scale)
    with config.open(encoding='utf-8') as f:
        opt = yaml.safe_load(f)
    opt.update(is_train=False, dist=False, rank=0, world_size=1,
               num_gpu=0 if device.type == 'cpu' else 1)
    opt['path']['pretrain_network_g'] = str(weights)
    model = build_model(opt)
    model.net_g.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--scale', type=int, choices=(2, 4, 8), required=True)
    parser.add_argument('--input', type=Path, required=True, help='One image or an LR folder')
    parser.add_argument('--output', type=Path, required=True, help='Output folder')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error('Weight file not found; see pretrained/README.md.')
    torch.set_num_threads(args.threads)
    if args.input.is_file():
        files = [args.input]
    elif args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'})
    else:
        parser.error('Input image/folder does not exist.')
    if not files:
        parser.error('No input images found.')
    if len({p.stem for p in files}) != len(files):
        parser.error('Input stems are not unique; output filenames would collide.')
    targets = [args.output / (p.stem + '.png') for p in files]
    if any(p.exists() for p in targets):
        parser.error('Output images already exist; choose a new output folder.')
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, args.scale, args.device)
    with torch.inference_mode():
        for i, (path, target) in enumerate(zip(files, targets), 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('Unreadable input: %s' % path)
            rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
            tensor = torch.from_numpy(rgb).float().div_(255).unsqueeze(0)
            model.feed_data({'lq': tensor})
            model.test()
            output = tensor2img(model.output)
            expected = (image.shape[0] * args.scale, image.shape[1] * args.scale, 3)
            if output.shape != expected:
                raise RuntimeError('Unexpected output shape: %s, expected %s' % (output.shape, expected))
            if not cv2.imwrite(str(target), output):
                raise OSError('Could not write %s' % target)
            if i % 100 == 0 or i == len(files):
                print('Saved %d/%d images' % (i, len(files)), flush=True)


if __name__ == '__main__':
    main()
