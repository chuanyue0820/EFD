"""Validate paired folders before training or testing."""
import argparse
from pathlib import Path
from PIL import Image


def check_pair(hr, lr, scale):
    def images(folder):
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths = [p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError('Duplicate image stems: %s' % folder)
        return {p.stem: p for p in paths}
    gt, lq = images(hr), images(lr)
    if not gt or set(gt) != set(lq):
        raise ValueError('HR/LR filename stems differ or are empty: %s / %s' % (hr, lr))
    for key in gt:
        with Image.open(gt[key]) as a, Image.open(lq[key]) as b:
            if a.size != (b.width * scale, b.height * scale):
                raise ValueError('Incorrect scale pairing for %s' % key)
            if a.mode != 'RGB' or b.mode != 'RGB':
                raise ValueError('Expected RGB files for %s' % key)
    return len(gt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='datasets/CelebA or datasets/Helen')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--scales', nargs='+', type=int, choices=(2, 4, 8), default=[2, 4, 8])
    args = parser.parse_args()
    for split in args.splits:
        for scale in args.scales:
            count = check_pair(args.root / split / 'HR', args.root / split / 'LR' / ('x%d' % scale), scale)
            print('%s x%d: %d pairs OK' % (split, scale, count))


if __name__ == '__main__':
    main()
