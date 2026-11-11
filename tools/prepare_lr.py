"""Create Pillow bicubic LR images from existing, processed HR images."""
import argparse
import io
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hr', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='LR root, containing x2/x4/x8')
    parser.add_argument('--scales', nargs='+', type=int, default=[2, 4, 8], choices=(2, 4, 8))
    args = parser.parse_args()
    if not args.hr.is_dir():
        parser.error('HR folder does not exist.')
    files = sorted(p for p in args.hr.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not files or len({p.stem for p in files}) != len(files):
        parser.error('Need nonempty HR images with unique filename stems.')
    for p in files:
        with Image.open(p) as image:
            rgb = image.convert('RGB')
            width, height = rgb.size
            for scale in args.scales:
                if width % scale or height % scale:
                    raise ValueError('HR dimensions must be divisible by scale: %s' % p)
                lr = rgb.resize((width // scale, height // scale), Image.Resampling.BICUBIC)
                buf = io.BytesIO()
                lr.save(buf, format='PNG')
                target = args.output / ('x%d' % scale) / (p.stem + '.png')
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = buf.getvalue()
                if target.exists():
                    if target.read_bytes() != raw:
                        raise FileExistsError('Existing LR differs; refusing to overwrite %s' % target)
                else:
                    with target.open('xb') as f:
                        f.write(raw)
    print('Prepared %d HR images at scales %s' % (len(files), args.scales))


if __name__ == '__main__':
    main()
