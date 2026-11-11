"""Bounded CPU checks; no downloaded weights or datasets are required."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from basicsr.archs.efd_arch import EFD
from basicsr.models import build_model
from basicsr.utils.options import parse_options
from tools.check_dataset import check_pair

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self, scale=4):
        with (ROOT / ('options/train/EFD_CelebA_x%d.yml' % scale)).open() as f:
            opt = yaml.safe_load(f)
        opt.update(num_gpu=0, dist=False, rank=0, world_size=1, is_train=True)
        opt['path']['pretrain_network_g'] = None
        return opt

    def test_shapes_parameter_counts_and_logging(self):
        counts = {2: 1308885, 4: 1328361, 8: 1406265}
        for scale, count in counts.items():
            opt = self.options(scale)['network_g']
            opt.pop('type')
            net = EFD(**opt).eval()
            keys = set(net.state_dict())
            rng = torch.get_rng_state().clone()
            self.assertIn('EFD', repr(net))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(keys, set(net.state_dict()))
            self.assertEqual(sum(p.numel() for p in net.parameters()), count)
            with torch.inference_mode():
                y = net(torch.rand(1, 3, 16, 16))
            self.assertEqual(tuple(y.shape), (1, 3, 16 * scale, 16 * scale))
            self.assertTrue(torch.isfinite(y).all())

    def test_training_checkpoint_and_metadata(self):
        model = build_model(self.options())
        old = model.net_g.conv_first.weight.detach().clone()
        model.feed_data({'lq': torch.rand(1, 3, 16, 16), 'gt': torch.rand(1, 3, 64, 64)})
        model.optimize_parameters(1)
        self.assertTrue(np.isfinite(model.get_current_log()['l_pix']))
        self.assertFalse(torch.equal(old, model.net_g.conv_first.weight))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pth'
            state = copy.deepcopy(model.net_g.state_dict())
            state['conv_first.total_ops'] = torch.tensor([0.0])
            torch.save({'params': state}, path)
            model.load_network(model.net_g, path, strict=True)
            state['not_a_model_parameter'] = torch.tensor([1.0])
            torch.save({'params': state}, path)
            with self.assertRaises(RuntimeError):
                model.load_network(model.net_g, path, strict=True)

    def test_cli_seed_override_precedes_rng_seed(self):
        argv = ['train.py', '-opt', str(ROOT / 'options/train/EFD_CelebA_x2.yml'),
                '--force_yml', 'manual_seed=42', 'num_gpu=0']
        with patch('sys.argv', argv):
            opt, _ = parse_options(str(ROOT), is_train=True)
        self.assertEqual(opt['manual_seed'], 42)
        self.assertAlmostEqual(random.random(), random.Random(42).random())

    def test_pair_validation_detects_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hr, lr = root / 'HR', root / 'LR'
            hr.mkdir(); lr.mkdir()
            Image.new('RGB', (128, 128)).save(hr / '1.png')
            Image.new('RGB', (32, 32)).save(lr / '1.png')
            self.assertEqual(check_pair(hr, lr, 4), 1)
            Image.new('RGB', (128, 128)).save(hr / '2.png')
            with self.assertRaises(ValueError):
                check_pair(hr, lr, 4)


if __name__ == '__main__':
    unittest.main()
