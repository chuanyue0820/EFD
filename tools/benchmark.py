"""Report inference latency and THOP MACs with an explicit measurement protocol."""
import argparse
import copy
import json
import time

import torch
from thop import profile

from basicsr.archs.efd_arch import EFD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale', type=int, choices=(2, 4, 8), default=4)
    p.add_argument('--height', type=int, default=32)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if min(a.height, a.width, a.batch_size, a.repeats) <= 0 or a.warmup < 0:
        p.error('Dimensions/batch/repeats must be positive; warmup must be nonnegative.')
    torch.set_num_threads(2)
    device = torch.device(a.device)
    net = EFD(upscale=a.scale, in_chans=3, semantic_ch=19, embed_dim=60,
              num_heads=6, depth=18, expansion_factor=2, split_size=[8, 32], img_size=64).to(device).eval()
    x = torch.rand(a.batch_size, 3, a.height, a.width, device=device)
    with torch.inference_mode():
        for _ in range(a.warmup):
            net(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            with torch.cuda.device(device):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(a.repeats):
                    net(x)
                end.record()
                end.synchronize()
            latency = start.elapsed_time(end) / a.repeats
            timer = 'CUDA events, synchronized'
        else:
            start = time.perf_counter()
            for _ in range(a.repeats):
                net(x)
            latency = 1000 * (time.perf_counter() - start) / a.repeats
            timer = 'time.perf_counter'
        # THOP modifies bookkeeping buffers; profile a disposable copy only.
        macs, _ = profile(copy.deepcopy(net), inputs=(x,), verbose=False)
    print(json.dumps(dict(scale=a.scale, input_shape=list(x.shape), output_shape=[a.batch_size, 3, a.height*a.scale, a.width*a.scale],
        warmup=a.warmup, repeats=a.repeats, dtype='float32', device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch_version=torch.__version__, timer=timer, latency_ms_per_batch=latency,
        latency_ms_per_image=latency/a.batch_size, parameters=sum(v.numel() for v in net.parameters()),
        profiling_tool='THOP', thop_macs=macs, flops_using_2_per_mac=2*macs,
        counting_note='THOP counts supported operations; custom wavelet/attention operations may be omitted.'), indent=2))


if __name__ == '__main__':
    main()
