import sys
import torchvision.transforms.functional as tvf
# 伪造模块路径以重定向导入
sys.modules['torchvision.transforms.functional_tensor'] = tvf
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from basicsr.utils.registry import ARCH_REGISTRY
import math
import time
from thop import profile, clever_format


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LightweightEdgeExtractor(nn.Module):
    """Lightweight LR edge prior extractor.

    Uses fixed Sobel filters and does not require additional edge labels.
    Input:  x: (B, C, H, W), usually RGB image in [0, 1]
    Output: edge: (B, 1, H, W), normalized edge magnitude in [0, 1]
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1., -2., -1.],
             [0., 0., 0.],
             [1., 2., 1.]]
        ).view(1, 1, 3, 3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x[:, 0:1]

        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return edge


class LightweightSemanticExtractor(nn.Module):
    """Lightweight semantic-prior estimation branch from LR input.

    Without external face parsing supervision, the output should be interpreted
    as learnable component-aware semantic guidance rather than guaranteed
    ground-truth facial parsing maps.
    Input:  x: (B, C, H, W)
    Output: sem: (B, semantic_ch, H, W), soft component maps
    """
    def __init__(self, in_chans=3, semantic_ch=19, base_ch=16):
        super().__init__()
        self.semantic_ch = int(semantic_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, base_ch, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch, 3, 1, 1, groups=base_ch),
            nn.Conv2d(base_ch, base_ch * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, 1, 1, groups=base_ch * 2),
            nn.Conv2d(base_ch * 2, semantic_ch, 1, 1, 0)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=1)


class GuidanceEncoder(nn.Module):
    """Encode optional edge / semantic priors to feature-space guidance.

    - edge: (B, 1, H, W) in [0,1] or arbitrary scale
    - sem:  (B, S, H, W) semantic logits/probs or one-hot maps

    Returns:
        g_low, g_high: (B, C, H, W) guidance tensors (can be None if inputs are None)
    """
    def __init__(self, dim: int, semantic_ch: int = 0, edge_ch: int = 1):
        super().__init__()
        self.dim = dim
        self.semantic_ch = int(semantic_ch)
        self.edge_ch = int(edge_ch)

        self.edge_proj = None
        self.sem_proj = None
        if self.edge_ch > 0:
            self.edge_proj = nn.Sequential(
                nn.Conv2d(self.edge_ch, dim, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            )
        if self.semantic_ch > 0:
            self.sem_proj = nn.Sequential(
                nn.Conv2d(self.semantic_ch, dim, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            )

        # learnable mixing for low/high guidance
        self.mix = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 1, 1, 0),
        )

    def forward(self, edge=None, sem=None):
        if edge is None and sem is None:
            return None, None

        feats = []
        if self.edge_proj is not None and edge is not None:
            feats.append(self.edge_proj(edge))
        if self.sem_proj is not None and sem is not None:
            # Allow runtime mismatch by slicing or zero-padding semantic channels.
            if sem.shape[1] != self.semantic_ch:
                if sem.shape[1] > self.semantic_ch:
                    sem = sem[:, :self.semantic_ch]
                else:
                    pad_ch = self.semantic_ch - sem.shape[1]
                    sem = F.pad(sem, (0, 0, 0, 0, 0, pad_ch), mode='constant', value=0)
            feats.append(self.sem_proj(sem))

        if not feats:
            return None, None

        g = feats[0] if len(feats) == 1 else sum(feats)
        # split into low/high guidance
        gh = torch.cat([g, torch.tanh(g)], dim=1)  # (B, 2C, H, W)
        gh = self.mix(gh)
        g_low, g_high = gh.chunk(2, dim=1)
        return g_low, g_high


class ComponentChannelInteraction(nn.Module):
    """Lightweight channel interaction guided by facial component semantics.

    sem is expected to be a soft map over components (B, S, H, W). We compute
    component-wise pooled descriptors and produce a channel gate.
    """
    def __init__(self, dim: int, semantic_ch: int):
        super().__init__()
        self.semantic_ch = int(semantic_ch)
        self.dim = dim
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat_map, sem):
        # feat_map: (B, C, H, W); sem: (B, S, H, W)
        if sem is None:
            return None
        B, C, H, W = feat_map.shape
        S = sem.shape[1]
        if S != self.semantic_ch:
            # allow runtime mismatch by slicing / padding
            S_use = min(S, self.semantic_ch)
            sem = sem[:, :S_use]
            S = S_use

        sem = F.interpolate(sem, size=(H, W), mode='bilinear', align_corners=False)
        sem = sem.clamp_min(0)
        denom = sem.flatten(2).sum(-1, keepdim=True) + 1e-6  # (B, S, 1)

        # component descriptors: (B, S, C)
        feat_flat = feat_map.view(B, C, H * W)
        sem_flat = sem.view(B, S, H * W)
        comp = torch.einsum('bsh,bch->bsc', sem_flat, feat_flat) / denom  # weighted avg

        # cross-component interaction (self-attn in component space)
        q = self.proj_q(comp)
        k = self.proj_k(comp)
        v = self.proj_v(comp)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)
        comp_out = torch.matmul(attn, v)  # (B, S, C)
        comp_out = self.out(comp_out)

        # aggregate components -> channel gate
        gate = comp_out.mean(dim=1).view(B, C, 1, 1)
        return gate


class MultiScaleFrequencyGate(nn.Module):
    """Multi-scale frequency gate used inside DFFN.

    It builds frequency-aware gating from multiple pooled scales to better
    capture low/mid/high frequency responses.
    """
    def __init__(self, dim: int, scales=(1, 2, 4), enable_ms_freq=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # 根据消融实验开关决定是否使用多尺度
        self.scales = tuple(scales) if enable_ms_freq else (1,)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.GELU()
            )
            for _ in self.scales
        ])
        self.fuse = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x, H, W, sem=None):
        # x: (B, L, C) where C == dim*2 in the caller split usage
        B, N, C = x.shape
        x = self.norm(x).transpose(1, 2).contiguous().view(B, C, H, W)
        outs = []
        for s, conv in zip(self.scales, self.convs):
            if s == 1:
                xs = x
            else:
                xs = F.avg_pool2d(x, kernel_size=s, stride=s, ceil_mode=False)
            ys = conv(xs)
            if ys.shape[-2:] != (H, W):
                ys = F.interpolate(ys, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(ys)
        y = self.fuse(sum(outs))
        return y.flatten(2).transpose(1, 2).contiguous()


class HFRB(nn.Module):
    """ Dual-Tree Complex Wavelet High-Frequency Residual Block.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Linear(dim, 2 * dim // 3)
        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')

    def forward(self, x, x_size, sem=None):
        """
        Input: x: (B, N, C), H, W
        Output: x: (B, C, H, W)
        """
        B, N, C = x.shape
        H, W = x_size
        x = self.down(x)
        x = x.view(B, H, W,  2 * C // 3)
        x = torch.permute(x, (0, 3, 1, 2))
        _, xh = self.xfm(x)
        xh = xh[0][..., 0].view(B, C, 4, xh[0].shape[-3], xh[0].shape[-2])
        xh = xh.view(B, C, -1)[... , :N]
        xh = xh.view(B, C, H, W)
        return xh

class Channel_mixing(nn.Module):
    """ Channel Mixing.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.ll = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )

        self.lh_1_0 = nn.Linear(self.block_size, self.block_size)
        self.lh_1_1 = nn.Linear(self.block_size, self.block_size)

        self.lh_2_0 = nn.Linear(self.block_size, self.block_size)
        self.lh_2_1 = nn.Linear(self.block_size, self.block_size)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink = 0.0

    def forward(self, x, H, W, g_low=None, g_high=None):
        """
        Input: x: (B, N, C), H, W
        Output: x: (B, N, C)
        """
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.permute(x, (0, 3, 1, 2))

        xl, xh = self.xfm(x) # B C H W,  B C 6 H W 2
        xl = self.ll(xl)

        # optional guidance for low/high frequency fusion
        if g_low is not None:
            if g_low.shape[-2:] != xl.shape[-2:]:
                g_low_ = F.interpolate(g_low, size=xl.shape[-2:], mode='bilinear', align_corners=False)
            else:
                g_low_ = g_low
            xl = xl * (1.0 + torch.sigmoid(g_low_))
        if g_high is not None:
            # match high-frequency spatial resolution
            hH, hW = xh[0].shape[-3], xh[0].shape[-2]
            if g_high.shape[-2:] != (hH, hW):
                g_high_ = F.interpolate(g_high, size=(hH, hW), mode='bilinear', align_corners=False)
            else:
                g_high_ = g_high
            # broadcast to (B, C, 6, hH, hW, 2)
            xh[0] = xh[0] * (1.0 + torch.sigmoid(g_high_).unsqueeze(2).unsqueeze(-1))

        xh[0] = torch.permute(xh[0], (5, 0, 2, 3, 4, 1)) #  2 B 6 H W C
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4],
                              self.num_blocks, self.block_size) # 2 B 6 H W num_b b_size

        x_real = xh[0][0] # B 6 H W num_b b_size
        x_imag = xh[0][1] # B 6 H W num_b b_size

        x_real_1 = F.relu(self.lh_1_0(x_real) - self.lh_1_1(x_imag))
        x_imag_1 = F.relu(self.lh_1_1(x_real) + self.lh_1_0(x_imag))

        x_real_2 = self.lh_2_0(x_real_1) - self.lh_2_1(x_imag_1)
        x_imag_2 = self.lh_2_1(x_real_1) - self.lh_2_0(x_imag_1)

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1) # B 6 H W num_b b_size 2
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6]) # B 6 H W C 2
        xh[0] = torch.permute(xh[0], (0, 4, 1, 2, 3, 5)) # # B C 6 H W 2

        x = self.ifm((xl, xh))
        x = torch.permute(x[..., :H, :W], (0, 2, 3, 1)).contiguous()
        x = x.reshape(B, N, C)  # permute is not same as reshape or view

        return x


class FrequencyProjection(nn.Module):
    """ Frequency Projection.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv_1 = nn.Conv2d(dim, dim // 2, 1, 1, 0)
        self.act = nn.GELU()
        self.res_2 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.GELU()
        )
        self.conv_out = nn.Conv2d(dim // 2, dim, 1, 1, 0)

    def forward(self, x, edge=None, sem=None):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        res = x
        x = self.conv_1(x)
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat((self.act(x1), self.res_2(x2)), dim=1)
        out = self.conv_out(out)
        return out + res


class ChannelProjection(nn.Module):
    """ Channel Projection.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 6, 1, 1, 0)
        self.CI1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1)
        )
        self.CI2 = nn.Sequential(
            nn.Conv2d(dim // 6, dim // 6, kernel_size=3, stride=1, padding=1, groups=dim // 6),
            nn.Conv2d(dim // 6, dim // 6, 7, stride=1, padding=9, groups=dim // 6, dilation=3),
            nn.Conv2d(dim // 6, dim // 6, kernel_size=1)
        )
        self.pro_out = nn.Conv2d(dim // 6, dim, kernel_size=1)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        x = self.pro_in(x)
        res = x
        ci1 = self.CI1(x)
        ci2 = self.CI2(x)
        out = self.pro_out(res * ci1 * ci2)
        return out


class SpatialProjection(nn.Module):
    """ Spatial Projection.
    Args:
        dim (int): input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.pro_in = nn.Conv2d(dim, dim // 6, 1, 1, 0)
        self.dwconv = nn.Conv2d(dim // 6,  dim // 6, kernel_size=3, stride=1, padding=1, groups= dim // 6)
        self.pro_out = nn.Conv2d(dim // 12, dim, kernel_size=1)

    def forward(self, x):
        """
        Input: x: (B, C, H, W)
        Output: x: (B, C, H, W)
        """
        x = self.pro_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.pro_out(x)
        return x


class FrequencyGate(nn.Module):
    """ Frequency-Gate.
    Args:
        dim (int): Input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
        )

    def forward(self, x, H, W, sem=None):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        B, N, C = x.shape
        x1, x2 = x.chunk(2, dim = -1)
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2


class DFFN(nn.Module):
    """ Dual frequency aggregation Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., enable_dffn_ms_freq=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # 传递多尺度频域聚合开关
        self.fg = MultiScaleFrequencyGate(hidden_features//2, enable_ms_freq=enable_dffn_ms_freq)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W, sem=None):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x1, x2 = x.chunk(2, dim=-1)
        gate = self.fg(x2, H, W)
        x = x1 * gate
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Spatial_Attention(nn.Module):
    """ Spatial Self-Attention.
    It supports rectangle window (containing square window).
    Args:
        dim (int): Number of input channels.
        idx (int): The indentix of window. (0/1)
        split_size (tuple(int)): Height and Width of spatial window.
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        attn_drop (float): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float): Dropout ratio of output. Default: 0.0
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        q,k,v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)

        x = windows2img(x, self.H_sp, self.W_sp, H, W)

        return x


class Spatial_Frequency_Attention(nn.Module):
    def __init__(self, dim, num_heads,
                 reso=64, split_size=[8,8], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., b_idx=0, enable_sfa_symmetry=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx  = b_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.hf = nn.Linear(dim, dim, bias=qkv_bias)
        self.enable_sfa_symmetry = enable_sfa_symmetry  # 接收 SFA 对称开关

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.dw_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )

        self.attns = nn.ModuleList([
                Spatial_Attention(
                    dim//2, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
                for i in range(self.branch_num)])
        if self.b_idx > 0 and (self.b_idx - 2) % 4 == 0:
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.channel_projection = ChannelProjection(dim)
        self.spatial_projection = SpatialProjection(dim)
        self.frequency_projection = FrequencyProjection(dim)
        # face-space symmetry (horizontal) 
        self.sym_weight = nn.Parameter(torch.tensor(0.0))

    def calculate_mask(self, H, W):
        img_mask_0 = torch.zeros((1, H, W, 1))  
        img_mask_1 = torch.zeros((1, H, W, 1))  
        h_slices_0 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                    slice(-self.split_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                    slice(-self.split_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1)
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W, sem=None):
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        hf = self.hf(x).transpose(-2,-1).contiguous().view(B, C, H, W)
        hf = self.frequency_projection(hf)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) 
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2)
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1) 
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if self.b_idx  > 0 and (self.b_idx  - 2) % 4 == 0:
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            attened_x = torch.cat([x1,x2], dim=2)

        conv_x = self.dw_block(v)

        # C-Map
        channel_map = self.channel_projection(conv_x)
        conv_x = conv_x  + channel_map
        hf = hf + channel_map
        channel_map = reduce(channel_map, 'b c h w -> b c 1 1', 'mean').permute(0, 2, 3, 1).contiguous().view(B, 1, C)

        # S-Map 
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_projection(attention_reshape)
        
        # 依据开关决定是否执行人脸空间对称机制
        if self.enable_sfa_symmetry and spatial_map is not None:
            w = torch.sigmoid(self.sym_weight)
            spatial_map = (1 - w) * spatial_map + w * torch.flip(spatial_map, dims=[3])
            
        hf = hf + attention_reshape
        if self.enable_sfa_symmetry and hf is not None:
            w = torch.sigmoid(self.sym_weight)
            hf = (1 - w) * hf + w * torch.flip(hf, dims=[3])

        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map) * torch.sigmoid(reduce(hf, 'b c h w -> b c 1 1', 'mean').permute(0, 2, 3, 1).contiguous().view(B, 1, C))
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x * torch.sigmoid(hf)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x + hf.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Channel_Transposed_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., semantic_ch: int = 0, enable_cta_component=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.channel_projection = ChannelProjection(dim)
        self.spatial_projection = SpatialProjection(dim)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
        )

        # optional facial component channel interaction (semantic-guided)
        self.semantic_ch = int(semantic_ch)
        self.enable_cta_component = enable_cta_component # 接收 CTA 组件交互开关
        self.comp_interact = ComponentChannelInteraction(dim=dim, semantic_ch=self.semantic_ch) if self.semantic_ch > 0 else None

    def forward(self, x, H, W, sem=None):
        # 依据开关强制切断语义信号流入 CTA 进行组件交互
        if not self.enable_cta_component:
            sem = None

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        conv_x = self.dwconv(v_)

        # semantic-guided component channel interaction (optional)
        if sem is not None and self.comp_interact is not None:
            gate = self.comp_interact(conv_x, sem)
            if gate is not None:
                conv_x = conv_x + gate

        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        channel_map = self.channel_projection(attention_reshape)
        attened_x = attened_x + channel_map.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        channel_map = reduce(channel_map, 'b c h w -> b c 1 1', 'mean')

        spatial_map = self.spatial_projection(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, C)

        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FCA(nn.Module):
    def __init__(self, dim, num_heads, reso=64, split_size=[2,4],shift_size=[1,2], expansion_factor=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, b_idx=0, semantic_ch: int = 0,
                 enable_sfa_symmetry=True, enable_cta_component=True, enable_dffn_ms_freq=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if b_idx % 2 == 0:
            # SFA
            self.attn = Spatial_Frequency_Attention(
                dim, num_heads=num_heads, reso=reso, split_size=split_size, shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, b_idx=b_idx, enable_sfa_symmetry=enable_sfa_symmetry
            )
        else:
            # CTA
            self.attn = Channel_Transposed_Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, semantic_ch=semantic_ch, enable_cta_component=enable_cta_component
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        ffn_hidden_dim = int(dim * expansion_factor)
        # DFFN
        self.ffn = DFFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer, enable_dffn_ms_freq=enable_dffn_ms_freq)

    def forward(self, x, x_size, edge=None, sem=None):
        H , W = x_size
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, sem=sem))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W, sem=sem))
        return x

class FDM(nn.Module):
    def __init__(self,
                 dim,
                 expansion_factor=4.,
                 semantic_ch: int = 0,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 enable_edge_guidance=True,
                 enable_sem_guidance=True,
                 enable_dffn_ms_freq=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cm_block = Channel_mixing(dim)
        self.semantic_ch = int(semantic_ch)
        self.guidance = GuidanceEncoder(dim, semantic_ch=self.semantic_ch)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = DFFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer, enable_dffn_ms_freq=enable_dffn_ms_freq)
        self.norm2 = norm_layer(dim)

        self.enable_edge_guidance = enable_edge_guidance
        self.enable_sem_guidance = enable_sem_guidance

    def forward(self, x, x_size, edge=None, sem=None):
        H , W = x_size
        
        # 依据开关强制阻断掩码输入，实现消融
        if not self.enable_edge_guidance: edge = None
        if not self.enable_sem_guidance: sem = None

        g_low, g_high = self.guidance(edge=edge, sem=sem)
        x = x + self.drop_path(self.cm_block(self.norm1(x), H, W, g_low=g_low, g_high=g_high))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x

class ResidualGroup(nn.Module):
    def __init__(self, dim, reso, num_heads, semantic_ch: int = 0, split_size=[2,4], expansion_factor=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_paths=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=2, use_chk=False,
                 enable_edge_guidance=True, enable_sem_guidance=True, enable_sfa_symmetry=True, enable_cta_component=True, enable_dffn_ms_freq=True):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso
        self.depth = depth

        self.cm_blocks = nn.ModuleList([
            FDM(
                dim=dim,
                expansion_factor=expansion_factor,
                semantic_ch=semantic_ch,
                act_layer=act_layer,
                norm_layer=norm_layer,
                enable_edge_guidance=enable_edge_guidance,
                enable_sem_guidance=enable_sem_guidance,
                enable_dffn_ms_freq=enable_dffn_ms_freq
            ) for _ in range(2)
        ])
        self.hf_layer = HFRB(dim)

        self.hfrb = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
            ) for _ in range(depth // 2)
        ])

        self.attn_blocks = nn.ModuleList([
            FCA(
                dim=dim,
                num_heads=num_heads,
                reso = reso,
                split_size = split_size,
                shift_size = [split_size[0]//2, split_size[1]//2],
                expansion_factor=expansion_factor,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                b_idx = i,
                semantic_ch=semantic_ch,
                enable_sfa_symmetry=enable_sfa_symmetry,
                enable_cta_component=enable_cta_component,
                enable_dffn_ms_freq=enable_dffn_ms_freq
            ) for i in range(depth)])

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, edge=None, sem=None):
        B, N, C = x.shape
        H, W = x_size
        res = x

        for cm_blk in self.cm_blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(cm_blk, x, x_size, edge, sem)
            else:
                x = cm_blk(x, x_size, edge=edge, sem=sem)
        x_high = self.hf_layer(x, x_size)
        k = 0
        for attn_blk, i in zip(self.attn_blocks, range(self.depth)):
            if self.use_chk:
                x = checkpoint.checkpoint(attn_blk, x, x_size, edge, sem)
            else:
                x = attn_blk(x, x_size, edge=edge, sem=sem)
            if i % 2 == 1 and  k < self.depth // 2 :
                x = x + self.hfrb[k](x_high).permute(0, 2, 3, 1).contiguous().view(B, N, C)
                k += 1
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = res + x
        return x

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops



@ARCH_REGISTRY.register()
class EFD(nn.Module):
    """EFD face super-resolution network; historical state-dict names are preserved."""
    def __init__(self,
                 img_size=64,
                 in_chans=3,
                 semantic_ch: int = 0,
                 embed_dim=180,
                 split_size=[2,4],
                 depth=18,
                 num_heads=2,
                 expansion_factor=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_chk=False,
                 upscale=2,
                 img_range=1.,
                 enable_edge_guidance: bool = True,
                 enable_sem_guidance: bool = True,
                 enable_sfa_symmetry: bool = True,
                 enable_cta_component: bool = True,
                 enable_dffn_ms_freq: bool = True,
                 **kwargs
                 ):
        super().__init__()

        self.enable_edge_guidance = bool(enable_edge_guidance)
        self.enable_sem_guidance = bool(enable_sem_guidance)
        self.enable_sfa_symmetry = bool(enable_sfa_symmetry)
        self.enable_cta_component = bool(enable_cta_component)
        self.enable_dffn_ms_freq = bool(enable_dffn_ms_freq)

        num_in_ch = in_chans
        num_out_ch = in_chans
        self.img_range = img_range
        
        # 使用 register_buffer 确保均值张量随模型移动
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.register_buffer('mean', torch.Tensor(rgb_mean).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, 1, 1, 1))
            
        self.upscale = upscale
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.use_chk = use_chk
        self.semantic_ch = int(semantic_ch)
        self.num_features = self.embed_dim = embed_dim

        # Internal prior extraction from LR input. This keeps the external input as LR-only.
        # External edge/sem can still be passed to forward(); if not passed, these branches generate them.
        self.edge_extractor = LightweightEdgeExtractor() if self.enable_edge_guidance else None
        self.sem_extractor = None
        if self.enable_sem_guidance and self.semantic_ch > 0:
            self.sem_extractor = LightweightSemanticExtractor(
                in_chans=in_chans,
                semantic_ch=self.semantic_ch,
                base_ch=16
            )

        self.before_RG = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.rg = ResidualGroup(
            dim=embed_dim, semantic_ch=semantic_ch, num_heads=num_heads, reso=img_size,
            split_size=split_size, expansion_factor=expansion_factor, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_paths=dpr, act_layer=act_layer, norm_layer=norm_layer,
            depth=depth, use_chk=use_chk,
            enable_edge_guidance=enable_edge_guidance,
            enable_sem_guidance=enable_sem_guidance,
            enable_sfa_symmetry=enable_sfa_symmetry,
            enable_cta_component=enable_cta_component,
            enable_dffn_ms_freq=enable_dffn_ms_freq)

        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (img_size, img_size))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, edge=None, sem=None):
        _, _, H, W = x.shape
        x_size = [H, W]
        x = self.before_RG(x)
        x = self.rg(x, x_size, edge=edge, sem=sem)
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x

    def forward(self, x, edge=None, sem=None):
        # LR-only inference: if external priors are absent, generate them internally from LR.
        lr_input = x

        if edge is None and self.edge_extractor is not None:
            edge = self.edge_extractor(lr_input)

        if sem is None and self.sem_extractor is not None:
            sem = self.sem_extractor(lr_input)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x, edge=edge, sem=sem)) + x
        x = self.upsample(x)
        x = x / self.img_range + self.mean
        return x

    # ======= 新增集成的方法 =======

    def get_flops(self, input_size=(3, 64, 64)):
        """ 使用 thop 计算 FLOPs """
        device = next(self.parameters()).device
        input_tensor = torch.randn(1, *input_size).to(device)
        flops, params = profile(self, inputs=(input_tensor,), verbose=False)
        return flops, params

    def get_inference_time(self, input_size=(3, 64, 64), warmup=10, repeats=100):
        """ 计算推理时间（毫秒） """
        device = next(self.parameters()).device
        input_tensor = torch.randn(1, *input_size).to(device)
        self.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = self(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(repeats):
                _ = self(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) * 1000 / repeats
        return avg_time

    def __repr__(self):
        # Logging must not run inference, change RNG state, or register THOP buffers.
        count = sum(p.numel() for p in self.parameters())
        return f'EFD(upscale={self.upscale}, parameters={count})'
