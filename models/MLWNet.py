#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18/04/2023 1:10 am
# @Author  : Tianheng Qiu
# @FileName: MLWNet_arch.py
# @Software: PyCharm

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


'''
ref. 
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


# --------------------------------------------------------------------------------------- #


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# --------------------------------------------------------------------------------------- #


import math
import torch
import torch.nn as nn

import numbers

from torch.nn import init
import torch.nn.functional as F

# from basicsr.models.archs.arch_util import LayerNorm2d
from models.wavelet_block import LWN
# from models.ours.wavelet_block import ResBlock_dwt


class WaveletBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.wavelet_block1 = LWN(c, wavelet='haar', initialize=True)
        # self.wavelet_block1 = FFT2(c)
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2, out_channels=c, kernel_size=1,
            padding=0, stride=1, groups=1, bias=True
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1,
                padding=0, stride=1, groups=1, bias=True
            ),
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c, out_channels=ffn_channel, kernel_size=1,
            padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2, out_channels=c, kernel_size=1,
            padding=0, stride=1, groups=1, bias=True
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.wavelet_block1(x)

        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)

        # gate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

    def get_wavelet_loss(self):
        return self.wavelet_block1.get_wavelet_loss()


# SEB
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=dw_channel, kernel_size=1,
            padding=0, stride=1, groups=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel, out_channels=dw_channel, kernel_size=3,
            padding=1, stride=1, groups=dw_channel, bias=True
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2, out_channels=c, kernel_size=1,
            padding=0, stride=1, groups=1, bias=True
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1,
                padding=0, stride=1, groups=1, bias=True
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c, out_channels=ffn_channel, kernel_size=1,
            padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2, out_channels=c, kernel_size=1,
            padding=0, stride=1, groups=1, bias=True
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma

    def get_wavelet_loss(self):
        return 0.


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Encoder(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        dim=32,
        num_blocks=[2, 4, 4, 6],
        ):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.feature_embed = nn.Conv2d(
            in_channels=inp_channels, out_channels=dim, kernel_size=3,
            padding=1, stride=1, groups=1, bias=True
        )
        self.b1 = nn.Sequential(*[NAFBlock(dim) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, 2 * dim, 2, 2)
        self.b2 = nn.Sequential(*[NAFBlock(dim * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 2 ** 2, 2, 2)
        self.b3 = nn.Sequential(*[NAFBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 2, 2)
        self.b4 = nn.Sequential(*[NAFBlock(dim * 2 ** 3) for _ in range(num_blocks[3])])

    def forward(self, x):
        x = self.feature_embed(x)  # (1, 32, 256, 256)
        x1 = self.b1(x)  # (1, 32, 256, 256)

        x = self.down1(x1)  # (1, 64, 128, 128)
        x2 = self.b2(x)  # (1, 64, 128, 128)

        x = self.down2(x2)  # (1, 128, 64, 64)
        x3 = self.b3(x)  # (1, 128, 64, 64)

        x = self.down3(x3)
        x4 = self.b4(x)

        return x4, x3, x2, x1


class Fusion(nn.Module):
    def __init__(
        self,
        dim=32,
        num_blocks=[2, 4, 4, 6],
        ):
        super(Fusion, self).__init__()
        self.num_blocks = num_blocks
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.d3 = nn.Sequential(*[WaveletBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.d2 = nn.Sequential(*[WaveletBlock(dim * 2) for _ in range(num_blocks[1])])

    def forward(self, x4, x3, x2, x1):
        x3_b = x3.contiguous()
        x = self.up43(x4) + x3
        x3 = self.d3(x)
        # deblur head x3(min) 128
        x2_b = x2.contiguous()
        x = self.up32(x3) + x2
        x2 = self.d2(x)
        return x4, x3, x3_b, x2, x2_b, x1

    def get_wavelet_loss(self):
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            if _ is not None:
                for block in getattr(self, f'd{index+1}'):
                    wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss

class Deblur_head(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()
        self.block = nn.Sequential(
            # nn.Conv2d(num_in, num_mid, kernel_size=1),
            # nn.BatchNorm2d(num_mid),
            # nn.GELU(),
            nn.Conv2d(num_in, num_out, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dim=64,
        out_channels=3,
        num_blocks=[2, 4, 4, 6],
        ):
        super().__init__()
        self.num_blocks = num_blocks
        self.head4 = Deblur_head(int(dim * 2 ** 3), int(dim * 3), 3)
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.head3 = Deblur_head(int(dim * 2 ** 2), int(dim * 2 ** 1), out_channels)
        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.head2 = Deblur_head(int(dim * 2 ** 1), int(dim), out_channels)
        self.up21 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 2, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.head1 = Deblur_head(dim, dim, out_channels)

        self.d4 = nn.Sequential(*[WaveletBlock(dim * 2 ** 3) for _ in range(num_blocks[3])])
        self.d3 = nn.Sequential(*[WaveletBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.d2 = nn.Sequential(*[WaveletBlock(dim * 2) for _ in range(num_blocks[1])])
        self.d1 = nn.Sequential(*[WaveletBlock(dim) for _ in range(num_blocks[0])])
        self.alpha = nn.Parameter(torch.zeros((1, dim * 2, 1, 1)), requires_grad=True)

    def forward(self, x4, x3, x3_b, x2, x2_b, x1):
        # x = x4.contiguous()
        x = self.d4(x4)
        x4 = self.head4(x)  # if self.training else None

        x = self.up43(x) + x3
        x = self.d3(x)
        x3 = self.head3(x)  # if self.training else None

        x2_n = x2.contiguous()
        x = self.up32(x) + x2
        x = self.d2(x)
        x2 = self.head2(x)  # if self.training else None

        x = self.up21(x + x2_n * self.alpha) + x1
        x = self.d1(x)
        x1 = self.head1(x)

        return x1, x2, x3, x4

    def get_wavelet_loss(self):
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            for block in getattr(self, f'd{index+1}'):
                wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss


class MLWNet(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=64,
        ):

        super(MLWNet, self).__init__()
        # [False, True, True, False]
        # [False, False, False, False]
        self.encoder = Encoder(
            inp_channels=inp_channels,
            dim=dim,
            num_blocks=[1, 2, 4, 24],
        )
        self.fusion = Fusion(
            dim=dim,
            num_blocks=[None, 2, 2, None],
        )
        self.decoder = Decoder(
            dim=dim,
            out_channels=out_channels,
            num_blocks=[2, 2, 2, 2],
        )

    def __repr__(self):
        return 'MLWNet'

    def forward(self, inp):
        x = self.encoder(inp)  # (1, 128, 64, 64), (1, 64, 128, 128), (1, 32, 256, 256)
        x = self.fusion(*x)  # (1, 128, 64, 64), (1, 64, 128, 128), (1, 32, 256, 256)
        x1, x2, x3, x4 = self.decoder(*x)
        return x1 + inp, x2, x3, x4

    def get_wavelet_loss(self):
        return self.fusion.get_wavelet_loss() + self.decoder.get_wavelet_loss()


class MLWNet_Local(Local_Base, MLWNet):
    def __init__(self, *args, base_size=None, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MLWNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        if base_size is not None:
            base_size = (int(base_size), int(base_size))
        else:
            base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)
