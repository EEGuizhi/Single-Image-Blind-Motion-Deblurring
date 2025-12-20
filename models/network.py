# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: network.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
    
    This code is based on:
    https://github.com/thqiu0419/MLWNet/blob/master/basicsr/models/archs/MLWNet_arch.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_modules import *
from models.wavelet_block import LWN


# ---------------------------------------------------------------------------------------------- #
# Basic Blocks
# ---------------------------------------------------------------------------------------------- #


class WaveletBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_conv: float = 2.0,
        expand_ffn: float = 2.0,
        drop_out_rate: float = 0.0
    ) -> None:
        # Initialization
        super().__init__()
        dw_channels  = round(dim * expand_conv)
        ffn_channels = round(dim * expand_ffn)
        mid_channels = [dw_channels // 2, ffn_channels // 2]

        # Layers
        self.wavelet_block1 = LWN(dim, wavelet='haar', initialize=True, kernel_size=[5, 3])
        self.conv3 = nn.Conv2d(mid_channels[0], dim, 1, padding=0, stride=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels[0], mid_channels[0], 1, padding=0, stride=1, bias=True)
        )
        self.conv4 = nn.Conv2d(dim, ffn_channels, 1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(mid_channels[1], dim, 1, padding=0, stride=1, bias=True)
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.gate1 = SimpleGate()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = x_in
        x = self.norm1(x)
        x = self.wavelet_block1(x)

        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = x_in + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)

        x = self.gate1(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

    def get_wavelet_loss(self):
        return self.wavelet_block1.get_wavelet_loss()


class NAFBlock(nn.Module):
    """Simple Embedding Block (NAFNet-like)"""
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        dilations: list[int] = [1],
        expand_conv: float = 2.0,
        expand_ffn: float = 2.0,
        drop_out_rate: float = 0.0
    ) -> None:
        # Initialization
        super().__init__()
        dw_channels  = round(in_channels * expand_conv)
        ffn_channels = round(in_channels * expand_ffn)
        mid_channels = [dw_channels // 2, ffn_channels // 2]

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels, dw_channels, 1, padding=0, stride=1, bias=True)
        # self.conv2 = nn.Conv2d(
        #     in_channels=dw_channels, out_channels=dw_channels, kernel_size=3,
        #     padding=1, stride=1, groups=dw_channels, bias=True
        # )
        self.conv2 = self._make_dwconv_layers(
            channels=dw_channels, kernel_size=kernel_size, stride=1, dilations=dilations
        )
        self.conv3 = nn.Conv2d(mid_channels[0], in_channels, 1, stride=1, padding=0, bias=True)
        self.conv4 = nn.Conv2d(in_channels, ffn_channels, 1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(mid_channels[1], in_channels, 1, padding=0, stride=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels[0], mid_channels[0], 1, padding=0, stride=1, bias=True)
        )

        # Activations
        self.gate1 = GroupSimpleGate(group=len(dilations))
        self.gate2 = GroupSimpleGate()

        # Normalizations
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

        # Dropouts
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

    def _make_dwconv_layers(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        dilations: list[int]
    ) -> nn.ModuleList:
        assert channels % len(dilations) == 0, "Channels must be divisible by the number of dilations."
        sub_ch = channels // len(dilations)
        dwconv_layers = nn.ModuleList()
        for d in dilations:
            p = ((kernel_size - 1) * d) // 2
            dwconv_layers.append(
                nn.Conv2d(
                    sub_ch, sub_ch, kernel_size=kernel_size,
                    stride=stride, padding=p, dilation=d, groups=sub_ch
                )
            )
        return dwconv_layers

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # ----------------- Part I ----------------- #
        x = x_in
        x = self.norm1(x)
        x = self.conv1(x)

        # Split channels for different dilated convolutions
        x_splits = torch.chunk(x, len(self.conv2), dim=1)
        x_dwconv = [conv(x_split) for conv, x_split in zip(self.conv2, x_splits)]
        x = torch.cat(x_dwconv, dim=1)

        x = self.gate1(x)
        x = self.sca(x) * x
        x = self.conv3(x)

        x = self.dropout1(x)
        y = x_in + self.beta * x

        # ----------------- Part II ----------------- #
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate2(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma

    def get_wavelet_loss(self):
        return 0.0


# ---------------------------------------------------------------------------------------------- #
# Core Modules
# ---------------------------------------------------------------------------------------------- #


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 32,
        expand_dim: float = 2.0,
        num_blocks: list[int] = [2, 4, 4, 6],
    ) -> None:
        # Initialization
        super(Encoder, self).__init__()
        dims = [round(dim * (expand_dim ** i)) for i in range(len(num_blocks))]
        self.num_blocks = num_blocks

        # Embedding Layer
        self.feature_embed = nn.Conv2d(in_channels, dim, 3, padding=1, stride=1, bias=True)

        # Stage 1
        self.b1 = nn.Sequential(*[NAFBlock(dims[0], 3, dilations=[1, 2]) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dims[0], dims[1], 2, 2)

        # Stage 2
        self.b2 = nn.Sequential(*[NAFBlock(dims[1], 3, dilations=[1, 2]) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dims[1], dims[2], 2, 2)

        # Stage 3
        self.b3 = nn.Sequential(*[NAFBlock(dims[2], 3, dilations=[1, 2]) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dims[2], dims[3], 2, 2)

        # Stage 4
        self.b4 = nn.Sequential(*[NAFBlock(dims[3], 3, dilations=[1, 2]) for _ in range(num_blocks[3])])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.feature_embed(x)
        x1 = self.b1(x)

        x = self.down1(x1)
        x2 = self.b2(x)

        x = self.down2(x2)
        x3 = self.b3(x)

        x = self.down3(x3)
        x4 = self.b4(x)
        return x1, x2, x3, x4


class Fusion(nn.Module):
    def __init__(
        self,
        dim: int = 32,
        expand_dim: float = 2.0,
        num_blocks: list[int] = [2, 4, 4, 6],
    ) -> None:
        # Initialization
        super(Fusion, self).__init__()
        dims = [round(dim * (expand_dim ** i)) for i in range(len(num_blocks))]
        self.num_blocks = num_blocks

        # Stage 4
        self.up43 = nn.Sequential(
            nn.Conv2d(dims[3], dims[2] * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )

        # Stage 3
        self.ch_reduce3 = nn.Conv2d(dims[2] * 2, dims[2], 1, bias=False)
        self.d3 = nn.Sequential(*[WaveletBlock(dims[2]) for _ in range(num_blocks[2])])
        self.up32 = nn.Sequential(
            nn.Conv2d(dims[2], dims[1] * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )

        # Stage 2
        self.ch_reduce2 = nn.Conv2d(dims[1] * 2, dims[1], 1, bias=False)
        self.d2 = nn.Sequential(*[WaveletBlock(dims[1]) for _ in range(num_blocks[1])])

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage 4
        x = torch.cat([self.up43(x4), x3], dim=1)

        # Stage 3
        x = self.ch_reduce3(x)
        x3 = self.d3(x)
        x = torch.cat([self.up32(x3), x2], dim=1)

        # Stage 2
        x = self.ch_reduce2(x)
        x2 = self.d2(x)

        return x1, x2, x3, x4

    def get_wavelet_loss(self) -> float:
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            if _ is not None:
                for block in getattr(self, f'd{index+1}'):
                    wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss


class DeblurHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        dim: int = 64,
        expand_dim: float = 2.0,
        num_blocks: list[int] = [2, 4, 4, 6],
        aux_heads: bool = False
    ) -> None:
        # Initialization
        super().__init__()
        dims = [round(dim * expand_dim ** i) for i in range(len(num_blocks))]
        self.num_blocks = num_blocks
        self.aux_heads = aux_heads

        # Stage 4
        self.d4 = nn.Sequential(*[WaveletBlock(dims[3]) for _ in range(num_blocks[3])])
        self.up43 = nn.Sequential(
            nn.Conv2d(dims[3], dims[2] * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )

        # Stage 3
        self.ch_reduce3 = nn.Conv2d(dims[2] * 2, dims[2], 1, bias=False)
        self.d3 = nn.Sequential(*[WaveletBlock(dims[2]) for _ in range(num_blocks[2])])
        self.up32 = nn.Sequential(
            nn.Conv2d(dims[2], dims[1] * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )

        # Stage 2
        self.ch_reduce2 = nn.Conv2d(dims[1] * 2, dims[1], 1, bias=False)
        self.d2 = nn.Sequential(*[WaveletBlock(dims[1]) for _ in range(num_blocks[1])])
        self.up21 = nn.Sequential(
            nn.Conv2d(dims[1], dims[0] * 4, 1, bias=False),
            nn.PixelShuffle(2)
        )

        # Stage 1
        self.ch_reduce1 = nn.Conv2d(dims[0] * 2, dims[0], 1, bias=False)
        self.d1 = nn.Sequential(*[WaveletBlock(dims[0]) for _ in range(num_blocks[0])])

        # Deblurring heads
        self.head1 = DeblurHead(dims[0], out_channels)
        if self.aux_heads:
            self.head2 = DeblurHead(dims[1], out_channels)
            self.head3 = DeblurHead(dims[2], out_channels)
            self.head4 = DeblurHead(dims[3], out_channels)

        self.alpha = nn.Parameter(torch.zeros((1, dims[1], 1, 1)), requires_grad=True)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage 4
        x = self.d4(x4)
        x4 = self.head4(x) if self.aux_heads else None

        # Stage 3
        x = torch.cat([self.up43(x), x3], dim=1)
        x = self.ch_reduce3(x)
        x = self.d3(x)
        x3 = self.head3(x) if self.aux_heads else None

        # Stage 2
        x2_n = x2.contiguous()
        x = torch.cat([self.up32(x), x2], dim=1)
        x = self.ch_reduce2(x)
        x = self.d2(x)
        x2 = self.head2(x) if self.aux_heads else None

        # Stage 1
        x = torch.cat([self.up21(x + self.alpha * x2_n), x1], dim=1)
        x = self.ch_reduce1(x)
        x = self.d1(x)
        x1 = self.head1(x)

        return x1, x2, x3, x4

    def get_wavelet_loss(self) -> float:
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            for block in getattr(self, f'd{index+1}'):
                wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss


# ---------------------------------------------------------------------------------------------- #
# Top Module
# ---------------------------------------------------------------------------------------------- #


class Network(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 64,
        expand_dim: float = 2.0,
        aux_heads: bool = False
    ) -> None:
        super(Network, self).__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            dim=dim,
            expand_dim=expand_dim,
            num_blocks=[1, 2, 4, 16],
        )
        self.fusion = Fusion(
            dim=dim,
            num_blocks=[None, 2, 2, None],
        )
        self.decoder = Decoder(
            out_channels=out_channels,
            dim=dim,
            expand_dim=expand_dim,
            num_blocks=[2, 2, 2, 2],
            aux_heads=aux_heads
        )

    def forward(
        self, x_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x_in)
        x = self.fusion(*x)
        x1, x2, x3, x4 = self.decoder(*x)
        return x1 + x_in, x2, x3, x4

    def get_wavelet_loss(self) -> float:
        return self.fusion.get_wavelet_loss() + self.decoder.get_wavelet_loss()
