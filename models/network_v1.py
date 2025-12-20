# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: network.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_modules import *


# ---------------------------------------------------------------------------------------------- #
# Basic Blocks
# ---------------------------------------------------------------------------------------------- #


class EncoderBlock(nn.Module):
    """Basic Encoder Block, modified from NAFBlock"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: list[int] = [1],
        expand_conv: float = 2.0,
        expand_ffn: float = 2.0
    ) -> None:
        # Initialization
        super(EncoderBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        dw_channels  = round(in_channels * expand_conv)
        ffn_channels = round(in_channels * expand_ffn)
        mid_channels = [dw_channels // 2, ffn_channels // 2]

        # Convolutional Layers
        self.conv1_pw = nn.Conv2d(in_channels, dw_channels, kernel_size=1, stride=1, padding=0)
        # self.conv2_dw = nn.Conv2d(
        #     dw_channels, dw_channels, kernel_size=kernel_size,
        #     stride=stride, padding=padding, dilation=dilation, groups=dw_channels
        # )
        self.conv2_dw = self._make_dwconv_layers(
            channels=dw_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations
        )
        self.conv3_pw = nn.Conv2d(mid_channels[0], in_channels, kernel_size=1, stride=1, padding=0)
        self.conv4_pw = nn.Conv2d(in_channels, ffn_channels, kernel_size=1, stride=1, padding=0)
        self.conv5_pw = nn.Conv2d(mid_channels[1], out_channels, kernel_size=1, stride=1, padding=0)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=mid_channels[0], out_channels=mid_channels[0],
                kernel_size=1, stride=1, padding=0
            )
        )

        # Shortcut Layers
        self.shortcut1 = (
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 else nn.Identity()
        )
        self.shortcut2 = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        # Activation (Simple Gate) and Normalization
        self.gate1 = GroupSimpleGate(group=len(dilations))
        self.gate2 = GroupSimpleGate()
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------------- Part I ----------------- #
        residual = self.shortcut1(x)
        x = self.norm1(x)
        x = self.conv1_pw(x)

        # Split channels for different dilated convolutions
        x_splits = torch.chunk(x, len(self.conv2_dw), dim=1)
        x_dwconv = [conv(x_split) for conv, x_split in zip(self.conv2_dw, x_splits)]
        x = torch.cat(x_dwconv, dim=1)

        x = self.gate1(x)
        x = self.sca(x) * x
        x = self.conv3_pw(x)
        x += residual

        # ----------------- Part II ----------------- #
        residual = self.shortcut2(x)
        x = self.norm2(x)
        x = self.conv4_pw(x)
        x = self.gate2(x)
        x = self.conv5_pw(x)
        x += residual

        return x


class DFFN(nn.Module):
    """Discriminative frequency domain-based FFN Module, modified from FFTFormer"""
    def __init__(
        self,
        dim: int,
        expand_ffn: float = 2.0,
        patch_size: int = 8,
        bias: bool = True
    ) -> None:
        # Initialization
        super(DFFN, self).__init__()
        dw_channels = int(dim * expand_ffn)
        self.dim = dim
        self.patch_size = patch_size

        self.project_in = nn.Conv2d(dim, dw_channels * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            dw_channels * 2, dw_channels * 2, kernel_size=3,
            stride=1, padding=1, groups=dw_channels * 2, bias=bias
        )
        self.fft = nn.Parameter(
            torch.ones((dw_channels * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(dw_channels, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)

        b, c, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, \
            "Height and Width must be divisible by patch_size"

        # # Original patch unfolding implementation
        # x_patch = rearrange(
        #     x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
        #     patch1=self.patch_size, patch2=self.patch_size
        # )

        # Non-overlapping patching
        x_patch = x.view(b, c,                        # (B, C, h, ps, w, ps)
            h // self.patch_size, self.patch_size,
            w // self.patch_size, self.patch_size
        )
        x_patch = x_patch.permute(0, 1, 2, 4, 3, 5)   # (B, C, h, w, ps, ps)
        x_patch = x_patch.contiguous()

        # Frequency domain weighting
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # # Original patch folding implementation
        # x = rearrange(
        #     x_patch,
        #     'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
        #     patch1=self.patch_size, patch2=self.patch_size
        # )

        # Non-overlapping unpatching
        x = x_patch.permute(0, 1, 2, 4, 3, 5).contiguous()   # (B, C, h, ps, w, ps)
        x = x.view(b, c, h, w)                               # (B, C, H, W)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class DecoderBlock(nn.Module):
    """Basic Decoder Block"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        kernel_size: int = 5,
        stride: int = 1,
        dilations: list[int] = [1],
        expand_conv: float = 2.0,
        expand_ffn: float = 2.0
    ) -> None:
        # Initialization
        super(DecoderBlock, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        dw_channels  = round(in_channels * expand_conv)
        ffn_channels = round(in_channels * expand_ffn)
        mid_channels = [dw_channels // 2, ffn_channels // 2]

        # Convolutional Layers
        self.conv1_pw = nn.Conv2d(in_channels, dw_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_dw = self._make_dwconv_layers(
            channels=dw_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations
        )
        self.conv3_pw = nn.Conv2d(mid_channels[0], in_channels, kernel_size=1, stride=1, padding=0)

        # Feed-forward Network
        self.ffn = DFFN(in_channels, expand_ffn=expand_ffn)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=mid_channels[0], out_channels=mid_channels[0],
                kernel_size=1, stride=1, padding=0
            )
        )

        # Shortcut Layers
        self.shortcut1 = (
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 else nn.Identity()
        )
        self.shortcut2 = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        # Activation (Simple Gate) and Normalization
        self.gate1 = GroupSimpleGate(group=len(dilations))
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------------- Part I ----------------- #
        residual = self.shortcut1(x)
        x = self.norm1(x)
        x = self.conv1_pw(x)

        # Split channels for different dilated convolutions
        x_splits = torch.chunk(x, len(self.conv2_dw), dim=1)
        x_dwconv = [conv(x_split) for conv, x_split in zip(self.conv2_dw, x_splits)]
        x = torch.cat(x_dwconv, dim=1)

        x = self.gate1(x)
        x = self.sca(x) * x
        x = self.conv3_pw(x)
        x += residual

        # ----------------- Part II ----------------- #
        residual = self.shortcut2(x)
        x = self.norm2(x)
        x = self.ffn(x)
        x += residual

        return x


# ---------------------------------------------------------------------------------------------- #
# Core Modules
# ---------------------------------------------------------------------------------------------- #


class FeatureEmbedding(nn.Module):
    """Feature Embedding Part"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        # Initialization
        super(FeatureEmbedding, self).__init__()

        # Convolutional Layer
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """Encoder Part"""
    def __init__(
        self,
        dim: int = 32,
        expand_dim: float = 2.0,
        num_blocks: list = [2, 4, 4, 6],
    ) -> None:
        # Initialization
        super(Encoder, self).__init__()
        dims = [round(dim * (expand_dim ** i)) for i in range(len(num_blocks))]

        # Stage 1
        self.fuse1 = nn.Sequential(*[
            EncoderBlock(dims[0], dims[0], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[0])
        ])
        self.down1 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)

        # Stage 2
        self.fuse2 = nn.Sequential(*[
            EncoderBlock(dims[1], dims[1], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[1])
        ])
        self.down2 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)

        # Stage 3
        self.fuse3 = nn.Sequential(*[
            EncoderBlock(dims[2], dims[2], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[2])
        ])
        self.down3 = nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)

        # Stage 4
        self.fuse4 = nn.Sequential(*[
            EncoderBlock(dims[3], dims[3], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[3])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fuse1(x)
        x = self.down1(x1)

        x2 = self.fuse2(x)
        x = self.down2(x2)

        x3 = self.fuse3(x)
        x = self.down3(x3)

        x4 = self.fuse4(x)
        return x1, x2, x3, x4


class Bridge(nn.Module):
    """Bridge Part (not used yet)"""
    def __init__(
        self,
        dim: int,
        expand_dim: float = 2.0,
        num_blocks: list = [0, 2, 2, 0],
    ) -> None:
        # Initialization
        super(Bridge, self).__init__()
        assert len(num_blocks) == 4, "num_blocks must have four elements."
        assert num_blocks[-1] == 0, "The last element of num_blocks must be zero."
        dims = [round(dim * (expand_dim ** i)) for i in range(len(num_blocks))]

        # Stage 4
        self.up4 = nn.Sequential(
            nn.Conv2d(dims[3], dims[3], kernel_size=3, padding=1, groups=dims[3]),
            nn.Conv2d(dims[3], dims[2] * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

        # Stage 3
        self.ch_reduce3 = nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1)
        self.fuse3 = nn.Sequential(*[
            DecoderBlock(dims[2], dims[2], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[2])
        ])
        self.up3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1, groups=dims[2]),
            nn.Conv2d(dims[2], dims[1] * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

        # Stage 2
        self.ch_reduce2 = nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1)
        self.fuse2 = nn.Sequential(*[
            DecoderBlock(dims[1], dims[1], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[1])
        ])
        self.up2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], kernel_size=3, padding=1, groups=dims[1]),
            nn.Conv2d(dims[1], dims[0] * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

        # Stage 1
        self.ch_reduce1 = nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1)
        self.fuse1 = nn.Sequential(*[
            DecoderBlock(dims[0], dims[0], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[0])
        ]) if num_blocks[0] > 0 else nn.Identity()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor
    ) -> torch.Tensor:
        # Stage 4
        x = self.up4(x4)
        
        # Stage 3
        x3 = torch.cat([x3, x], dim=1)
        x3 = self.ch_reduce3(x3)
        x3 = self.fuse3(x3)
        x = self.up3(x3)
        
        # Stage 2
        x2 = torch.cat([x2, x], dim=1)
        x2 = self.ch_reduce2(x2)
        x2 = self.fuse2(x2)
        x = self.up2(x2)
        
        # Stage 1
        x1 = torch.cat([x1, x], dim=1)
        x1 = self.ch_reduce1(x1)
        x1 = self.fuse1(x1)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    """Decoder Part"""
    def __init__(
        self,
        out_channels: int,
        dim: int,
        expand_dim: float = 2.0,
        num_blocks: list = [2, 4, 4, 6],
        aux_heads: bool = False
    ) -> None:
        # Initialization
        super(Decoder, self).__init__()
        self.aux_heads = aux_heads
        dims = [round(dim * (expand_dim ** i)) for i in range(len(num_blocks))]

        # Stage 4
        self.fuse4 = nn.Sequential(*[
            DecoderBlock(dims[3], dims[3], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[3])
        ])
        self.up4 = nn.Sequential(
            nn.Conv2d(dims[3], dims[3], kernel_size=3, padding=1, groups=dims[3]),
            nn.Conv2d(dims[3], dims[2] * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

        # Stage 3
        self.ch_reduce3 = nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1)
        self.fuse3 = nn.Sequential(*[
            DecoderBlock(dims[2], dims[2], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[2])
        ])
        self.up3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1, groups=dims[2]),
            nn.Conv2d(dims[2], dims[1] * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

        # Stage 2
        self.ch_reduce2 = nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1)
        self.fuse2 = nn.Sequential(*[
            DecoderBlock(dims[1], dims[1], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[1])
        ])
        self.up2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], kernel_size=3, padding=1, groups=dims[1]),
            nn.Conv2d(dims[1], dims[0] * 4, kernel_size=1),
            nn.PixelShuffle(2)
        )

        # Stage 1
        self.ch_reduce1 = nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1)
        self.fuse1 = nn.Sequential(*[
            DecoderBlock(dims[0], dims[0], dilations=[1, 2], expand_conv=2, expand_ffn=2)
            for _ in range(num_blocks[0])
        ])

        # Deblurring Heads
        if aux_heads:
            self.aux_head4 = nn.Conv2d(dims[3], out_channels, kernel_size=3, padding=1)
            self.aux_head3 = nn.Conv2d(dims[2], out_channels, kernel_size=3, padding=1)
            self.aux_head2 = nn.Conv2d(dims[1], out_channels, kernel_size=3, padding=1)
        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor
    ) -> torch.Tensor:
        # Stage 4
        x4 = self.fuse4(x4)
        aux4 = self.aux_head4(x4) if self.aux_heads else None
        x4 = self.up4(x4)

        # Stage 3
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.ch_reduce3(x3)
        x3 = self.fuse3(x3)
        aux3 = self.aux_head3(x3) if self.aux_heads else None
        x3 = self.up3(x3)

        # Stage 2
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.ch_reduce2(x2)
        x2 = self.fuse2(x2)
        aux2 = self.aux_head2(x2) if self.aux_heads else None
        x2 = self.up2(x2)

        # Stage 1
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.ch_reduce1(x1)
        x1 = self.fuse1(x1)
        out = self.head(x1)

        return out, aux2, aux3, aux4


# ---------------------------------------------------------------------------------------------- #
# Top Module
# ---------------------------------------------------------------------------------------------- #


class Network(nn.Module):
    """Lightweight FFT-Based Network for Image Deblurring"""
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        expand_dim: float = 2.0,
        aux_heads: bool = True,
        use_bridge: bool = True
    ) -> None:
        # Initialization
        super(Network, self).__init__()

        # Modules
        self.embedding = FeatureEmbedding(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.encoder = Encoder(
            dim=dim,
            expand_dim=expand_dim,
            num_blocks=[1, 2, 4, 24]
        )
        self.bridge = Bridge(
            dim=dim,
            expand_dim=expand_dim,
            num_blocks=[0, 2, 2, 0]
        ) if use_bridge else None
        self.decoder = Decoder(
            out_channels=out_channels,
            dim=dim,
            expand_dim=expand_dim,
            num_blocks=[2, 3, 4, 6],
            aux_heads=aux_heads
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x1, x2, x3, x4 = self.encoder(x)
        if self.bridge is not None:
            x1, x2, x3, x4 = self.bridge(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.decoder(x1, x2, x3, x4)
        return x1, x2, x3, x4
