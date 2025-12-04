# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: basic_modules.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
    
    Modified from:
    "NAFNet: Nonlinear Activation Free Network for Image Restoration"
    "FFTFormer: Efficient Frequency Domain-based Transformers for High-Quality Image Deblurring"
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """Simple Gate Layer (NAFNet)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GroupSimpleGate(nn.Module):
    """Simple Gate Layer (NAFNet-like) with optional grouping"""
    def __init__(self, group: int = 1):
        super(GroupSimpleGate, self).__init__()
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.group
        assert C % (2 * g) == 0, "Channels must be divisible by 2*group"

        x = x.view(B, g, 2, C // (2 * g), H, W)  # [B, group, 2, C//(2*group), H, W]
        y = x[:, :, 0] * x[:, :, 1]
        return y.reshape(B, C // 2, H, W)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.Function,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float
    ) -> torch.Tensor:
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.Function,
        grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
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
    def __init__(self, channels: int, eps: float = 1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
