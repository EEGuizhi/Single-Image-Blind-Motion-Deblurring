# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: loss.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from optimize.loss_util import weighted_loss


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error, MAE) Loss Function"""
    def __init__(self, loss_weight: float=1.0, reduction: str='mean'):
        super(L1Loss, self).__init__()
        assert reduction in ['none', 'mean', 'sum'], f'Unsupported reduction mode: {reduction}.'

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.l1_loss(pred, target, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) Loss Function"""
    def __init__(self, loss_weight: float=1.0, reduction: str='mean'):
        super(MSELoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum'], f'Unsupported reduction mode: {reduction}.'

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * F.mse_loss(pred, target, reduction=self.reduction)


class PSNRLoss(nn.Module):
    """PSNR Loss Function"""
    def __init__(self, loss_weight: float=1.0, reduction: str='mean'):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_weight * self.scale * torch.log(
            ((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8
        ).mean()


class SIMOLoss(nn.Module):
    """Single Input Multi-Output Loss Function"""
    def __init__(self, reduction: str='mean', loss_weight: float=1.0, eps: float=1e-8):
        super(SIMOLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.resize_kwargs = dict(mode='area', recompute_scale_factor=True)
        self.eps = eps

    def forward(self, batch_p: list[torch.Tensor], batch_l: torch.Tensor) -> torch.Tensor:
        loss = self.loss_weight * self.scale * torch.log(
            ((batch_p[0] - batch_l) ** 2).mean(dim=(1, 2, 3)) + self.eps
        ).mean()

        loss += 0.5 * self.loss_weight * self.scale * torch.log(
            ((batch_p[1] - F.interpolate(batch_l, scale_factor=0.5, **self.resize_kwargs)) ** 2)
            .mean(dim=(1, 2, 3)) + self.eps
        ).mean()

        loss += 0.25 * self.loss_weight * self.scale * torch.log(
            ((batch_p[2] - F.interpolate(batch_l, scale_factor=0.25, **self.resize_kwargs)) ** 2)
            .mean(dim=(1, 2, 3)) + self.eps
        ).mean()

        loss += 0.125 * self.loss_weight * self.scale * torch.log(
            ((batch_p[3] - F.interpolate(batch_l, scale_factor=0.125, **self.resize_kwargs)) ** 2)
            .mean(dim=(1, 2, 3)) + self.eps
        ).mean()

        return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float=1e-3, reduction: str="mean"):
        super().__init__()
        if reduction != "mean" and reduction != "sum" and reduction != "None":
            raise ValueError("Reduction type not supported")
        else:
            self.reduction = reduction
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = output - target

        out = torch.sqrt((diff * diff) + (self.eps * self.eps))
        if self.reduction == "mean":
            out = torch.mean(out)
        elif self.reduction == "sum":
            out = torch.sum(out)

        return out


class EdgeLoss(nn.Module):
    def __init__(self, weight: float=0.05):
        """
        Taken from:
        https://github.com/swz30/MPRNet/blob/main/Deblurring/losses.py
        """
        super(EdgeLoss, self).__init__()
        self.weight = weight
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img: torch.Tensor) -> torch.Tensor:
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current: torch.Tensor) -> torch.Tensor:
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return self.weight * loss
