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
