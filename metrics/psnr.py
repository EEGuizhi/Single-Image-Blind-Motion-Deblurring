# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: psnr.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217

    The PSNR metric (without alignment) used during training.
"""

import math
import torch


def psnr_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float=1.0
) -> float:
    """
    Compute PSNR between two images.

    Args:
        img1 (torch.Tensor):
            First image tensor, shape (B, C, H, W).
        img2 (torch.Tensor):
            Second image tensor, shape (B, C, H, W).
        data_range (float, optional):
            The data range of the input images (i.e., the difference between
            the maximum and minimum possible values). Default is 1.0.

    Returns:
        float: PSNR value in decibels (dB).
    """

    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, "
            f"got {img1.shape} and {img2.shape}."
        )

    # Convert to float64 for higher precision
    img1 = img1.detach()
    img2 = img2.detach()

    # Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:  # Two images are identical
        return float("inf")

    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
