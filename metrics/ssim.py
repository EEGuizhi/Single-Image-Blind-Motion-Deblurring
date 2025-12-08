# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: ssim.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217

    The SSIM metric (without alignment) used during training.
"""

import math
import torch


def _gaussian_window(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
    channels: int = 1
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    window = g[:, None] @ g[None, :]  # (K,K)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    K: tuple[float, float] = (0.01, 0.03),
) -> float:
    """
    Compute SSIM between two images (single pair, non-batched).

    Args:
        img1 (torch.Tensor):
            First image tensor of shape (H, W) or (C, H, W) or (B, C, H, W).
        img2 (torch.Tensor):
            Second image tensor of shape (H, W) or (C, H, W) or (B, C, H, W).
        data_range (float, optional):
            The data range of the input images (i.e., the difference between
            the maximum and minimum possible values). Default is 1.0.
        window_size (int, optional):
            Size of the Gaussian window. Default is 11.
        sigma (float, optional):
            Standard deviation of the Gaussian window. Default is 1.5.
        K (tuple of float, optional):
            Constants for SSIM calculation. Default is (0.01, 0.03).

    Returns:
        float: SSIM value between img1 and img2.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Input images must have the same shape, "
                         f"got {img1.shape} and {img2.shape}.")

    if img1.dim() == 2:
        # (H, W) -> (1, 1, H, W)
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        # (C, H, W) -> (1, C, H, W)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Convert to float64 for higher precision
    device = img1.device
    img1 = img1.detach()
    img2 = img2.detach()

    B, C, H, W = img1.shape
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    if window_size > min(H, W):
        raise ValueError(f"window_size ({window_size}) must be <= min(H, W) ({min(H, W)}).")

    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Create Gaussian window
    base_window = _gaussian_window(window_size, sigma, device=device, dtype=img1.dtype)
    window = base_window.expand(C, 1, window_size, window_size)  # (C, 1, ws, ws)

    padding = window_size // 2

    # μ1, μ2
    mu1 = torch.nn.functional.conv2d(img1, window, padding=padding, groups=C)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=padding, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # σ1^2, σ2^2, σ12
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=padding, groups=C) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=padding, groups=C) - mu2_sq
    sigma12   = torch.nn.functional.conv2d(img1 * img2, window, padding=padding, groups=C) - mu1_mu2

    # SSIM map
    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-12)

    # Average over all spatial dimensions, channels, and batch
    return ssim_map.mean().item()
