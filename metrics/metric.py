# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: metric.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import torch
import math


def mse_torch(
    img1: torch.Tensor,
    img2: torch.Tensor
    ) -> float:
    """
    Compute Mean Squared Error (MSE) between two images.

    Args:
        img1 (torch.Tensor):
            First image tensor.
        img2 (torch.Tensor):
            Second image tensor.

    Returns:
        float: MSE value.
    """

    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, "
            f"got {img1.shape} and {img2.shape}."
        )

    # Convert to float64 for higher precision
    img1 = img1.detach().to(dtype=torch.float64, device="cpu")
    img2 = img2.detach().to(dtype=torch.float64, device="cpu")

    # Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2).item()
    return mse


def mae_torch(
    img1: torch.Tensor,
    img2: torch.Tensor
    ) -> float:
    """
    Compute Mean Absolute Error (MAE) between two images.

    Args:
        img1 (torch.Tensor):
            First image tensor.
        img2 (torch.Tensor):
            Second image tensor.

    Returns:
        float: MAE value.
    """

    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, "
            f"got {img1.shape} and {img2.shape}."
        )

    # Convert to float64 for higher precision
    img1 = img1.detach().to(dtype=torch.float64, device="cpu")
    img2 = img2.detach().to(dtype=torch.float64, device="cpu")

    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(img1 - img2)).item()
    return mae


def psnr_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float=None
    ) -> float:
    """
    Compute PSNR between two images.

    Args:
        img1 (torch.Tensor):
            First image tensor.
        img2 (torch.Tensor):
            Second image tensor.
        data_range (float, optional):
            The data range of the input images (i.e., the difference between
            the maximum and minimum possible values). If None, it is
            determined from the input images.

    Returns:
        float: PSNR value in decibels (dB).
    """

    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, "
            f"got {img1.shape} and {img2.shape}."
        )

    # Convert to float64 for higher precision
    img1 = img1.detach().to(dtype=torch.float64, device="cpu")
    img2 = img2.detach().to(dtype=torch.float64, device="cpu")

    # Mean Squared Error (MSE)
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:  # Two images are identical
        return float("inf")

    # Dynamic range: if data_range is not provided, estimate it from the max-min of the two images
    if data_range is None:
        max_val = torch.max(torch.stack([img1.max(), img2.max()])).item()
        min_val = torch.min(torch.stack([img1.min(), img2.min()])).item()
        data_range = max_val - min_val

        # If max == min (constant image), do a fallback
        if data_range <= 0:
            data_range = max(abs(max_val), abs(min_val))
            if data_range == 0:
                # Both images are all zeros and identical → mse is already 0, return inf
                return float("inf")

    # PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    """Create 2D Gaussian window of shape (1, 1, ws, ws)."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)   # outer product -> (ws, ws)
    window_2d = window_2d / window_2d.sum()
    return window_2d.unsqueeze(0).unsqueeze(0)    # (1, 1, ws, ws)


def ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = None,
    window_size: int = 11,
    sigma: float = 1.5,
    K: tuple[float, float] = (0.01, 0.03),
    ) -> float:
    """
    Compute SSIM between two images (single pair, non-batched).

    Args:
        img1 (torch.Tensor):
            First image tensor of shape (H, W) or (C, H, W).
        img2 (torch.Tensor):
            Second image tensor of shape (H, W) or (C, H, W).
        data_range (float, optional):
            The data range of the input images (i.e., the difference between
            the maximum and minimum possible values). If None, it is
            determined from the input images.
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
    else:
        raise ValueError(f"Only supports 2D or 3D tensors, got dim={img1.dim()}.")

    # Convert to float64 for higher precision
    device = img1.device
    img1 = img1.detach().to(dtype=torch.float64, device=device)
    img2 = img2.detach().to(dtype=torch.float64, device=device)

    _, C, H, W = img1.shape
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    if window_size > min(H, W):
        raise ValueError(f"window_size ({window_size}) must be <= min(H, W) ({min(H, W)}).")

    # Dynamic range estimation
    if data_range is None:
        max_val = torch.max(torch.stack([img1.max(), img2.max()])).item()
        min_val = torch.min(torch.stack([img1.min(), img2.min()])).item()
        data_range = max_val - min_val
        if data_range <= 0:
            # Use absolute max value as fallback
            data_range = max(abs(max_val), abs(min_val))
            if data_range == 0:
                return 1.0  # Both images are constant and identical

    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # Create Gaussian window
    base_window = _gaussian_window(window_size, sigma, device=device, dtype=torch.float64)
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
