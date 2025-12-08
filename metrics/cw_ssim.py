# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: cw_ssim.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import math
import torch
import torch.nn.functional as F


def _complex_gabor_kernel(
    ksize: int,
    sigma: float,
    freq: float,
    theta: float,
    device: torch.device,
    dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a complex Gabor kernel.
    2D Gabor function:
        g(x,y) = exp(-(x'^2 + y'^2)/(2σ^2)) * exp(j * 2π f x')
    Return (real_kernel, imag_kernel), shape = (1, 1, ksize, ksize)
    """
    assert ksize % 2 == 1, "ksize must be odd."
    half = ksize // 2

    y, x = torch.meshgrid(
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        torch.arange(-half, half + 1, device=device, dtype=dtype),
        indexing="ij"
    )

    # Rotate coordinates
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)

    gauss = torch.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
    phase = 2.0 * math.pi * freq * x_theta

    real = gauss * torch.cos(phase)
    imag = gauss * torch.sin(phase)

    real = real.view(1, 1, ksize, ksize)
    imag = imag.view(1, 1, ksize, ksize)
    return real, imag


def _gaussian_window(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]              # (ws, ws)
    window = window.view(1, 1, window_size, window_size)
    return window


def cw_ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 7,
    win_sigma: float = 1.5,
    ksize: int = 9,
    wavelet_sigma: float = 2.0,
    wavelet_freq: float = 0.25,
    theta: float = 0.0,
    K: float = 0.01,
) -> float:
    """
    Complex Wavelet SSIM (CW-SSIM) metric (single-scale, single-orientation version).

    Args:
        img1, img2:
            shape = (H, W) or (C, H, W) or (B, C, H, W), values are recommended to be in [0, 1]
        data_range:
            max - min range, default 1.0
        win_size, win_sigma:
            Gaussian window size and sigma for local sum
        ksize, wavelet_sigma, wavelet_freq, theta:
            Complex Gabor wavelet kernel parameters:
            - ksize: kernel size (must be odd)
            - wavelet_sigma: Gaussian envelope σ
            - wavelet_freq: frequency (approximately between 0.1 and 0.4)
            - theta: orientation (0 means horizontal)
        K:
            stability constant (multiplied by data_range^2)

    Returns:
        float: scalar CW-SSIM value
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Input images must have the same shape, "
            f"got {img1.shape} and {img2.shape}."
        )

    # Convert to (B, C, H, W)
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)                # (1,C,H,W)
        img2 = img2.unsqueeze(0)
    elif img1.dim() != 4:
        raise ValueError(f"Only supports 2D, 3D or 4D tensors, got dim={img1.dim()}.")

    img1 = img1.detach()
    img2 = img2.detach()
    device = img1.device
    B, C, H, W = img1.shape

    # Convert to single-channel luminance (Y), CW-SSIM is usually computed on luminance only
    if C == 3:
        r, g, b = img1[:, 0:1], img1[:, 1:2], img1[:, 2:3]
        img1_y = 0.299 * r + 0.587 * g + 0.114 * b

        r2, g2, b2 = img2[:, 0:1], img2[:, 1:2], img2[:, 2:3]
        img2_y = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
    else:
        img1_y = img1.mean(dim=1, keepdim=True)   # (B,1,H,W)
        img2_y = img2.mean(dim=1, keepdim=True)

    # Wavelet kernel (Gabor)
    real_k, imag_k = _complex_gabor_kernel(
        ksize=ksize,
        sigma=wavelet_sigma,
        freq=wavelet_freq,
        theta=theta,
        device=device,
        dtype=img1_y.dtype,
    )
    pad_w = ksize // 2

    # Convolve both images to get complex wavelet coefficients
    real1 = F.conv2d(img1_y, real_k, padding=pad_w)
    imag1 = F.conv2d(img1_y, imag_k, padding=pad_w)
    real2 = F.conv2d(img2_y, real_k, padding=pad_w)
    imag2 = F.conv2d(img2_y, imag_k, padding=pad_w)

    # Complex coefficients c1 = a + j b, c2 = c + j d
    a, b = real1, imag1
    c, d = real2, imag2

    # |c1|^2, |c2|^2
    mag1_sq = a * a + b * b
    mag2_sq = c * c + d * d

    # Σ c1 c2*  real / imag parts (local window sum)
    # c1 c2* = (a + j b)(c - j d) = (ac + bd) + j (bc - ad)
    real_prod = a * c + b * d
    imag_prod = b * c - a * d

    # Gaussian window for local sum
    if win_size % 2 == 0:
        raise ValueError("win_size must be odd.")
    if win_size > min(H, W):
        raise ValueError(f"win_size ({win_size}) must be <= min(H,W) ({min(H,W)}).")

    win = _gaussian_window(win_size, win_sigma, device=device, dtype=img1_y.dtype)
    pad = win_size // 2

    # local sums
    real_sum = F.conv2d(real_prod, win, padding=pad)  # (B,1,H,W)
    imag_sum = F.conv2d(imag_prod, win, padding=pad)
    mag1_sum = F.conv2d(mag1_sq, win, padding=pad)
    mag2_sum = F.conv2d(mag2_sq, win, padding=pad)

    # | Σ c1 c2* |
    cross_mag = torch.sqrt(real_sum * real_sum + imag_sum * imag_sum + 1e-12)

    Kc = (K * data_range) ** 2

    cw_ssim_map = (2.0 * cross_mag + Kc) / (mag1_sum + mag2_sum + Kc)

    # Average over all pixels and batch
    return cw_ssim_map.mean().item()


if __name__ == "__main__":
    # Example usage
    pred = torch.rand(2, 3, 256, 256)       # [B, 3, H, W] in [0, 1]
    target = torch.rand(2, 3, 256, 256)     # [B, 3, H, W] in [0, 1]
    cw_ssim = cw_ssim_torch(pred, target)   # returns float
