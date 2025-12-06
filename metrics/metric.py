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
import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity

# ------------------------------------------------------------------------------ #
# RealBlur / MLWNet style evaluation (numpy + OpenCV)
# ------------------------------------------------------------------------------ #

def _image_align_np(
    deblurred: np.ndarray,
    gt: np.ndarray,
    n_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    deblurred, gt: H x W x 3, float32, range [0, 1], RGB
    returns: aligned_deblurred, aligned_gt, mask, warp_matrix
    """
    # this function is based on kohler evaluation code
    z = deblurred
    x = gt

    # simple intensity matching
    zs = (np.sum(x * z) / np.sum(z * z)) * z

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    number_of_iterations = n_iterations
    termination_eps = 0.0
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations, termination_eps
    )

    # ECC on grayscale
    cc, warp_matrix = cv2.findTransformECC(
        cv2.cvtColor(x,  cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
        warp_matrix, warp_mode, criteria,
        inputMask=None, gaussFiltSize=5
    )

    target_shape = x.shape  # (H, W, C)

    zr = cv2.warpPerspective(
        zs,
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )

    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix,
        (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    zr = zr * cr
    xr = x * cr

    return zr, xr, cr, warp_matrix


def _compute_psnr_np(image_true: np.ndarray,
                     image_test: np.ndarray,
                     image_mask: np.ndarray,
                     data_range: float = 1.0) -> float:
    """
    image_true, image_test, image_mask: H x W x C, float32
    """
    err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
    return 10.0 * np.log10((data_range ** 2) / err)


def _compute_ssim_np(
    tar_img: np.ndarray,
    prd_img: np.ndarray,
    cr1: np.ndarray
) -> float:
    """
    tar_img, prd_img, cr1: H x W x C, float32, range [0,1]
    RealBlur style: RGB SSIM + mask + crop boundary.
    """
    # skimage SSIM (RGB)
    ssim_pre, ssim_map = structural_similarity(
        tar_img, prd_img,
        multichannel=True,
        gaussian_weights=True,
        use_sample_covariance=False,
        data_range=1.0,
        channel_axis=-1,
        full=True
    )

    # apply mask
    ssim_map = ssim_map * cr1

    # same crop as RealBlur script
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2

    ssim_crop = ssim_map[pad:-pad, pad:-pad, :]
    crop_cr1 = cr1[pad:-pad, pad:-pad, :]

    # mean over spatial with mask per-channel, then mean over channels
    ssim = ssim_crop.sum(axis=0).sum(axis=0) / crop_cr1.sum(axis=0).sum(axis=0)
    ssim = float(np.mean(ssim))
    return ssim


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype, channels: int):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    window = g[:, None] @ g[None, :]  # (K,K)
    window = window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def _ssim_map_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    K=(0.01, 0.03)
):
    """
    img1, img2: [1, C, H, W]  float64
    return SSIM map: [1, C, H, W]
    """
    device = img1.device
    C = img1.size(1)

    # Gaussian kernel (groups=C → weight must be [C,1,k,k])
    window = _gaussian_window(window_size, sigma, device=device, dtype=img1.dtype, channels=C)
    padding = window_size // 2

    # local means
    mu1 = torch.nn.functional.conv2d(img1, window, padding=padding, groups=C)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=padding, groups=C)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # variances & covariance
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=padding, groups=C) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=padding, groups=C) - mu2_sq
    sigma12   = torch.nn.functional.conv2d(img1 * img2, window, padding=padding, groups=C) - mu1_mu2

    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return num / (den + 1e-12)


def _compute_ssim_torch(
    aligned_prd, aligned_tar, mask_np,
    data_range=1.0, window_size=11, sigma=1.5
) -> float:
    """
    aligned_* : numpy HWC float32
    mask_np   : numpy HWC float32
    return scalar SSIM
    """

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # numpy -> torch
    prd = torch.from_numpy(aligned_prd).permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float64)
    tar = torch.from_numpy(aligned_tar).permute(2,0,1).unsqueeze(0).to(device=device, dtype=torch.float64)

    # mask (take 1 channel)
    mask = torch.from_numpy(mask_np[...,0]).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float64)
    mask = mask.expand(1, prd.size(1), -1, -1)

    # SSIM map
    ssim_map = _ssim_map_torch(
        tar, prd,
        data_range=data_range,
        window_size=window_size,
        sigma=sigma
    )  # [1,C,H,W]

    # RealBlur crop
    r = int(3.5 * 1.5 + 0.5)
    win = 2*r + 1
    pad = (win - 1) // 2

    if pad > 0 and prd.size(2) > 2*pad and prd.size(3) > 2*pad:
        ssim_map = ssim_map[:,:,pad:-pad, pad:-pad]
        mask = mask[:,:,pad:-pad, pad:-pad]

    # masked average
    ssim = (ssim_map * mask).sum() / (mask.sum() + 1e-12)
    return float(ssim)


def realblur_psnr_ssim_torch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    data_range: float = 1.0
) -> tuple[float, float]:
    """
    RealBlur / MLWNet style PSNR & SSIM with ECC alignment and mask,
    but interface is PyTorch tensor.

    Args:
        pred : torch.Tensor
            [B, 3, H, W], float, range [0, 1], model prediction (deblurred)
        gt : torch.Tensor
            [B, 3, H, W], float, range [0, 1], ground truth
        data_range : float

    Returns:
        avg_psnr : float
        avg_ssim : float
    """
    assert pred.shape == gt.shape, "pred and gt must have same shape"
    assert pred.dim() == 4 and pred.size(1) == 3, "Expect [B, 3, H, W] tensors"

    pred = pred.clamp(0.0, 1.0)
    gt   = gt.clamp(0.0, 1.0)

    B, C, H, W = pred.shape

    psnr_list = []
    ssim_list = []

    for b in range(B):
        # [C,H,W] -> [H,W,C]， numpy float32
        prd_np = pred[b].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        tar_np = gt[b].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)

        # start_time = time.time()
        aligned_prd, aligned_tar, mask, _ = _image_align_np(prd_np, tar_np, n_iterations=30)
        # end_time = time.time()
        # print(f"Image alignment took {end_time - start_time:.4f} seconds")

        # start_time = time.time()
        psnr_val = _compute_psnr_np(aligned_tar, aligned_prd, mask, data_range=data_range)
        # end_time = time.time()
        # print(f"PSNR computation took {end_time - start_time:.4f} seconds")

        # start_time = time.time()
        ssim_val = _compute_ssim_torch(aligned_tar, aligned_prd, mask)
        # end_time = time.time()
        # print(f"SSIM computation took {end_time - start_time:.4f} seconds")

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    avg_psnr = float(sum(psnr_list) / len(psnr_list))
    avg_ssim = float(sum(ssim_list) / len(ssim_list))

    return avg_psnr, avg_ssim


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


# ------------------------------------------------------------------------------ #
# Metrics during training
# ------------------------------------------------------------------------------ #


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
