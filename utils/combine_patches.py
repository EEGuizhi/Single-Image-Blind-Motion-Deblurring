# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: combine_patches.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""


import numpy as np
import torch


def combine_patches_numpy(
    patches: np.ndarray,
    image_size: tuple,
    start_pos: np.ndarray,
    hamming_window: bool=True
    ) -> np.ndarray:
    """
    Combine overlapping patches into a full image.

    Args:
        patches (np.ndarray):
            Array of image patches with shape (N, H_patch, W_patch, C).
        image_size (tuple):
            Size of the full image (height, width).
        start_pos (np.ndarray):
            Array of starting indices for each patch with shape (N, 2),
            where each entry is (start_H, start_W).

    Returns:
        np.ndarray: Combined full image.
    """
    H, W = image_size
    N, patch_H, patch_W, C = patches.shape

    # Initialize Hamming window if needed
    if hamming_window:
        hamming_H = np.hamming(patch_H)
        hamming_W = np.hamming(patch_W)
        hamming_2d = np.outer(hamming_H, hamming_W)
        hamming_2d = np.expand_dims(hamming_2d, axis=-1)  # Shape: (H_patch, W_patch, 1)

    # Initialize full image and weight matrix
    full_image = np.zeros((H, W, C), dtype=np.float32)
    weight_matrix = np.zeros((H, W, C), dtype=np.float32)
    for i in range(N):
        start_H, start_W = start_pos[i]
        end_H = start_H + patch_H
        end_W = start_W + patch_W

        if hamming_window:
            weighted_patch = patches[i] * hamming_2d
            full_image[start_H:end_H, start_W:end_W, :] += weighted_patch
            weight_matrix[start_H:end_H, start_W:end_W, :] += hamming_2d
        else:
            full_image[start_H:end_H, start_W:end_W, :] += patches[i]
            weight_matrix[start_H:end_H, start_W:end_W, :] += 1.0

    # Avoid division by zero
    weight_matrix[weight_matrix == 0] = 1.0
    combined_image = full_image / weight_matrix

    return combined_image



def combine_patches_torch(
    patches: torch.Tensor,
    image_size: tuple,
    start_pos: torch.Tensor,
    hamming_window: bool=True
    ) -> torch.Tensor:
    """
    Combine overlapping patches into a full image using PyTorch.

    Args:
        patches (torch.Tensor):
            Tensor of image patches with shape (N, C, H_patch, W_patch).
        image_size (tuple):
            Size of the full image (height, width).
        start_pos (torch.Tensor):
            Tensor of starting indices for each patch with shape (N, 2),
            where each entry is (start_H, start_W).

    Returns:
        torch.Tensor: Combined full image with shape (C, H, W).
    """
    H, W = image_size
    N, C, patch_H, patch_W = patches.shape

    # Initialize Hamming window if needed
    if hamming_window:
        hamming_H = torch.hamming_window(patch_H, periodic=False, device=patches.device)
        hamming_W = torch.hamming_window(patch_W, periodic=False, device=patches.device)
        hamming_2d = torch.ger(hamming_H, hamming_W).unsqueeze(0)  # Shape: (1, H_patch, W_patch)

    # Initialize full image and weight matrix
    full_image = torch.zeros((C, H, W), dtype=patches.dtype, device=patches.device)
    weight_matrix = torch.zeros((C, H, W), dtype=patches.dtype, device=patches.device)
    for i in range(N):
        start_H, start_W = start_pos[i]
        end_H = start_H + patch_H
        end_W = start_W + patch_W

        if hamming_window:
            weighted_patch = patches[i] * hamming_2d
            full_image[:, start_H:end_H, start_W:end_W] += weighted_patch
            weight_matrix[:, start_H:end_H, start_W:end_W] += hamming_2d
        else:
            full_image[:, start_H:end_H, start_W:end_W] += patches[i]
            weight_matrix[:, start_H:end_H, start_W:end_W] += 1.0

    # Avoid division by zero
    weight_matrix[weight_matrix == 0] = 1.0
    combined_image = full_image / weight_matrix

    return combined_image
