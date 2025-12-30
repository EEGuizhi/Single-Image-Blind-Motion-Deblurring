# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: predict.py
Author: JRKang
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
    
    Predict deblurred image from a single blurred input image.
"""

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

import torch
import torchvision

import models
from configs.config import *
from utils.combine_patches import combine_patches_torch


def split_image_to_patches(
    image: np.ndarray,
    patch_size: tuple[int, int],
    overlap: tuple[int, int]
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split image into overlapping patches.

    Args:
        image (np.ndarray): Input image (H, W, C).
        patch_size (tuple[int, int]): Size of each patch (H, W).
        overlap (tuple[int, int]): Overlap between patches (H_overlap, W_overlap).

    Returns:
        tuple: List of patches and list of patch positions.
    """
    h, w = image.shape[:2]
    patch_h, patch_w = patch_size
    overlap_h, overlap_w = overlap
    step_h = patch_h - overlap_h
    step_w = patch_w - overlap_w

    # Calculate patch indices
    h_indices = list(range(0, h - patch_h + 1, step_h))
    w_indices = list(range(0, w - patch_w + 1, step_w))

    # Ensure the entire image is covered
    if h_indices[-1] + patch_h < h:
        h_indices.append(h - patch_h)
    if w_indices[-1] + patch_w < w:
        w_indices.append(w - patch_w)

    # Extract patches
    patches = []
    positions = []
    for h_idx in h_indices:
        for w_idx in w_indices:
            patch = image[h_idx:h_idx+patch_h, w_idx:w_idx+patch_w, :]
            patches.append(patch)
            positions.append((h_idx, w_idx))

    return patches, positions


def predict(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    patch_size: tuple[int, int] = (256, 256),
    overlap: tuple[int, int] = (128, 128),
    device: torch.device = torch.device('cuda'),
    batch_size: int = 4
) -> None:
    """Predict deblurred image from a single blurred input.

    Args:
        model (torch.nn.Module): The trained deblurring model.
        input_path (str): Path to the input blurred image.
        output_path (str): Path to save the output deblurred image.
        patch_size (tuple[int, int]): Size of image patches (H, W).
        overlap (tuple[int, int]): Overlap size between patches (H_overlap, W_overlap).
        device (torch.device): Device to run inference on.
        batch_size (int): Batch size for processing patches.
    """
    # Read input image
    print(f"[Predict] Loading image from {input_path}")
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image from {input_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    print(f"[Predict] Image size: {orig_h} x {orig_w}")

    image_mean = torch.from_numpy(np.mean(image_rgb, axis=(0, 1), keepdims=True)).float() / 255.0
    image_mean = image_mean.permute(2, 0, 1)

    # Split image into patches
    print(f"[Predict] Splitting image into patches (size={patch_size}, overlap={overlap})...")
    patches, positions = split_image_to_patches(image_rgb, patch_size, overlap)
    print(f"[Predict] Total patches: {len(patches)}")

    # Prepare model
    model.eval()
    model.to(device)

    # Process patches in batches
    output_patches = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Processing patches"):
            # Get batch of patches
            batch_patches = patches[i : i+batch_size]

            # Convert to tensor and normalize
            batch_tensor = []
            for patch in batch_patches:
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                batch_tensor.append(patch_tensor)

            batch_tensor = torch.stack(batch_tensor, dim=0).to(device)

            # Forward pass
            batch_tensor = batch_tensor / SCALE
            model_outputs = model(batch_tensor)
            outputs = model_outputs[0] if isinstance(model_outputs, (list, tuple)) else model_outputs
            outputs = outputs * SCALE

            # Store results
            for j in range(outputs.size(0)):
                output_patches.append(outputs[j].cpu())

    # Combine patches
    print(f"[Predict] Combining patches...")
    output_patches_tensor = torch.stack(output_patches, dim=0)
    positions_tensor = torch.tensor(positions, dtype=torch.long)

    full_size = (orig_h, orig_w)
    combined_output = combine_patches_torch(output_patches_tensor, full_size, positions_tensor)
    combined_output = combined_output / combined_output.mean(dim=(1, 2), keepdim=True) * image_mean
    combined_output = torch.clamp(combined_output, 0.0, 1.0)

    # Convert to numpy and save
    output_image = combined_output.permute(1, 2, 0).cpu().numpy()
    output_image = (output_image * 255.0).astype(np.uint8)
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save output
    cv2.imwrite(output_path, output_image_bgr)
    print(f"[Predict] Deblurred image saved to {output_path}")


if __name__ == "__main__":
    # Read settings from configs.config.PREDICT_CONFIG
    MODEL_NAME  = PREDICT_CONFIG.get('model_name', 'Network')
    MODEL_DIM   = PREDICT_CONFIG.get('model_dim', 32)
    WEIGHTS     = PREDICT_CONFIG.get('weights_path', f"{ROOT_DIR}/pretrain_weights/Network_d32_best.pth")
    INPUT_PATH  = PREDICT_CONFIG.get('input_path')
    OUTPUT_PATH = PREDICT_CONFIG.get('output_path')
    PRED_SINGLE = PREDICT_CONFIG.get('predict_single_image')
    INPUT_DIR   = PREDICT_CONFIG.get('input_dir')
    OUTPUT_DIR  = PREDICT_CONFIG.get('output_dir')
    PATCH_SIZE  = PREDICT_CONFIG.get('patch_size', (256, 256))
    OVERLAP     = PREDICT_CONFIG.get('overlap', (128, 128))
    BATCH_SIZE  = PREDICT_CONFIG.get('batch_size', 4)
    DEVICE_STR  = PREDICT_CONFIG.get('device', 'cuda')
    SCALE       = 1.1

    # Setup
    device = torch.device(DEVICE_STR if torch.cuda.is_available() else 'cpu')
    print(f"[Predict] Using device: {device}")
    print(f"[Predict] Model: {MODEL_NAME}, Dim: {MODEL_DIM}")
    print(f"[Predict] Weights: {WEIGHTS}")
    print(f"[Predict] Input: {INPUT_PATH}")
    print(f"[Predict] Output: {OUTPUT_PATH}")
    # Predict-only; no GT path

    # Validate required paths based on mode
    if PRED_SINGLE:
        # Single image mode
        if not INPUT_PATH or not OUTPUT_PATH:
            raise ValueError("PREDICT_CONFIG must set 'input_path' and 'output_path' for single image mode.")
    else:
        # Batch directory mode
        if not INPUT_DIR or not OUTPUT_DIR:
            raise ValueError("PREDICT_CONFIG must set 'input_dir' and 'output_dir' for batch mode.")

    # Load model
    print(f"[Predict] Loading model...")
    model = models.load_model(MODEL_NAME, dim=MODEL_DIM, aux_heads=False)
    model = models.load_weights(model, WEIGHTS)
    model = model.to(device)
    print(f"[Predict] Model loaded successfully")

    # Run prediction
    start_time = time.time()

    if PRED_SINGLE:
        # Single image prediction
        print(f"\n[Predict] Running in SINGLE IMAGE mode")
        predict(
            model=model,
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            device=device,
            batch_size=BATCH_SIZE
        )
    else:
        # Batch directory prediction
        print(f"\n[Predict] Running in BATCH DIRECTORY mode")
        print(f"[Predict] Input directory: {INPUT_DIR}")
        print(f"[Predict] Output directory: {OUTPUT_DIR}")

        # Supported image extensions
        supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in os.listdir(INPUT_DIR)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]
        image_files.sort()
        print(f"[Predict] Found {len(image_files)} images to process")

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Process each image
        successful = 0
        failed = 0
        for idx, img_file in enumerate(image_files, 1):
            input_file_path = os.path.join(INPUT_DIR, img_file)
            name, ext = os.path.splitext(img_file)
            output_file_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_{name.replace('blur', 'sharp')}{ext}")

            print(f"\n[Predict] [{idx}/{len(image_files)}] Processing: {img_file}")
            try:
                predict(
                    model=model,
                    input_path=input_file_path,
                    output_path=output_file_path,
                    patch_size=PATCH_SIZE,
                    overlap=OVERLAP,
                    device=device,
                    batch_size=BATCH_SIZE
                )
                successful += 1
            except Exception as e:
                print(f"[Predict] ERROR: {e}")
                failed += 1
                continue

        print(f"\n[Predict] Batch processing completed: {successful} successful, {failed} failed")

    elapsed_time = time.time() - start_time
    print(f"\n[Predict] Total time: {elapsed_time:.2f} seconds")
