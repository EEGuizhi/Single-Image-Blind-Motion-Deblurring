# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: test.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import os
import cv2
import math
import time
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import models
from configs.config import *
from metrics import *
from datasets.dataset import *
from utils.misc import *
from utils.combine_patches import *


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    img_size: tuple[int, int],
    device: torch.device,
    factor: int = None,
    ecc_iters: int = 100,
    save_output: bool = False,
    show_images: list[int] = None
    ) -> tuple[float, float]:
    """Testing loop for evaluating the model on the test dataset.
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        img_size (tuple[int, int]): Size of the input image patches (height, width).
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        factor (int, optional): Factor to pad image size to be divisible by this number. Default is None.
        ecc_iters (int, optional): Number of ECC iterations for image alignment. Default is 100.
        save_output (bool, optional): Whether to save the output images. Default is False.
        show_images (list[int], optional): List of image indices to save outputs for visualization.
    Returns:
        tuple[float, float]: Average PSNR and SSIM over the test dataset.
    """
    # Model preparation
    model.eval()
    model.to(device)

    # Testing loop
    with torch.no_grad():
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0

        prd_patches, tgt_patches = None, None
        for batch_data in tqdm(test_loader, desc="Testing", unit="batch"):
            # Prepare data
            inputs    = batch_data[0].to(device)
            targets   = batch_data[1].to(device)
            patch_pos = batch_data[2].to(device)
            img_idx   = batch_data[3]
            n_patches = batch_data[4]
            save_path = batch_data[5][0].replace('/blur/', '/pred/').replace('/blur_', '/sharp_')

            if ORIG_SIZE:
                size = inputs.size()
                if size[2] % factor != 0 or size[3] % factor != 0:
                    new_h = math.ceil(size[2] / factor) * factor
                    new_w = math.ceil(size[3] / factor) * factor
                    inputs = F.pad(inputs, (0, new_w - size[3], 0, new_h - size[2]), mode='reflect')

            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            # Get full size image
            if ORIG_SIZE:
                outputs = outputs[:, :, 0:size[2], 0:size[3]]
                combined_output = outputs.detach().squeeze(0)
                combined_target = targets.detach().squeeze(0)
            else:
                # Collect patches for each image
                if prd_patches is None:
                    patches_cnt = 0
                    prd_patches = torch.zeros((n_patches, 3, img_size[0], img_size[1]), device=device)
                    tgt_patches = torch.zeros((n_patches, 3, img_size[0], img_size[1]), device=device)
                    start_pos   = torch.zeros((n_patches, 2), dtype=torch.long, device=device)

                prd_patches[patches_cnt:patches_cnt + inputs.size(0)] = outputs.detach()
                tgt_patches[patches_cnt:patches_cnt + inputs.size(0)] = targets.detach()
                start_pos[patches_cnt:patches_cnt + inputs.size(0)] = patch_pos.detach()
                patches_cnt += inputs.size(0)

            # Evaluate and save predictions
            if patches_cnt == n_patches or ORIG_SIZE:
                # Combine patches if all patches are collected
                if not ORIG_SIZE:
                    full_image_size = (
                        start_pos[-1, 0] + img_size[0],
                        start_pos[-1, 1] + img_size[1]
                    )
                    combined_output = combine_patches_torch(prd_patches, full_image_size, start_pos)
                    combined_target = combine_patches_torch(tgt_patches, full_image_size, start_pos)

                # Save output image
                if save_output:
                    save_image_tensor(combined_output, save_path)

                # Save example outputs
                if show_images is not None and img_idx.item() in show_images:
                    torchvision.utils.save_image(
                        combined_output.clamp(0, 1),
                        f"{OUTPUT_DIR}/images/output_img_{img_idx.item()}.png"
                    )
                    torchvision.utils.save_image(
                        combined_target.clamp(0, 1),
                        f"{OUTPUT_DIR}/images/target_img_{img_idx.item()}.png"
                    )

                # Compute metrics
                psnr, ssim = realblur_psnr_ssim_torch(
                    combined_output.unsqueeze(0), combined_target.unsqueeze(0), ecc_iters=ecc_iters
                )

                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1

                # Reset for next image
                if not ORIG_SIZE:
                    prd_patches, tgt_patches = None, None

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    return avg_psnr, avg_ssim


if __name__ == "__main__":
    # Initialization
    MODEL_NAME  = TEST_CONFIG["model_name"]
    MODEL_DIM   = TEST_CONFIG["model_dim"]
    WEIGHT_PATH = TEST_CONFIG["weights_path"]
    BATCH_SIZE  = TEST_CONFIG["batch_size"]
    IMG_SIZE    = TEST_CONFIG["patch_size"]
    OVERLAP     = TEST_CONFIG["overlap"]
    ORIG_SIZE   = TEST_CONFIG["orig_size"]
    FACTOR      = TEST_CONFIG["factor"]
    ECC_ITERS   = TEST_CONFIG["ecc_iters"]
    SAVE_OUTPUT = TEST_CONFIG["save_outputs"]
    SHOW_IMAGES = TEST_CONFIG["show_image_indices"]
    NUM_WORKERS = TEST_CONFIG["num_workers"]

    # Logger setup
    LOG_PATH = (
        f"{OUTPUT_DIR}/reports"
        + f"/report"
        + f"_{MODEL_NAME}"
        + f"_d{MODEL_DIM}"
        + (f"_{IMG_SIZE[0]}_{OVERLAP[0]}" if ORIG_SIZE is False else f"_OrigSize")
        + ".txt"
    )
    log = logger(LOG_PATH)
    log.print_log(f">> Starting Model Testing")

    # Start timing
    start_time = time.time()
    log.print_log(f"Start Time: {time.ctime(start_time)}\n")

    # Change execution directory to project root
    os.chdir(ROOT_DIR)
    log.print_log(f"Current working directory: {os.getcwd()}")
    log.print_log(f"Model Weights Path: {WEIGHT_PATH}\n")

    # Show configurations
    log.print_log(f">> Testing Configuration:")
    for key, value in TEST_CONFIG.items():
        log.print_log(f"    - {key}: {value}")
    log.print_log("")

    # Load dataset
    if USE_DATASETS == 'GoPro':
        test_dataset = GoProDataset(
            split='test',
            img_size=IMG_SIZE, overlap=OVERLAP, orig_size=ORIG_SIZE,
            root=DATASET_ROOT, cache_size=CACHE_SIZE, with_path=True
        )
    else:
        test_dataset = RealBlurDataset(
            split='test', img_type=IMG_TYPE,
            img_size=IMG_SIZE, overlap=OVERLAP, orig_size=ORIG_SIZE,
            root=DATASET_ROOT, cache_size=CACHE_SIZE, with_path=True
        )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=NUM_WORKERS
    )

    # Load model
    model = models.load_model(MODEL_NAME, dim=MODEL_DIM, aux_heads=False)
    model = models.load_weights(model, WEIGHT_PATH)
    model = model.to(DEVICE)

    # Run testing
    avg_psnr, avg_ssim = test(
        model, test_loader, IMG_SIZE, DEVICE,
        FACTOR, ECC_ITERS, SAVE_OUTPUT, SHOW_IMAGES
    )
    log.print_log(f"Average PSNR: {avg_psnr:.5f} dB")
    log.print_log(f"Average SSIM: {avg_ssim:.5f}")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Log results
    log.print_log(f"\nEnd Time: {time.ctime(end_time)}")
    log.print_log(f"Total Elapsed Time: {elapsed_time/60:.3f} minutes")
