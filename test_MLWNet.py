# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: test_MLWNet.py
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
from torch.utils.data import DataLoader, Dataset

from configs.config import *
from utils.misc import *
from datasets.dataset import RealBlurDataset, custom_collate_fn
from datasets.augmentation import RealBlurAugmentation
from models.MLWNet import MLWNet_Local
# from metrics.psnr_ssim import PSNR, SSIM  # original metrics
from metrics.metric import *
from utils.combine_patches import combine_patches_torch


SHOW_IMAGE_INDICES = [3, 8, 18]  # Indices of images to save during testing


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    show_images: list[int]=None
    ) -> tuple[float, float]:
    """Testing loop for evaluating the model on the test dataset.
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
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

            # Forward pass
            outputs = model(inputs)[0]

            # Collect patches for each image
            if prd_patches is None:
                patches_cnt = 0
                prd_patches = torch.zeros((n_patches, 3, IMG_SIZE[0], IMG_SIZE[1]), device=device)
                tgt_patches = torch.zeros((n_patches, 3, IMG_SIZE[0], IMG_SIZE[1]), device=device)
                start_pos   = torch.zeros((n_patches, 2), dtype=torch.long, device=device)

            prd_patches[patches_cnt:patches_cnt + inputs.size(0)] = outputs.detach()
            tgt_patches[patches_cnt:patches_cnt + inputs.size(0)] = targets.detach()
            start_pos[patches_cnt:patches_cnt + inputs.size(0)] = patch_pos.detach()
            patches_cnt += inputs.size(0)

            # If all patches for the current image are collected, combine and evaluate
            if patches_cnt == n_patches:
                full_image_size = (
                    start_pos[-1, 0] + IMG_SIZE[0],
                    start_pos[-1, 1] + IMG_SIZE[1]
                )
                combined_output = combine_patches_torch(prd_patches, full_image_size, start_pos)
                combined_target = combine_patches_torch(tgt_patches, full_image_size, start_pos)

                # Save random sample outputs
                if show_images is not None and img_idx.item() in show_images:
                    torchvision.utils.save_image(
                        combined_output.clamp(0, 1),
                        f"{OUTPUT_DIR}/output_img_{img_idx.item()}.png"
                    )
                    torchvision.utils.save_image(
                        combined_target.clamp(0, 1),
                        f"{OUTPUT_DIR}/target_img_{img_idx.item()}.png"
                    )

                # Compute metrics
                psnr = psnr_torch(combined_output, combined_target)
                ssim = ssim_torch(combined_output, combined_target)

                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1

                # Reset for next image
                prd_patches, tgt_patches = None, None

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    return avg_psnr, avg_ssim


def load_model(
    model: torch.nn.Module,
    weights_path: str,
    device: torch.device
    ) -> torch.nn.Module:
    """Load model weights from a specified path.
    Args:
        model (torch.nn.Module): The model architecture.
        weights_path (str): Path to the model weights file.
        device (torch.device): Device to load the model onto (CPU or GPU).
    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict["params"])
    model.to(device)
    return model


if __name__ == "__main__":
    # Log start
    log = logger(TEST_RESULT_LOG)
    log.print_log(f">> Starting MLWNet Testing")

    # Start timing
    start_time = time.time()
    log.print_log(f"Start Time: {time.ctime(start_time)}\n")

    # Change execution directory to project root
    os.chdir(ROOT_DIR)
    log.print_log(f"Current working directory: {os.getcwd()}")
    log.print_log(f"Model Weights Path: {MODEL_WEIGHTS_PATH}\n")

    log.print_log(f"Image Size: {IMG_SIZE}, Overlap: {OVERLAP}")
    log.print_log(f"Show Image Indices: {SHOW_IMAGE_INDICES}")

    # Load dataset
    test_dataset = RealBlurDataset(
        split='test', img_type=IMG_TYPE, img_size=IMG_SIZE,
        overlap=OVERLAP, root=DATASET_ROOT, cache_size=CACHE_SIZE
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=8
    )

    # Load model
    model = MLWNet_Local(dim=32 if "32" in MODEL_WEIGHTS_PATH else 64)
    model = load_model(model, MODEL_WEIGHTS_PATH, DEVICE)

    # Run testing
    avg_psnr, avg_ssim = test(model, test_loader, DEVICE, SHOW_IMAGE_INDICES)
    log.print_log(f"Average PSNR: {avg_psnr:.5f} dB")
    log.print_log(f"Average SSIM: {avg_ssim:.5f}")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    log.print_log(f"Testing completed in {elapsed_time/60:.3f} minutes.\n")

    # Log results
    log.print_log(f"End Time: {time.ctime(end_time)}")
    log.print_log(f"Total Elapsed Time: {elapsed_time/60:.3f} minutes")
