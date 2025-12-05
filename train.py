# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: train.py
Author: BSChen, JRKang
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
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset

import models
from optimize import *
from configs.config import *
from datasets.dataset import RealBlurDataset, custom_collate_fn
from datasets.augmentation import RealBlurAugmentation
from metrics.metric import *
from utils.misc import *
from utils.combine_patches import combine_patches_torch


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
    ) -> float:
    """Training loop for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on (CPU or GPU).
        epoch (int): Current epoch number.
    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    model.to(device)
    total_loss = 0.0
    num_batches = 0

    for batch_data in tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch"):
        # Prepare data
        inputs = batch_data[0].to(device)
        targets = batch_data[1].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)[0]

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


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


def get_dataloaders(
    batch_size: int,
    img_size: tuple[int, int],
    overlap: tuple[int, int],
    use_augmentation: bool,
    rand_crop: bool,
    num_workers: int
) -> tuple[DataLoader, DataLoader]:
    """Utility function to create training and testing DataLoaders.
    Args:
        batch_size (int): Batch size for DataLoaders.
        img_size (tuple[int, int]): Image patch size (H, W).
        overlap (tuple[int, int]): Overlap size between patches (H_overlap, W_overlap).
        use_augmentation (bool): Whether to use data augmentation for training.
        rand_crop (bool): Whether to use random cropping for training.
        num_workers (int): Number of worker threads for data loading.
    Returns:
        tuple[DataLoader, DataLoader]: Training and testing DataLoaders.
    """
    # Load training dataset
    augmentation = RealBlurAugmentation() if use_augmentation else None
    train_dataset = RealBlurDataset(
        split='train', img_type=IMG_TYPE, img_size=img_size,
        overlap=overlap, root=DATASET_ROOT, cache_size=CACHE_SIZE,
        transform=augmentation, rand_crop=rand_crop
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=num_workers
    )

    # Load testing dataset
    test_dataset = RealBlurDataset(
        split='test', img_type=IMG_TYPE, img_size=img_size,
        overlap=overlap, root=DATASET_ROOT, cache_size=CACHE_SIZE
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Initialization
    MODEL_NAME  = TRAIN_CONFIG["model_name"]
    MODEL_DIM   = TRAIN_CONFIG["model_dim"]
    WEIGHT_PATH = TRAIN_CONFIG["weights_path"]
    BATCH_SIZE  = TRAIN_CONFIG["batch_size"]
    IMG_SIZE    = TRAIN_CONFIG["patch_size"]
    OVERLAP     = TRAIN_CONFIG["overlap"]
    USE_AUGMENT = TRAIN_CONFIG["augmentation"]
    RAND_CROP   = TRAIN_CONFIG["rand_crop"]
    OPTIMIZER   = TRAIN_CONFIG["optimizer"]
    SCHEDULER   = TRAIN_CONFIG["scheduler"]
    NUM_EPOCHS  = TRAIN_CONFIG["num_epochs"]
    LR          = TRAIN_CONFIG["learning_rate"]
    CHECKPOINT  = TRAIN_CONFIG["checkpoint"]
    NUM_WORKERS = TRAIN_CONFIG["num_workers"]

    # Logger setup
    LOG_PATH = (
        EXPERIMENT_DIR
        + f"/train_log"
        + f"_{MODEL_NAME}"
        + f"_d{MODEL_DIM}"
        + f"_{IMG_SIZE[0]}_{OVERLAP[0]}"
        + ".txt"
    )
    log = logger(LOG_PATH)
    csv_log = csv_logger(LOG_PATH.replace(".txt", ".csv"))
    log.print_log(f">> Starting Model Training")

    # Start timing
    start_time = time.time()
    log.print_log(f"Start Time: {time.ctime(start_time)}\n")

    # Change execution directory to project root
    os.chdir(ROOT_DIR)
    log.print_log(f"Current working directory: {os.getcwd()}")

    # Show configurations
    log.print_log(f">> Training Configuration:")
    for key, value in TRAIN_CONFIG.items():
        log.print_log(f"    - {key}: {value}")
    log.print_log("")

    # Get DataLoaders
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        overlap=OVERLAP,
        use_augmentation=USE_AUGMENT,
        rand_crop=RAND_CROP,
        num_workers=NUM_WORKERS
    )

    # Load model
    model = models.load_model(MODEL_NAME, dim=MODEL_DIM)
    model = models.load_weights(model, WEIGHT_PATH)
    model = model.to(DEVICE)

    # Loss function and optimizer
    criterion = SIMOLoss()
    optimizer = Optimizer.get_optimizer(OPTIMIZER, model.parameters(), LR)
    scheduler = Scheduler.get_scheduler(SCHEDULER, optimizer)

    log.print_log(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    log.print_log(f"Training on device: {DEVICE}\n")

    # Training loop
    best_psnr = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train for one epoch
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        log.print_log(f"Epoch [{epoch}/{NUM_EPOCHS}] - Loss: {avg_loss:.6f}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        log.print_log(f"Learning Rate: {current_lr:.6e}")

        # Run testing
        avg_psnr, avg_ssim = test(model, test_loader, DEVICE)
        log.print_log(f"Test PSNR: {avg_psnr:.5f} dB, SSIM: {avg_ssim:.5f}")

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_model_path = f"{OUTPUT_DIR}/mlwnet_best.pth"
            torch.save({
                'epoch': epoch,
                'params': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }, best_model_path)
            log.print_log(f"Best model saved with PSNR: {best_psnr:.5f} dB\n")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    log.print_log(f"Training completed in {elapsed_time/60:.3f} minutes.\n")

    # Log results
    log.print_log(f"End Time: {time.ctime(end_time)}")
    log.print_log(f"Total Elapsed Time: {elapsed_time/60:.3f} minutes")
    log.print_log(f"Best PSNR: {best_psnr:.5f} dB")
