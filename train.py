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
from metrics import *
from utils.misc import *
from utils.checkpoint import *


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: optim.lr_scheduler._LRScheduler = None,
    accumulated_iter: int = 0
) -> dict:
    """Training loop for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on (CPU or GPU).
        scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
    Returns:
        dict: Dictionary containing information about the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    with tqdm(train_loader, desc=f"Training", unit="batch") as pbar:
        for batch_data in pbar:
            # Prepare data
            inputs = batch_data[0].to(device)
            targets = batch_data[1].to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            if hasattr(model, 'get_wavelet_loss'):
                loss += model.get_wavelet_loss()
            pbar.set_postfix({"loss": loss.item()})

            # Backward pass and optimize
            loss = loss / ACCUM_ITER
            loss.backward()
            accumulated_iter += 1

            if accumulated_iter >= ACCUM_ITER:
                accumulated_iter = 0
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches * ACCUM_ITER
    return {"train_loss": avg_loss}, accumulated_iter


def val_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validation loop for one epoch.
    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run validation on (CPU or GPU).
    Returns:
        dict: Dictionary containing information about the epoch.
    """
    model.eval()
    total_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        with tqdm(val_loader, desc=f"Validating", unit="batch") as pbar:
            for batch_data in pbar:
                # Prepare data
                inputs = batch_data[0].to(device)
                targets = batch_data[1].to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)
                pbar.set_postfix({"loss": loss.item()})

                # Compute PSNR and SSIM
                mo = isinstance(outputs, (list, tuple))
                psnr = psnr_torch(outputs[0] if mo else outputs, targets)
                ssim = ssim_torch(outputs[0] if mo else outputs, targets)

                # Accumulate metrics
                total_loss += loss.item()
                if hasattr(model, 'get_wavelet_loss'):
                    total_loss += model.get_wavelet_loss().item()
                running_psnr += psnr
                running_ssim += ssim
                num_batches += 1

    avg_loss = total_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches
    return {"val_loss": avg_loss, "val_psnr": avg_psnr, "val_ssim": avg_ssim}


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
        overlap=(0, 0), root=DATASET_ROOT, cache_size=CACHE_SIZE
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
    IMG_SIZE    = TRAIN_CONFIG["patch_size"]
    OVERLAP     = TRAIN_CONFIG["overlap"]
    USE_AUGMENT = TRAIN_CONFIG["augmentation"]
    RAND_CROP   = TRAIN_CONFIG["rand_crop"]
    NUM_EPOCHS  = TRAIN_CONFIG["num_epochs"]
    BATCH_SIZE  = TRAIN_CONFIG["batch_size"]
    ACCUM_ITER  = TRAIN_CONFIG["accum_iter"]
    LR          = TRAIN_CONFIG["learning_rate"]
    OPTIMIZER   = TRAIN_CONFIG["optimizer"]
    SCHEDULER   = TRAIN_CONFIG["scheduler"]
    METRIC      = TRAIN_CONFIG["metric"]
    VAL_EPOCHS  = TRAIN_CONFIG["val_interval"]
    CHECKPOINT  = TRAIN_CONFIG["checkpoint"]
    WGT_ONLY    = TRAIN_CONFIG["weight_only"]
    NUM_WORKERS = TRAIN_CONFIG["num_workers"]

    # Directory setup
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    # Check checkpoint
    continue_training = False
    if CHECKPOINT is not None and os.path.isfile(CHECKPOINT):
        print(f">> Detected existing checkpoint at: {CHECKPOINT}")
        continue_training = True

    # Logger setup
    LOG_PATH = (
        EXPERIMENT_DIR
        + f"/train_log"
        + f"_{MODEL_NAME}"
        + f"_d{MODEL_DIM}"
        + f"_{IMG_SIZE[0]}_{OVERLAP[0]}"
        + ".txt"
    )
    log = logger(LOG_PATH, clear=not continue_training)
    csv_log = csv_logger(LOG_PATH.replace(".txt", ".csv"), clear=not continue_training)
    if continue_training:
        log.print_log("\n# " + '-'*33 + " Resuming Training " + '-'*33 + " #\n")
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
    iter_per_epoch = len(train_loader)

    # Load model
    model = models.load_model(MODEL_NAME, dim=MODEL_DIM)
    model = model.to(DEVICE)
    log.print_log(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    log.print_log(f"Training on device: {DEVICE}\n")

    # Loss function and optimizer
    criterion = CustomLoss(TRAIN_CONFIG)
    optimizer = Optimizer.get_optimizer(OPTIMIZER, model.parameters(), LR)
    scheduler = Scheduler.get_scheduler(
        SCHEDULER, optimizer, mode='max',  # PSNR is to be maximized
        T_max=NUM_EPOCHS * iter_per_epoch // ACCUM_ITER,
    )

    # Load checkpoint if provided
    if continue_training:
        start_epoch, best_eval = load_checkpoint(
            CHECKPOINT, model, optimizer, scheduler, DEVICE, weight_only=WGT_ONLY
        )
        log.print_log(f"Resumed training from checkpoint at epoch {start_epoch}\n")
        start_epoch += 1  # Start from the next epoch
    else:
        start_epoch = 1
        best_eval = float('-inf')
        log.print_log("No checkpoint provided, starting training from scratch.\n")

    # ---------------------------------- Training loop ---------------------------------- #
    accumulated_iter = 0
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        start_epoch_time = time.time()
        epoch_dict = {"epoch": epoch, "num_epochs": NUM_EPOCHS, 'best_eval': best_eval}

        # Train for one epoch
        train_dict, accumulated_iter = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            scheduler if SCHEDULER != 'ReduceLROnPlateau' else None,
            accumulated_iter
        )
        epoch_dict.update(train_dict)

        # Validate for one epoch
        if epoch % VAL_EPOCHS == 0 or epoch == NUM_EPOCHS or "val_dict" not in locals():
            # Update val_dict
            val_dict = val_epoch(model, test_loader, criterion, DEVICE)
        else:
            pass
        epoch_dict.update(val_dict)

        if METRIC == 'PSNR':
            val_eval = val_dict["val_psnr"]
        elif METRIC == 'SSIM':
            val_eval = val_dict["val_ssim"]
        else:
            raise ValueError(f"Unsupported metric: {METRIC}")

        # Update learning rate
        if SCHEDULER == 'ReduceLROnPlateau':
            scheduler.step(val_eval)

        # Log epoch results
        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        epoch_dict["epoch_time"] = epoch_time
        epoch_dict["learning_rate"] = optimizer.param_groups[0]['lr']
        csv_log.log_epoch(epoch_dict, log)

        # Save best model
        save_checkpoint(CHECKPOINT, epoch, best_eval, model, optimizer, scheduler)
        if val_eval > best_eval:
            best_eval = val_eval
            save_checkpoint(
                CHECKPOINT.replace(".pth", f"_best.pth"),
                epoch, best_eval, model, optimizer, scheduler
            )
            log.print_log(f">> Best model saved with {METRIC}: {best_eval:.5f}\n")
    # ----------------------------------------------------------------------------------- #

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    log.print_log(f"Training completed in {elapsed_time/60:.3f} minutes.\n")

    # Log results
    log.print_log(f"End Time: {time.ctime(end_time)}")
    log.print_log(f"Total Elapsed Time: {elapsed_time/60:.3f} minutes")
    log.print_log(f"Best {METRIC}: {best_eval:.5f}{' dB' if METRIC == 'PSNR' else ''}")
