# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: config.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import os
import torch

# --------------------------------- Basic configuration --------------------------------- #
ROOT_DIR   = "/home/bschen/Single-Image-Blind-Motion-Deblurring"
OUTPUT_DIR = f"{ROOT_DIR}/outputs"
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------- Dataset Configurations ------------------------------- #
# Path Settings
DATASET_ROOT = f"{ROOT_DIR}/datasets/realblur"
TRAIN_J_FILE = f"{DATASET_ROOT}/RealBlur_J_train_list.txt"
TEST_J_FILE  = f"{DATASET_ROOT}/RealBlur_J_test_list.txt"
TRAIN_R_FILE = f"{DATASET_ROOT}/RealBlur_R_train_list.txt"
TEST_R_FILE  = f"{DATASET_ROOT}/RealBlur_R_test_list.txt"

# Type of dataset
IMG_TYPE = 'J'  # Options: 'J' for RealBlur-J, 'R' for RealBlur-R

# DataLoader Settings
CACHE_SIZE = 1000


# ------------------------------- Training Configurations ------------------------------- #
EXPERIMENT_DIR = f"{OUTPUT_DIR}/experiments/realblur_exp_06"
TRAIN_CONFIG = {
    'model_name': 'MLWNet',   # Options: 'MLWNet_Local', 'MLWNet', 'Network'
    'model_dim': 32,           # Options: 32, 64
    'patch_size': (256, 256),  # (H, W)
    'overlap': (64, 64),       # (H_overlap, W_overlap)

    'augmentation': True,
    'rand_crop': True,
    'num_epochs': 1300,
    'batch_size': 8,
    'learning_rate': 1e-3,

    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',  # Options: 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
    'metric': 'SSIM',                  # Options: 'PSNR', 'SSIM', 'CW-SSIM'

    'checkpoint': f"{EXPERIMENT_DIR}/MLWNet_d32.pth",
    'num_workers': 8,
}


# ------------------------------- Testing Configurations ------------------------------- #
TEST_CONFIG = {
    'model_name': 'MLWNet',       # Options: 'MLWNet_Local', 'MLWNet', 'Network'
    'model_dim': 32,               # Options: 32, 64
    # 'weights_path': f"{ROOT_DIR}/pretrain_weights/realblur_j-width32.pth",
    'weights_path': f"{ROOT_DIR}/pretrain_weights/MLWNet_d32.pth",

    'batch_size': 1,
    'patch_size': (512, 512),  # (H, W)
    'overlap': (128, 128),     # (H_overlap, W_overlap)

    'ecc_iters': 50,
    'num_workers': 8,
    'show_image_indices': [3, 8, 18],  # Indices of images to save during testing
}


# --------------------------------- Summary Configuration --------------------------------- #
SUMMARY_CONFIG = {
    'model_name': 'Network',   # Options: 'MLWNet_Local', 'Network'
    'model_dim': 64,           # Options: 32, 64
    'patch_size': (256, 256),  # (H, W)
}
