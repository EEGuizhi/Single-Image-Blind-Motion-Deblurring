# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: config.py
Author: BSChen, JRKang
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
# Select datasets
USE_DATASETS = 'RealBlur'  # Options: 'RealBlur', 'GoPro'

# Path Settings
DATASET_ROOT = f"{ROOT_DIR}/datasets/realblur"
TRAIN_J_FILE = f"{DATASET_ROOT}/RealBlur_J_train_list.txt"
TEST_J_FILE  = f"{DATASET_ROOT}/RealBlur_J_test_list.txt"
TRAIN_R_FILE = f"{DATASET_ROOT}/RealBlur_R_train_list.txt"
TEST_R_FILE  = f"{DATASET_ROOT}/RealBlur_R_test_list.txt"

# Type of dataset
IMG_TYPE = 'J'  # Options: 'J' for RealBlur-J, 'R' for RealBlur-R

# GOPRO Dataset Paths
GOPRO_ROOT = f"{ROOT_DIR}/datasets/gopro"
# GOPRO_TRAIN_FILE = f"{GOPRO_ROOT}/GoPro_train_list.json"
# GOPRO_TEST_FILE = f"{GOPRO_ROOT}/GoPro_test_list.json"

# DataLoader Settings
CACHE_SIZE = 1000


# ------------------------------- Training Configurations ------------------------------- #
EXPERIMENT_DIR = f"{OUTPUT_DIR}/experiments/realblur_exp_09"
TRAIN_CONFIG = {
    'model_name': 'FFTformer',   # Options: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network'
    'model_dim': 48,           # Options: 32, 64
    'patch_size': (256, 256),  # (H, W)
    'overlap': (64, 64),       # (H_overlap, W_overlap)

    'augmentation': True,
    'rand_crop': True,
    'num_epochs': 320,
    'batch_size': 1,
    'accum_iter': 4,
    'learning_rate': 4e-4,

    'simo_loss': False,
    'edge_loss': False,
    'l1_loss': True,
    'mse_loss': False,
    'fft_loss': True,

    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',  # Options: 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
    'metric': 'PSNR',                  # Options: 'PSNR', 'SSIM'

    'checkpoint': f"{EXPERIMENT_DIR}/FFTformer_d48.pth",
    'weight_only': False,
    'num_workers': 8,
}


# ------------------------------- Testing Configurations ------------------------------- #
TEST_CONFIG = {
    # 'model_name': 'MLWNet',       # Options: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network'
    # 'model_dim': 32,               # Options: 32, 64
    # 'weights_path': f"{ROOT_DIR}/pretrain_weights/MLWNet_d32.pth",

    # 'model_name': 'Network',       # Options: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network'
    # 'model_dim': 32,               # Options: 32, 64
    # 'weights_path': f"{ROOT_DIR}/pretrain_weights/Network_d32.pth",

    'model_name': 'FFTformer',     # Options: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network'
    'model_dim': 48,               # Options: 48
    'weights_path': f"{EXPERIMENT_DIR}/FFTformer_d48.pth",

    'batch_size': 1,
    'patch_size': (512, 512),  # (H, W)
    'overlap': (256, 256),     # (H_overlap, W_overlap)

    'ecc_iters': 50,
    'num_workers': 8,
    'show_image_indices': [3, 8, 18],  # Indices of images to save during testing
}
