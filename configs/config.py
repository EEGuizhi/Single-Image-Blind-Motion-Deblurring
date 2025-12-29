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
DATASET_ROOT = f"{ROOT_DIR}/datasets/{'realblur' if USE_DATASETS == 'RealBlur' else 'gopro'}"
TRAIN_J_FILE = f"{DATASET_ROOT}/RealBlur_J_train_list.txt"
TEST_J_FILE  = f"{DATASET_ROOT}/RealBlur_J_test_list.txt"
TRAIN_R_FILE = f"{DATASET_ROOT}/RealBlur_R_train_list.txt"
TEST_R_FILE  = f"{DATASET_ROOT}/RealBlur_R_test_list.txt"

# Type of dataset
IMG_TYPE = 'J'  # Options: 'J' for RealBlur-J, 'R' for RealBlur-R

# DataLoader Settings
CACHE_SIZE = 1000


# ------------------------------- Training Configurations ------------------------------- #
EXPERIMENT_DIR = f"{OUTPUT_DIR}/experiments/realblur_exp_11"
TRAIN_CONFIG = {
    ### Models: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network', 'HybridNet' ###

    'model_name': 'Network',   # Options: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network'
    'model_dim': 32,           # Options: 32, 48, 64
    'patch_size': (256, 256),  # (H, W)
    'overlap': (64, 64),       # (H_overlap, W_overlap)

    'augmentation': True,
    'rand_crop': True,
    'num_epochs': 2000,
    'batch_size': 8,
    'accum_iter': 1,
    'learning_rate': 9e-4,
    'min_lr': 1e-7,

    'simo_loss': True,
    'edge_loss': False,
    'l1_loss': False,
    'mse_loss': False,
    'fft_loss': False,

    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingLR',  # Options: 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
    'metric': 'PSNR',                  # Options: 'PSNR', 'SSIM'

    'val_interval': 10,
    'checkpoint': f"{EXPERIMENT_DIR}/Network_d32.pth",
    'weight_only': True,
    'num_workers': 8,
}


# ------------------------------- Testing Configurations ------------------------------- #
TEST_CONFIG = {
    ### Models: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network', 'HybridNet' ###

    # 'model_name': 'MLWNet',
    # 'model_dim': 32,               # Options: 32, 64
    # 'weights_path': f"{ROOT_DIR}/outputs/experiments/realblur_exp_06/MLWNet_d32.pth",

    'model_name': 'Network',
    'model_dim': 32,               # Options: 32, 64
    # 'weights_path': f"{ROOT_DIR}/pretrain_weights/Network_d32.pth",
    'weights_path': f"{ROOT_DIR}/outputs/experiments/realblur_exp_11/Network_d32.pth",

    # 'model_name': 'FFTformer',
    # 'model_dim': 48,               # Options: 48
    # 'weights_path': f"{ROOT_DIR}/pretrain_weights/FFTformer_d48.pth",

    # 'model_name': 'HybridNet',
    # 'model_dim': 32,               # Options: 32
    # 'weights_path': f"{ROOT_DIR}/pretrain_weights/HybridNet_d32.pth",

    'batch_size': 1,
    'patch_size': (256, 256),  # (H, W)
    'overlap': (128, 128),     # (H_overlap, W_overlap)
    'orig_size': False,        # Use original image size for testing
    'factor': 128,             # Factor to pad image size to be divisible by this number

    'ecc_iters': 50,
    'num_workers': 8,
    'save_outputs': False,
    'show_image_indices': [3, 8, 18],  # Indices of images to save during testing
}


# --------------------------------- Summary Configuration --------------------------------- #
SUMMARY_CONFIG = {
    'model_name': 'Network',   # Options: 'MLWNet_Local', 'FFTformer', 'Network', 'HybridNet'
    'model_dim': 32,           # Options: 32, 48, 64
    'patch_size': (256, 256),  # (H, W)
}


# --------------------------------- Predict Configuration --------------------------------- #
PREDICT_CONFIG = {
    ### Models: 'MLWNet_Local', 'MLWNet', 'FFTformer', 'Network', 'HybridNet' ###
    'model_name': 'Network',
    'model_dim': 32,               # Options: 48
    'weights_path': f"{ROOT_DIR}/pretrain_weights/submit/Network_d32.pth",

    'input_dir': f"{ROOT_DIR}/predict_pictures/blur/",
    'output_dir': f"{ROOT_DIR}/predict_pictures/sharp/",
    
    'predict_single_image': True,
    'input_path': f"{ROOT_DIR}/predict_pictures/blur/test_03.jpg",
    'output_path': f"{ROOT_DIR}/predict_pictures/sharp/sharp_test_03.png",

    # Inference settings
    'patch_size': (512, 512),
    'overlap': (256, 256),
    'batch_size': 1,
    'device': 'cuda',
}
