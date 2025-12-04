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

# -------------------------------- Global configuration -------------------------------- #
ROOT_DIR = "/home/bschen/Single-Image-Blind-Motion-Deblurring"
OUTPUT_DIR = f"{ROOT_DIR}/outputs"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------- Dataset Configurations ------------------------------- #
# Path Settings
DATASET_ROOT = f"{ROOT_DIR}/datasets/realblur"
TRAIN_J_FILE = f"{DATASET_ROOT}/RealBlur_J_train_list.txt"
TEST_J_FILE  = f"{DATASET_ROOT}/RealBlur_J_test_list.txt"
TRAIN_R_FILE = f"{DATASET_ROOT}/RealBlur_R_train_list.txt"
TEST_R_FILE  = f"{DATASET_ROOT}/RealBlur_R_test_list.txt"

# Type of dataset
IMG_TYPE = 'J'  # Options: 'J' for RealBlur-J, 'R' for RealBlur-R

# Image Settings
IMG_SIZE   = (512, 512)  # (H, W)
OVERLAP    = (128, 128)  # (H_overlap, W_overlap)
CACHE_SIZE = 10

# ------------------------------- Training Configurations ------------------------------- #
TRAIN_BATCH_SIZE = 2


# ------------------------------- Testing Configurations ------------------------------- #
TEST_BATCH_SIZE = 1  # Must be 1 due to patch-wise testing
MODEL_WEIGHTS_PATH = f"{ROOT_DIR}/pretrain_weights/realblur_j-width32.pth"
TEST_RESULT_LOG = f"{OUTPUT_DIR}/test_results.txt"
SUMMARY_LOG = f"{OUTPUT_DIR}/network_summary.txt"
