# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: preprocess.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from configs.config import DATASET_ROOT


def prepare_realblur(
    root: str = DATASET_ROOT,
    list_file: str = "RealBlur_J_train_list.txt"
    ) -> None:
    """Prepare images information for RealBlur Dataset
    
    Args:
        root (str): Root directory of RealBlur dataset.
        list_file (str): File containing list of image paths. (relative to root)
    """
    # Show info
    list_file = os.path.join(root, list_file)
    print(f"[INFO] Preparing RealBlur dataset from {list_file} ...")

    # Create image path lists
    json_list = []
    with open(list_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing images"):
            # Get image paths
            gt_image_path, blur_image_path = line.strip().split()
            gt_image_path = os.path.join(root, gt_image_path)
            blur_image_path = os.path.join(root, blur_image_path)

            # Split patch indexes
            image_shape = cv2.imread(f"{gt_image_path}").shape
            h, w = image_shape[0], image_shape[1]

            # Append to json list            
            info_dict = {
                "blur_image": blur_image_path,
                "gt_image": gt_image_path,
                "height": h,
                "width": w
            }
            json_list.append(info_dict)

    # Save to json file
    output_json_path = os.path.join(root, list_file.replace('.txt', '.json'))
    with open(output_json_path, 'w') as json_file:
        json.dump(json_list, json_file, indent=4)

    # Finish initialization
    print(f"[INFO] Finish preparing {list_file}\n")


if __name__ == '__main__':
    # Determine data list file
    train_J_file = "RealBlur_J_train_list.txt"
    test_J_file  = "RealBlur_J_test_list.txt"
    train_R_file = "RealBlur_R_train_list.txt"
    test_R_file  = "RealBlur_R_test_list.txt"

    # Prepare RealBlur dataset
    prepare_realblur(root=DATASET_ROOT, list_file=train_J_file)
    prepare_realblur(root=DATASET_ROOT, list_file=test_J_file)
    prepare_realblur(root=DATASET_ROOT, list_file=train_R_file)
    prepare_realblur(root=DATASET_ROOT, list_file=test_R_file)
