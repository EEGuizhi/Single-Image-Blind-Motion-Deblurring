# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: preprocess.py
Author: BSChen, JRKang
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
from configs.config import *


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


def prepare_gopro(
    root: str = GOPRO_ROOT,
    ) -> None:
    """Prepare images information for GoPro Dataset
    
    Args:
        root (str): Root directory of GoPro dataset.
    """
    # Process train and test splits
    for split in ['train', 'test']:
        split_path = os.path.join(root, split)
        print(f"[INFO] Preparing GoPro {split} dataset from {split_path} ...")

        # Create image path lists
        json_list = []

        # Iterate through all scene folders
        scene_folders = sorted([
            d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))
        ])

        for scene_folder in tqdm(scene_folders, desc=f"Processing {split} scenes"):
            scene_path = os.path.join(split_path, scene_folder)
            blur_folder = os.path.join(scene_path, 'blur')
            sharp_folder = os.path.join(scene_path, 'sharp')

            # Get all blur images
            if os.path.isdir(blur_folder) and os.path.isdir(sharp_folder):
                blur_images = sorted([
                    f for f in os.listdir(blur_folder) if f.endswith(('.png', '.jpg', '.jpeg'))
                ])

                for blur_img in blur_images:
                    blur_image_path = os.path.join(blur_folder, blur_img)
                    gt_image_path = os.path.join(sharp_folder, blur_img)

                    # Check if corresponding sharp image exists
                    if os.path.exists(gt_image_path):
                        # Read image to get height and width
                        image = cv2.imread(gt_image_path)
                        if image is not None:
                            h, w = image.shape[0], image.shape[1]

                            # Append to json list
                            info_dict = {
                                "blur_image": blur_image_path,
                                "gt_image": gt_image_path,
                                "height": h,
                                "width": w
                            }
                            json_list.append(info_dict)

        # Save to json file
        output_json_path = os.path.join(root, f'GoPro_{split}_list.json')
        with open(output_json_path, 'w') as json_file:
            json.dump(json_list, json_file, indent=4)

        # Finish initialization
        print(f"[INFO] Finish preparing GoPro_{split}_list.json ({len(json_list)} images)\n")


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

    # Prepare GOPRO dataset
    prepare_gopro(root=GOPRO_ROOT)
