# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: augmentation.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader


OUTPUT_DIR = "./outputs"


class RealBlurAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            # ----------------- Geometry Transforms ----------------- #
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.SafeRotate(limit=90, p=0.5),
            # A.RandomResizedCrop(
            #     size=(img_size[0], img_size[1]),
            #     scale=(0.9, 1.0), ratio=(2.0, 2.0), p=0.5
            # ),

            # # ----------------- Color Transforms ----------------- #
            # A.RandomBrightnessContrast(p=0.5),
            # A.HueSaturationValue(p=0.5),

            # # ----------------- Noise Transforms ----------------- #
            # A.GaussNoise(std_range=(0.05, 0.2), p=0.3)
        ], additional_targets={"gt": "image"})

    def __call__(self, image: np.ndarray, gt: np.ndarray):
        augmented = self.transform(image=image, gt=gt)
        return augmented["image"], augmented["gt"]


def show_transformed_samples(
    dataloader: DataLoader,
    num_samples: int=8,
    output_dir: str=OUTPUT_DIR
    ):
    """
    Plot `num_samples` images with their corresponding ground-truth images
    after applying augmentation.
    """
    assert num_samples % 2 == 0, "num_samples should be even."
    fig, axes = plt.subplots(num_samples // 2, 4, figsize=(9, int(num_samples / 3 * 2)))

    for i in tqdm(range(num_samples), desc="Plotting augmented samples"):
        img, gt = next(iter(dataloader))
        img, gt = img[0], gt[0]

        # Convert torch tensor to numpy array for plotting
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = img

        gt_np = gt.permute(1, 2, 0).numpy() if isinstance(gt, torch.Tensor) else gt

        # Clip image to [0, 1]
        img_np = img_np.clip(0, 1)

        axes[i // 2, (i % 2) * 2].imshow(img_np)
        axes[i // 2, (i % 2) * 2].set_title(f"Image {i+1:02d}", fontsize=10)
        axes[i // 2, (i % 2) * 2].axis("off")

        axes[i // 2, (i % 2) * 2 + 1].imshow(gt_np, cmap='tab20')
        axes[i // 2, (i % 2) * 2 + 1].set_title(f"GT {i+1:02d}", fontsize=10)
        axes[i // 2, (i % 2) * 2 + 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "augmented_samples.png"), dpi=300)


if __name__ == '__main__':
    print("[INFO] Testing Augmentation module...")
    from dataset import RealBlurDataset
    
    # Data augmentation
    transforms = RealBlurAugmentation()

    # Datasets
    TrainingDataset = RealBlurDataset("train", "J", (512, 512), transform=transforms)
    TrainingLoader  = DataLoader(TrainingDataset, batch_size=4, shuffle=True, num_workers=4)

    # Show transformed samples
    show_transformed_samples(TrainingLoader, num_samples=16)

    print("[INFO] Augmentation module test finished.")
