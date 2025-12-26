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
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader


OUTPUT_DIR = "./outputs"


class RealBlurAugmentation:
    # def __init__(self):
    #     self.transform = A.Compose([
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.5),
    #         # A.RandomRotate90(p=1.0)
    #     ], additional_targets={"gt": "image"})

    # def __call__(self, image: np.ndarray, gt: np.ndarray):
    #     augmented = self.transform(image=image, gt=gt)
    #     return augmented["image"], augmented["gt"]

    def __call__(
        self, image, gt,
        hflip=True, rotation=True, flows=None, return_status=False, vflip=False
    ):
        """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

        We use vertical flip and transpose for rotation implementation.
        All the images in the list use the same augmentation.

        Args:
            imgs (list[ndarray] | ndarray): Images to be augmented. If the input
                is an ndarray, it will be transformed to a list.
            hflip (bool): Horizontal flip. Default: True.
            rotation (bool): Ratotation. Default: True.
            flows (list[ndarray]: Flows to be augmented. If the input is an
                ndarray, it will be transformed to a list.
                Dimension is (h, w, 2). Default: None.
            return_status (bool): Return the status of flip and rotation.
                Default: False.

        Returns:
            list[ndarray] | ndarray: Augmented images and flows. If returned
                results only have one element, just return ndarray.
        """
        imgs = [image, gt]
        hflip = hflip and random.random() < 0.5
        if vflip or rotation:
            vflip = random.random() < 0.5
        rot90 = rotation and random.random() < 0.5

        def _augment(img):
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
                if img.shape[2] == 6:
                    img = img[:,:,[3,4,5,0,1,2]].copy() # swap left/right
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        def _augment_flow(flow):
            if hflip:  # horizontal
                cv2.flip(flow, 1, flow)
                flow[:, :, 0] *= -1
            if vflip:  # vertical
                cv2.flip(flow, 0, flow)
                flow[:, :, 1] *= -1
            if rot90:
                flow = flow.transpose(1, 0, 2)
                flow = flow[:, :, [1, 0]]
            return flow

        if not isinstance(imgs, list):
            imgs = [imgs]
        imgs = [_augment(img) for img in imgs]
        if len(imgs) == 1:
            imgs = imgs[0]

        if flows is not None:
            if not isinstance(flows, list):
                flows = [flows]
            flows = [_augment_flow(flow) for flow in flows]
            if len(flows) == 1:
                flows = flows[0]
            return imgs, flows
        else:
            if return_status:
                return imgs, (hflip, vflip, rot90)
            else:
                return imgs


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
