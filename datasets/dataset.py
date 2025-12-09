# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: dataset.py
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
from collections import OrderedDict

import torch
import torch.functional as F
import albumentations as A
from torch.utils.data import Dataset, DataLoader


DATASET_ROOT = "./datasets/realblur"


def read_json_list(json_file: str) -> list[dict]:
    """Read image information from a JSON file."""
    with open(json_file, 'r') as f:
        json_list = json.load(f)
    return json_list


class RealBlurDataset(Dataset):
    """RealBlur Dataset for Blind Motion Image Deblurring Task
    Args:
        split (str): 'train', 'val' or 'test'.
        img_type (str): 'J' for RealBlur-J, 'R' for RealBlur-R. Default: 'J'.
        orig_size (bool): Whether to use original images size. Default: False.
        img_size (tuple[int, int]): Image size (H, W). Default: (256, 256).
        overlap (tuple[int, int]): Overlap size (H_overlap, W_overlap). Default: (0, 0).
        rand_crop (bool): Whether to use random cropping. Default: False.
        root (str): Root directory of dataset.
        transform: Transformations to be applied on images. Default: None.
    """
    def __init__(
        self,
        split:      str,
        img_type:   str             = 'J',
        orig_size:  bool            = False,
        img_size:   tuple[int, int] = (256, 256),
        overlap:    tuple[int, int] = (0, 0),
        rand_crop:  bool            = False,
        root:       str             = DATASET_ROOT,
        transform:  A.Compose       = None,
        cache_size: int             = 0
        ) -> None:
        print(f"[Dataset] Initializing RealBlurDataset with split='{split}', img_type='{img_type}'")
        assert split in ["train", "test"], "Split must be 'train', or 'test'"
        assert img_type in ['J', 'R'], "Dataset type must be 'J' or 'R'"
        if img_type == 'J':
            data_list_file = f"{root}/RealBlur_J_{split}_list.json"
        else:
            data_list_file = f"{root}/RealBlur_R_{split}_list.json"
        assert os.path.exists(data_list_file), f"Data list file {data_list_file} does not exist."

        # Initialize attributes
        self.root       = root
        self.split      = split
        self.img_type   = img_type
        self.orig_size  = orig_size
        self.img_size   = img_size  if not orig_size else None
        self.rand_crop  = rand_crop if not orig_size else False
        self.overlap    = overlap   if not orig_size else (0, 0)
        self.step_size  = (
            img_size[0] - overlap[0],
            img_size[1] - overlap[1]
        ) if not orig_size else None
        self.transform  = transform
        self.cache_size = cache_size if cache_size and cache_size > 0 else 0

        # LRU cache: key = img_idx, value = (blur_img, gt_img)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        # Create image path lists
        print(f"Reading image paths from {data_list_file} ...")
        self.blur_image_list = []
        self.gt_image_list   = []
        json_list = read_json_list(data_list_file)

        for idx, data in tqdm(enumerate(json_list), desc="Reading image info"):
            # Get image paths
            blur_image_path = data["blur_image"]
            gt_image_path   = data["gt_image"]

            h, w = data["height"], data["width"]
            if self.orig_size:
                # Use original image size
                # (path, (h_idx, w_idx), image_idx, n_patches)
                self.blur_image_list.append((blur_image_path, [0, 0], idx, 1))
                self.gt_image_list.append((gt_image_path, [0, 0], idx, 1))
            else:
                # Split patch indexes
                h_step, w_step = self.step_size
                h_indices = list(range(0, h - self.img_size[0] + 1, h_step))
                w_indices = list(range(0, w - self.img_size[1] + 1, w_step))
                if h_indices[-1] + self.img_size[0] < h:
                    h_indices.append(h - self.img_size[0])
                if w_indices[-1] + self.img_size[1] < w:
                    w_indices.append(w - self.img_size[1])

                # Append patch image paths
                n_patches = len(h_indices) * len(w_indices)
                for h_idx in h_indices:
                    for w_idx in w_indices:
                        # (path, (h_idx, w_idx), image_idx, n_patches)
                        self.blur_image_list.append((blur_image_path, [h_idx, w_idx], idx, n_patches))
                        self.gt_image_list.append((gt_image_path, [h_idx, w_idx], idx, n_patches))

        # Finish initialization
        print(f"[Dataset] RealBlurDataset initialized with {len(self.blur_image_list)} samples.\n")

    def _load_image_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load a pair of blur and ground-truth images with caching."""
        blur_path, _, img_idx, _ = self.blur_image_list[index]
        gt_path   = self.gt_image_list[index][0]

        # If cache is enabled and the image is in cache, use it directly
        if self.cache_size > 0 and img_idx in self._cache:
            blur_img, gt_img = self._cache[img_idx]
            # LRU: Move this key to the end to indicate recent use
            self._cache.move_to_end(img_idx)
            return blur_img, gt_img

        # Otherwise, load from disk
        blur_img = cv2.imread(blur_path, cv2.IMREAD_COLOR)
        gt_img   = cv2.imread(gt_path,   cv2.IMREAD_COLOR)

        # If cache is enabled, add to cache and evict oldest if necessary
        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                # pop the first item (least recently used)
                self._cache.popitem(last=False)
            self._cache[img_idx] = (blur_img, gt_img)

        return blur_img, gt_img

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Load blur and ground-truth images
        blur_img, gt_img = self._load_image_pair(index)
        blur_img  = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        gt_img    = cv2.cvtColor(gt_img,   cv2.COLOR_BGR2RGB)
        img_idx   = self.blur_image_list[index][2]
        n_patches = self.blur_image_list[index][3]

        if self.orig_size:
            # Use original image size
            h_idx, w_idx = 0, 0
        else:
            h, w = blur_img.shape[0], blur_img.shape[1]

            # Crop image patch
            h_idx, w_idx = self.blur_image_list[index][1]
            if self.rand_crop:
                h_bias = np.random.randint(-self.step_size[0] // 2, self.step_size[0] // 2 + 1)
                w_bias = np.random.randint(-self.step_size[1] // 2, self.step_size[1] // 2 + 1)
                h_idx += h_bias
                w_idx += w_bias
                h_idx = np.clip(h_idx, 0, h - self.img_size[0])
                w_idx = np.clip(w_idx, 0, w - self.img_size[1])
            blur_img = blur_img[h_idx:h_idx+self.img_size[0], w_idx:w_idx+self.img_size[1], :]
            gt_img   = gt_img[h_idx:h_idx+self.img_size[0], w_idx:w_idx+self.img_size[1], :]

        # Apply transformations
        if self.transform is not None:
            blur_img, gt_img = self.transform(blur_img, gt_img)

        # Normalize & To Tensor
        blur_img = torch.from_numpy(blur_img).permute(2, 0, 1).float() / 255.0
        gt_img   = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0
        return blur_img, gt_img, (h_idx, w_idx), img_idx, n_patches

    def __len__(self):
        return len(self.blur_image_list)


def custom_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Custom collate function to handle variable-length data in DataLoader."""
    blur_imgs = torch.stack([item[0] for item in batch], dim=0)
    gt_imgs   = torch.stack([item[1] for item in batch], dim=0)
    start_pos = torch.stack([torch.tensor(item[2], dtype=torch.long) for item in batch], dim=0)
    img_idxs  = torch.tensor([item[3] for item in batch], dtype=torch.long)
    n_patches = batch[0][4]  # All items have the same n_patches
    return blur_imgs, gt_imgs, start_pos, img_idxs, n_patches
