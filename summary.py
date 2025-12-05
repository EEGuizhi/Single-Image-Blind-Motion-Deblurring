# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: network_info.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchinfo

import models
from configs.config import *
from utils.misc import *


if __name__ == "__main__":
    # Initialization
    SUMMARY_PATH = (
        OUTPUT_DIR
        + f"/summary"
        + f"_{SUMMARY_CONFIG['model_name']}"
        + f"_d{SUMMARY_CONFIG['model_dim']}"
        + ".txt"
    )
    log = logger(SUMMARY_PATH)
    img_size = SUMMARY_CONFIG["patch_size"]

    # Print configuration
    log.print_log(f">> Summary Configuration:")
    for key, value in SUMMARY_CONFIG.items():
        log.print_log(f"    - {key}: {value}")
    log.print_log("")

    # Load model
    model = models.load_model(
        SUMMARY_CONFIG["model_name"],
        dim=SUMMARY_CONFIG["model_dim"]
    ).to(DEVICE)

    # Print model summary
    summary = torchinfo.summary(
        model,
        input_data=(torch.randn(1, 3, img_size[0], img_size[1]).to(DEVICE)),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
    )
    log.print_log(f"{summary}")
    log.print_log("")
    log.print_log(f">> Summary saved to {SUMMARY_PATH}")
