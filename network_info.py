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

from models.network import Network
from configs.config import *
from utils.misc import *


if __name__ == "__main__":
    log = logger(SUMMARY_LOG)

    H, W = IMG_SIZE

    model = Network(
        in_channels=3,
        out_channels=3,
        base_dim=32,
        expand_dim=2,
        aux_heads=True
    ).to(DEVICE)

    summary = torchinfo.summary(
        model,
        input_data=(torch.randn(1, 3, H, W).to(DEVICE)),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
    )
    log.print_log(summary)
