# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: optimizer.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Optimizer:
    @staticmethod
    def get_optimizer(optimizer_name: str, parameters, learning_rate: float):
        """Get optimizer by name.
        Args:
            optimizer_name (str): Name of the optimizer ('AdamW', 'SGD', etc.).
            parameters: Model parameters to optimize.
            learning_rate (float): Learning rate for the optimizer.
        Returns:
            torch.optim.Optimizer: Initialized optimizer.
        """
        if optimizer_name == 'AdamW':
            return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
        elif optimizer_name == 'Adam':
            return torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999))
        elif optimizer_name == 'SGD':
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
