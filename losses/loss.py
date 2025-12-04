# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: loss.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIMOLoss(nn.Module):
    """Single Input Multi-Output Loss Function"""
    def __init__(self):
        super(SIMOLoss, self).__init__()

