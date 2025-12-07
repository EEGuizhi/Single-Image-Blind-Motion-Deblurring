# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: scheduler.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Scheduler:
    @staticmethod
    def get_scheduler(scheduler_name: str, optimizer, **kwargs):
        """Get learning rate scheduler by name.
        Args:
            scheduler_name (str): Name of the scheduler ('StepLR', 'ReduceLROnPlateau', etc.).
            optimizer: Optimizer for which to schedule the learning rate.
            **kwargs: Additional arguments for the scheduler.
        Returns:
            torch.optim.lr_scheduler._LRScheduler: Initialized learning rate scheduler.
        """
        if scheduler_name == 'StepLR':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'ReduceLROnPlateau':
            mode = kwargs.get('mode', 'min')
            factor = kwargs.get('factor', 0.5)
            patience = kwargs.get('patience', 4)
            min_lr = kwargs.get('min_lr', 1e-6)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor,
                patience=patience, min_lr=min_lr
            )
        elif scheduler_name == 'CosineAnnealingLR':
            T_max = kwargs.get('T_max', 50)
            eta_min = kwargs.get('eta_min', 0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
