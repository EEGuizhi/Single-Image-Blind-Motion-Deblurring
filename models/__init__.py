# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: __init__.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import torch
import torch.nn as nn

from .MLWNet import MLWNet_Local
from .network import Network


__all__ = ['MLWNet_Local', 'Network']


def load_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to load a model by name.
    Args:
        model_name (str): Name of the model to load.
        **kwargs: Additional keyword arguments for the model constructor.
    Returns:
        nn.Module: Instantiated model.
    """
    if model_name == 'MLWNet_Local':
        dim = kwargs.get('model_dim', 32)
        return MLWNet_Local(dim=dim)
    elif model_name == 'Network':
        dim = kwargs.get('model_dim', 32)
        expand_dim = kwargs.get('expand_dim', 2)
        aux_heads = kwargs.get('aux_heads', True)
        return Network(dim=dim, expand_dim=expand_dim, aux_heads=aux_heads)
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")


def load_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """Load model weights from a specified path.
    Args:
        model (nn.Module): The model architecture.
        weights_path (str): Path to the model weights file.
    Returns:
        nn.Module: The model with loaded weights.
    """
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict["params"], strict=False)
    return model
