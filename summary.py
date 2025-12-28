# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: network_info.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

import models
from configs.config import *
from utils.misc import *

import time
import torch
import torch.nn as nn
import numpy as np


def test_inference_time(
    model: nn.Module,
    img_size: tuple[int, int],
    device: torch.device,
    iterations: int = 100,
    warmup: int = 20,
) -> float:
    """Measure the average inference time of the model (ms).
    Uses CUDA events if device is GPU, otherwise uses perf_counter for CPU.
    """
    # Initialization
    model.eval()
    model.to(device)
    times = []
    input_tensor = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # GPU timing
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(iterations):
                starter.record()
                _ = model(input_tensor)
                ender.record()

                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))  # milliseconds
    # CPU timing
    else:
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # milliseconds
    return float(np.mean(times))



if __name__ == "__main__":
    # Initialization
    SUMMARY_PATH = (
        f"{OUTPUT_DIR}/summaries"
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
        dim=SUMMARY_CONFIG["model_dim"],
        aux_heads=False,
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

    # Measure inference time
    avg_inference_time = test_inference_time(
        model, img_size, DEVICE, iterations=100
    )
    log.print_log(f">> Average Inference Time over 100 iterations: {avg_inference_time:.3f} ms")
    log.print_log("")

    log.print_log(f">> Summary saved to {SUMMARY_PATH}")
