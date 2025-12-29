# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: plot.py
Author: JRKang
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import os
import numpy as np
import matplotlib.pyplot as plt

save_dir = os.path.join(os.path.dirname(__file__), 'outputs/figures')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

psnr = {
    "FFTformer": 31.83560,
    "MLWNet":    32.21568,
    "HybridNet": 31.91366,
    "CE-MLWNet": 32.41885
}

GMACs= {
    "FFTformer": 131.44,
    "MLWNet":    27.78,
    "HybridNet": 30.40,
    "CE-MLWNet": 24.95
}

params = {
    "FFTformer": 16_560_474,
    "MLWNet":    24_109_164,
    "HybridNet": 15_626_243,
    "CE-MLWNet": 20_330_531
}

inference_time = {
    "FFTformer": 47.898e-3,
    "MLWNet":    10.843e-3,
    "HybridNet": 12.490e-3,
    "CE-MLWNet": 10.184e-3
}


if __name__ == "__main__":
    # ---------------------------- GMACs x GParams vs PSNR ---------------------------- #
    x = [GMACs[model] * (params[model] / 1e9) for model in psnr.keys()]
    y = list(psnr.values())
    labels = list(psnr.keys())

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    sizes = [250, 250, 250, 250]
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=sizes[i], c=colors[i], marker=markers[i], alpha=0.6)

    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=12)

    plt.xlabel('GMACs × GParams', fontsize=14)
    plt.ylabel('PSNR(dB)', fontsize=14)
    plt.title('Experimental Results', fontsize=16)
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    x_range = [0, 3]
    y_range = [31.5, 32.5]
    plt.xlim(x_range), plt.ylim(y_range)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experimental_results.png'), dpi=300)


    # ---------------------------- Inference Time vs PSNR ---------------------------- #
    x = list(inference_time.values())
    y = list(psnr.values())
    labels = list(psnr.keys())

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    sizes = [250, 250, 250, 250]
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=sizes[i], c=colors[i], marker=markers[i], alpha=0.6)

    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points', fontsize=12)

    plt.xlabel('Inference Time (s)', fontsize=14)
    plt.ylabel('PSNR(dB)', fontsize=14)
    plt.title('Inference Time vs PSNR', fontsize=16)
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    x_range = [0, 60e-3]
    y_range = [31.5, 32.5]
    plt.xlim(x_range), plt.ylim(y_range)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'inference_time.png'), dpi=300)

