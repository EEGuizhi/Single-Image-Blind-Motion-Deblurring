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
import gdown

ROOT = "./datasets/realblur"


# ------------------------------------ Select one of the following models to download ------------------------------------ #

# RealBlur Dataset (tar.gz)
url = "https://drive.google.com/file/d/17v5Dj0M2WExgxJgsGpEWc-6UyhK1IWWK/view?usp=drive_link"
file_name = "RealBlur.tar.gz"

# ------------------------------------------------------------------------------------------------------------------------- #


def download_from_google_drive(url: str, file_name: str):
    url = url.replace(".usercontent", "").replace("open?id=", "uc?id=").replace("&authuser=0", "")
    url = url.replace("/view?usp=drive_link", "").replace("file/d/", "uc?id=")
    output = file_name
    gdown.download(url, output, quiet=False)


if __name__ == "__main__":
    # Change execution directory to project root
    curr_root = os.getcwd()
    if "/datasets/realblur" not in curr_root:
        os.chdir(ROOT)
        print(f">> Change working directory to {ROOT}")

    print(f">> Downloading {file_name} from Google Drive...")
    download_from_google_drive(url, file_name)
    print(">> Done.")
