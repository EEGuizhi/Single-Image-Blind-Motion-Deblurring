# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: misc.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""

import os

class logger:
    """Simple logger to log messages to both console and a file.
    Args:
        log_file (str): Path to the log file.
    """
    def __init__(self, log_file: str, clear: bool = True) -> None:
        self.log_file = log_file

        # Clear the log file
        if clear:
            with open(self.log_file, 'w') as f:
                f.write("")

    def print_log(self, *message: str) -> None:
        """Log a message to both console and the log file.
        Args:
            message (str): The message to log.
        """
        print(*message)
        with open(self.log_file, 'a') as f:
            print(*message, file=f)


class csv_logger:
    """Simple CSV logger to log data in CSV format.
    Args:
        csv_file (str): Path to the CSV file.
    """
    def __init__(self, csv_file: str, dict_keys: list = None, clear: bool = True) -> None:
        self.csv_file = csv_file
        if dict_keys is None:
            self.dict_keys = [
                "Epoch",
                "Train Loss",
                "Val Loss",
                "Train PSNR",
                "Val PSNR",
                "Train SSIM",
                "Val SSIM",
                "Learning Rate"
            ]
        else:
            self.dict_keys = dict_keys

        # Clear the CSV file
        if clear:
            with open(self.csv_file, 'w') as f:
                headers = ','.join(self.dict_keys)
                f.write(headers + '\n')

    def log_epoch(self, epoch_dict: dict, logger_instance: logger) -> None:
        """Log epoch information to the CSV file.
        Args:
            epoch_dict (dict): Dictionary containing epoch information.
            logger_instance (logger): Logger instance for console logging.
        """
        if logger_instance is not None:
            logger_instance.print_log(
                f"Epoch [{epoch_dict['epoch']}/{epoch_dict['num_epochs']}], "
                f"Train Loss: {epoch_dict['train_loss']:.4f}, "
                f"Val Loss: {epoch_dict['val_loss']:.4f}, "
                f"LR: {epoch_dict['learning_rate']:.6f}, "
                f"Val PSNR: {epoch_dict['val_psnr']:.2f}, "
                f"Val SSIM: {epoch_dict['val_ssim']:.4f}"
            )

        with open(self.csv_file, 'a') as f:
            line = ','.join([str(epoch_dict[key.lower().replace(' ', '_')]) for key in self.dict_keys])
            f.write(line + '\n')


def init_directory(dir_path: str) -> None:
    """Create a directory if it does not exist.
    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
