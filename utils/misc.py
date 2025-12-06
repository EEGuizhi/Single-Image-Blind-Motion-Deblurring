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

    def print_log(self, *message: str, end: str = '\n') -> None:
        """Log a message to both console and the log file.
        Args:
            message (str): The message to log.
        """
        print(*message, end=end)
        with open(self.log_file, 'a') as f:
            print(*message, file=f, end=end)


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
                "Val PSNR",
                "Val SSIM",
                "Learning Rate",
                "Epoch Time"
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
        # Formulate
        epoch_dict = {k.lower().replace(' ', '_'): v for k, v in epoch_dict.items()}

        # Log to console
        if logger_instance is not None:
            logger_instance.print_log(
                f"Epoch [{epoch_dict['epoch']:3d}/{epoch_dict['num_epochs']:3d}]", end=' '
            )
            for key in self.dict_keys:
                k = key.lower().replace(' ', '_')
                if k in epoch_dict and key not in ["Epoch", "Num Epochs"]:
                    logger_instance.print_log(f"| {key}: {epoch_dict[k]:.6f}", end=' ')
            logger_instance.print_log("")

        # Log to CSV file
        with open(self.csv_file, 'a') as f:
            line = ','.join([str(epoch_dict[key.lower().replace(' ', '_')]) for key in self.dict_keys])
            f.write(line + '\n')


if __name__ == "__main__":
    # Test logger
    log = logger("test_log.txt", clear=True)
    log.print_log("This is a test log message.")

    # Test CSV logger
    csv_log = csv_logger("test_log.csv", clear=True)
    epoch_info = {
        "Epoch": 1,
        "Num Epochs": 10,
        "Train Loss": 0.123456,
        "Val Loss": 0.234567,
        "Val PSNR": 30.123456,
        "Val SSIM": 0.912345,
        "Learning Rate": 0.001,
        "Epoch Time": 120.5
    }
    csv_log.log_epoch(epoch_info, log)
