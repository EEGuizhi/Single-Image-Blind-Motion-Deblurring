# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: misc.py
Author: BSChen
Description:
    NYCU IEE Deep Learning Final Project - Single Image Blind Motion Deblurring
    Group 25: 313510156, 313510217
"""


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
