# BSChen
import os
import torch

ROOT = "./pretrain_weights"
FILE_PATH = "realblur_j-width32.pth"
LOG_INFO_PATH = FILE_PATH.replace(".pth", "_log.txt")

def print_log(*args, **kwargs):
    print(*args, **kwargs)
    with open(LOG_INFO_PATH, "a") as f:
        print(*args, **kwargs, file=f)

def print_keys(d, prefix=''):
    if isinstance(d, dict):
        for k, v in d.items():
            # print(prefix + str(k), end='')
            print_keys(v, prefix + str(k) + ' ')
    else:
        if isinstance(d, torch.Tensor):
            print_log(f"{prefix:<60}  (shape: {d.shape})")
        else:
            print_log(f"{prefix:<60}  (type: {type(d)})")

if __name__ == "__main__":
    # Change execution directory to project root
    curr_root = os.getcwd()
    if "pretrain_weights" not in curr_root:
        os.chdir(ROOT)
        print(f">> Change working directory to {ROOT}")

    # Initialize log file
    with open(LOG_INFO_PATH, "w") as f:
        f.write("")

    # Load checkpoint
    checkpoint = torch.load(FILE_PATH, map_location="cpu")
    print_log(f">> Keys in checkpoint loaded from {FILE_PATH}:")
    print_keys(checkpoint)
    print_log(">> Done.")
