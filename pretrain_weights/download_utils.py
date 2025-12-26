# BSChen
import os
import gdown

ROOT = "./pretrain_weights"

# ------------------------------------ Select one of the following models to download ------------------------------------ #

# # GoPro width 64
# url = "https://drive.usercontent.google.com/open?id=15SjPtkfQ0m6NuVss0rwvscatKwRZ2l4i&authuser=0"
# file_name = "gopro-width64.pth"

# # RealBlur-J width 64
# url = "https://drive.usercontent.google.com/open?id=1lv3MCJgZUWgITUFNvgY8nU9YlpgE050W&authuser=0"
# file_name = "realblur_j-width64.pth"

# # RealBlur-J width 32
url = "https://drive.usercontent.google.com/open?id=1WYhczcWj9vPn-C3fCfLk5Tq0GkD3ylY3&authuser=0"
file_name = "realblur_j-width32.pth"

# RealBlur-R width 64
# url = "https://drive.usercontent.google.com/open?id=1hp4Qu2n_lOmc7LsSOHYjf4NhUzZtskGP&authuser=0"
# file_name = "realblur_r-width64.pth"

# ------------------------------------------------------------------------------------------------------------------------- #


def download_from_google_drive(url: str, file_name: str):
    url = url.replace(".usercontent", "").replace("open?id=", "uc?id=").replace("&authuser=0", "")
    url = url.replace("/view?usp=drive_link", "").replace("file/d/", "uc?id=")
    output = file_name
    gdown.download(url, output, quiet=False)


if __name__ == "__main__":
    # Change execution directory to project root
    curr_root = os.getcwd()
    if "pretrain_weights" not in curr_root:
        os.chdir(ROOT)
        print(f">> Change working directory to {ROOT}")

    print(f">> Downloading {file_name} from Google Drive...")
    download_from_google_drive(url, file_name)
    print(">> Done.")
