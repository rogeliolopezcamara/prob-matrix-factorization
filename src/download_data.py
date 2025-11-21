# src/download_data.py

import os
import subprocess

# Name of the Kaggle dataset to download
KAGGLE_DATASET = "shuyangli94/food-com-recipes-and-user-interactions"

# Directory where the raw data will be stored
DATA_DIR = "data/raw"


def ensure_data_dir():
    """
    Create the data directory if it does not exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def download_dataset():
    """
    Download the Kaggle dataset using the Kaggle CLI.
    """
    ensure_data_dir()

    print(f"Downloading dataset '{KAGGLE_DATASET}' into '{DATA_DIR}'...")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        DATA_DIR,
        "--force",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("Download completed.")

    except subprocess.CalledProcessError as e:
        print("Error downloading dataset:", e)
        print("Make sure your Kaggle API is correctly set up in ~/.kaggle/kaggle.json")


if __name__ == "__main__":
    download_dataset()