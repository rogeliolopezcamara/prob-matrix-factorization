# src/unzip_data.py

import zipfile
import os

RAW_DIR = "data/raw"

def unzip_files():
    """
    Unzip all .zip files inside data/raw/.
    """
    for file in os.listdir(RAW_DIR):
        if file.endswith(".zip"):
            zip_path = os.path.join(RAW_DIR, file)
            print(f"Unzipping {zip_path}...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)

            print(f"Extracted contents of {file}")

if __name__ == "__main__":
    unzip_files()