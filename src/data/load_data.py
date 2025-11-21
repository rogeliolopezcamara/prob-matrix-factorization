# src/data/load_data.py

import pandas as pd
import os

DATA_DIR = "data/raw"


def load_interactions(split):
    """
    Load interaction data for a given split: 'train', 'validation', or 'test'.
    Returns only columns ['u', 'i', 'rating'].
    """
    filename = f"interactions_{split}.csv"
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    return df[["u", "i", "rating"]]


def load_all_splits():
    """Loads raw (uncentered) train, val, test."""
    train = load_interactions("train")
    val   = load_interactions("validation")
    test  = load_interactions("test")
    return train, val, test


def load_all_splits_centered():
    """
    Loads train/val/test and returns:
        train_centered, val_centered, test_centered, global_mean
    """
    train, val, test = load_all_splits()

    # Compute global mean ONLY from training set
    global_mean = train["rating"].mean()

    # Center ratings
    train_c = train.copy()
    val_c   = val.copy()
    test_c  = test.copy()

    train_c["rating"] = train_c["rating"] - global_mean
    val_c["rating"]   = val_c["rating"]   - global_mean
    test_c["rating"]  = test_c["rating"]  - global_mean

    return train_c, val_c, test_c, global_mean