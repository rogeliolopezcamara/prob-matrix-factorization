# src/data/load_data.py

import pandas as pd
import numpy as np
import os

DATA_DIR = "data/processed"

def preprocess_data():
    """
    Create new train, validation and test set from raw data.
    """
    # get raw data
    train_raw = pd.read_csv("data/raw/interactions_train.csv",usecols=["u", "i", "rating"])
    val_raw = pd.read_csv("data/raw/interactions_validation.csv",usecols=["u", "i", "rating"])
    test_raw = pd.read_csv("data/raw/interactions_test.csv",usecols=["u", "i", "rating"])

    # combine all data
    df = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)

    # keep recipes i with n >= 10 
    item_i = (
        df.value_counts("i")
        .reset_index(name="n")
        .query("n >= 10")
        [["i"]]
    )
    first_filter = df.merge(item_i, on="i", how="inner")

    # keep users with >= 5 interactions
    user_u = (
        first_filter.value_counts("u")
        .reset_index(name="n")
        .query("n >= 5")
        [["u"]]
    )
    second_filter = first_filter.merge(user_u, on="u", how="inner")

    # shuffle ratings within users
    final_df = second_filter.copy()
    
    final_df = (
        final_df.groupby("u")
        .apply(lambda g: g.sample(frac=1,random_state=42).assign(idx=range(len(g))))
        .reset_index(drop=True)
    )

    # total ratings per user
    final_df["total"] = final_df.groupby("u")["u"].transform("size")
    

    # split assignment
    final_df["split"] = np.where(
        final_df["total"] - (final_df["idx"]+1) == 0, "3.test",
        np.where(final_df["total"] - (final_df["idx"]+1) <= 2, "2.val", "1.train")
    )
    
    # new i mapping
    dict_i = (
        final_df[["i"]].drop_duplicates().sort_values("i")
        .reset_index(drop=True)
        .assign(i_new=lambda x: x.index)
    )

    # new u mapping
    dict_u = (
        final_df[["u"]].drop_duplicates().sort_values("u")
        .reset_index(drop=True)
        .assign(u_new=lambda x: x.index)
    )

    # join mappings
    final_df = (
        final_df
        .merge(dict_i, on="i")
        .merge(dict_u, on="u")
        .drop(columns=["i", "u"])
        .rename(columns={"i_new": "i", "u_new": "u"})
    )

    # final splits
    train_df = final_df.query('split == "1.train"').copy()
    val_df   = final_df.query('split == "2.val"').copy()
    test_df  = final_df.query('split == "3.test"').copy()

    # save preprocessed data
    train_df.to_csv(os.path.join(DATA_DIR, "interactions_train.csv"))
    val_df.to_csv(os.path.join(DATA_DIR, "interactions_validation.csv"))
    test_df.to_csv(os.path.join(DATA_DIR, "interactions_test.csv"))
    dict_i.to_csv(os.path.join(DATA_DIR, "dict_i.csv"))
    dict_u.to_csv(os.path.join(DATA_DIR, "dict_u.csv"))


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

if __name__ == "__main__":
    preprocess_data()