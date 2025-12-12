import pandas as pd
import numpy as np
import os
import time
from dataclasses import asdict

from src.data.load_data import load_all_splits
from src.experiments.compare_models import load_best_hyperparams
from src.experiments.compare_models import load_best_hyperparams
from src.models.gaussian_mf_cavi_bias import GaussianMFCAVI, GaussianMFCAVIConfig
from src.evaluation.metrics import rmse, macro_mae
from src.utils.mapping import get_recipe_id_map

import argparse

def train_full_gaussian(dataset_mode='train'):
    print(f"=== Training Full Gaussian MF (CAVI) | Mode: {dataset_mode} ===")
    
    # 1. Load Data
    print("Loading data using load_all_splits...")
    train_df, val_df, test_df = load_all_splits()
    
    if dataset_mode == 'train':
        df = train_df[['u', 'i', 'rating']]
    elif dataset_mode == 'train+val':
        print("Concatenating train and validation sets...")
        df = pd.concat([train_df, val_df])[['u', 'i', 'rating']]
    elif dataset_mode == 'full':
        print("Concatenating train, validation, and test sets...")
        df = pd.concat([train_df, val_df, test_df])[['u', 'i', 'rating']]
    else:
        raise ValueError(f"Invalid dataset_mode: {dataset_mode}. Choose from 'train', 'train+val', 'full'.")

    # Preprocessing: Center Data
    global_mean = df["rating"].mean()
    print(f"Centering data (Global Mean = {global_mean:.4f})...")
    df_centered = df.copy()
    df_centered["rating"] -= global_mean
    
    # 2. Configure Model (Best Params)
    print("Loading best hyperparameters...")
    hyperparams = load_best_hyperparams()
    config_dict = hyperparams.get('GaussianMF', {})
    
    if config_dict:
        print(f"Using loaded config: {config_dict}")
        config = GaussianMFCAVIConfig(**config_dict)
    else:
        print("Using default config (fallback)")
        config = GaussianMFCAVIConfig(
            n_factors=20,
            sigma2=0.5,
            eta_theta2=0.1,
            eta_beta2=0.01,
            eta_bias2=0.01,
            max_iter=100,
            tol=1e-8,
            random_state=42,
            verbose=True
        )
    
    model = GaussianMFCAVI(config)
    
    # 3. Train
    print("Starting training...")
    start_time = time.time()
    model.fit(df_centered, global_mean=global_mean)
    print(f"Training finished in {time.time() - start_time:.1f}s")
    
    # 4. Save Embeddings
    output_dir = 'data/embeddings/gaussian_mf'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving embeddings to {output_dir}...")
    
    # User Embeddings
    pd.DataFrame(model.m_theta).to_csv(os.path.join(output_dir, 'user_embeddings.csv'), index=False)
    
    # Item Embeddings with recipe_id
    item_emb_df = pd.DataFrame(model.m_beta)
    
    # Load mapping
    id_map = get_recipe_id_map()
    if id_map is not None:
        # Check alignment
        if len(id_map) > len(item_emb_df):
            # Trim map if model has fewer items (unlikely if trained on full)
            # Or trim to match model size
            id_map = id_map[:len(item_emb_df)]
        elif len(id_map) < len(item_emb_df):
            print(f"Warning: id_map smaller ({len(id_map)}) than item embeddings ({len(item_emb_df)})")
            # Pad or ignore?
        
        # Insert at 0
        # Check length match again
        if len(id_map) == len(item_emb_df):
            item_emb_df.insert(0, 'recipe_id', id_map)
        else:
            print("Skipping recipe_id insertion due to size mismatch.")
    
    item_emb_df.to_csv(os.path.join(output_dir, 'item_embeddings.csv'), index=False)
    
    # Save config
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(str(asdict(config)))
        f.write(f"\nglobal_mean: {global_mean}")

    # 5. Save Test Predictions
    print("Generating predictions on Test Set...")
    pred_dir = 'data/predictions/gaussian_mf'
    os.makedirs(pred_dir, exist_ok=True)

    # Filter test users/items that are in the model range
    # (The model handles this in predict but we need valid indices for the DF)
    test_u = test_df["u"].to_numpy()
    test_i = test_df["i"].to_numpy()
    y_true = test_df["rating"].to_numpy()
    
    # Predict
    y_pred = model.predict(test_u, test_i, global_mean=global_mean)
    
    # Compute Metrics
    test_macro_mae = macro_mae(y_true, y_pred)
    test_rmse = rmse(y_true, y_pred)
    print(f"Test Set Metrics: MacroMAE={test_macro_mae:.4f} | RMSE={test_rmse:.4f}")
    
    preds_df = pd.DataFrame({
        'u': test_u,
        'i': test_i,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    preds_df.to_csv(os.path.join(pred_dir, 'test_predictions.csv'), index=False)
    print(f"Saved test predictions to {pred_dir}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Gaussian MF')
    parser.add_argument('--dataset_mode', type=str, default='train', 
                        choices=['train', 'train+val', 'full'],
                        help='Which dataset splits to use for training')
    args = parser.parse_args()
    
    train_full_gaussian(dataset_mode=args.dataset_mode)
