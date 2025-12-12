import pandas as pd
import numpy as np
import os
import time
from dataclasses import asdict

from src.data.load_data import load_all_splits
from src.experiments.compare_models import load_best_hyperparams
from src.models.poisson_mf_cavi import PoissonMFCAVI, PoissonMFCAVIConfig
from src.evaluation.metrics import rmse, macro_mae
from src.utils.mapping import get_recipe_id_map

import argparse

def train_full_poisson(dataset_mode='train'):
    print(f"=== Training Full Poisson MF (CAVI) | Mode: {dataset_mode} ===")
    
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
    
    # 2. Configure Model (Best Params)
    print("Loading best hyperparameters...")
    hyperparams = load_best_hyperparams()
    config_dict = hyperparams.get('PoissonMF', {})
    
    if config_dict:
        print(f"Using loaded config: {config_dict}")
        config = PoissonMFCAVIConfig(**config_dict)
    else:
        print("Using default config (fallback)")
        config = PoissonMFCAVIConfig(
            n_factors=100,
            a0=0.1,
            b0=1.0,
            max_iter=100,
            tol=1e-4,
            random_state=42,
            verbose=True
        )
    
    model = PoissonMFCAVI(config)
    
    # 3. Train
    print("Starting training...")
    start_time = time.time()
    model.fit(df) # Trains on everything loaded
    print(f"Training finished in {time.time() - start_time:.1f}s")
    
    # 4. Save Embeddings
    output_dir = 'data/embeddings/poisson_mf'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving embeddings to {output_dir}...")
    # Expected value of theta = gamma_u / rho_u
    # Expected value of beta = lambda_i / eta_i
    if hasattr(model, 'E_theta'):
        user_emb = model.E_theta
    else:
        user_emb = model.gamma_u / model.rho_u
        
    if hasattr(model, 'E_beta'):
        item_emb = model.E_beta
    else:
        item_emb = model.lambda_i / model.eta_i
        
    pd.DataFrame(user_emb).to_csv(os.path.join(output_dir, 'user_embeddings.csv'), index=False)
    
    item_emb_df = pd.DataFrame(item_emb)
    
    # Load mapping
    id_map = get_recipe_id_map()
    if id_map is not None:
        if len(id_map) > len(item_emb_df):
            id_map = id_map[:len(item_emb_df)]
        
        if len(id_map) == len(item_emb_df):
            item_emb_df.insert(0, 'recipe_id', id_map)
        else:
            print("Skipping recipe_id insertion due to size mismatch.")
            
    item_emb_df.to_csv(os.path.join(output_dir, 'item_embeddings.csv'), index=False)
    
    # Save config
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(str(asdict(config)))
        
    # 5. Save Test Predictions
    print("Generating predictions on Test Set...")
    pred_dir = 'data/predictions/poisson_mf'
    os.makedirs(pred_dir, exist_ok=True)

    test_u = test_df["u"].to_numpy()
    test_i = test_df["i"].to_numpy()
    y_true = test_df["rating"].to_numpy()
    
    y_pred = model.predict(test_u, test_i)
    
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
    parser = argparse.ArgumentParser(description='Train Poisson MF')
    parser.add_argument('--dataset_mode', type=str, default='train', 
                        choices=['train', 'train+val', 'full'],
                        help='Which dataset splits to use for training')
    args = parser.parse_args()
    
    train_full_poisson(dataset_mode=args.dataset_mode)
