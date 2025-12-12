import pandas as pd
import numpy as np
import os
import time
import torch
from dataclasses import asdict

from src.data.load_data import load_all_splits
from src.experiments.compare_models import load_best_hyperparams
from src.models.hpf_pytorch import HPF_PyTorch, HPF_PyTorch_Config
from src.evaluation.metrics import rmse, macro_mae
from src.utils.mapping import get_recipe_id_map

import argparse

def train_full_hpf_pytorch(dataset_mode='train'):
    print(f"=== Training Full HPF (PyTorch) | Mode: {dataset_mode} ===")
    
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
    
    # Preprocessing: Shift Ratings by +1
    print("Shifting ratings by +1 for HPF...")
    df_shifted = df.copy()
    df_shifted["rating"] += 1
    
    # Activity counts for model init
    n_users = df_shifted["u"].max() + 1
    n_items = df_shifted["i"].max() + 1
    user_counts = np.zeros(n_users)
    item_counts = np.zeros(n_items)
    u_vals, u_counts = np.unique(df_shifted["u"], return_counts=True)
    user_counts[u_vals] = u_counts
    i_vals, i_counts = np.unique(df_shifted["i"], return_counts=True)
    item_counts[i_vals] = i_counts
    
    # 2. Configure Model (Best Params)
    print("Loading best hyperparameters...")
    hyperparams = load_best_hyperparams()
    config_dict = hyperparams.get('HPF_PyTorch', {})
    
    if config_dict:
        # Filter valid keys
        valid_keys = HPF_PyTorch_Config.__annotations__.keys()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        print(f"Using loaded config: {filtered_config}")
        config = HPF_PyTorch_Config(**filtered_config)
    else:
        print("Using default config (fallback)")
        config = HPF_PyTorch_Config(
            n_factors=20,
            a=1.0,
            a_prime=1.0,
            b_prime=1.0,
            c=1.0,
            c_prime=1.0,
            d_prime=1.0,
            lr=0.01,
            epochs=50,
            verbose=True
        )
    
    model = HPF_PyTorch(n_users, n_items, user_counts, item_counts, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # 3. Train
    print("Starting training...")
    start_time = time.time()
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.u = torch.LongTensor(df["u"].values)
            self.i = torch.LongTensor(df["i"].values)
            self.r = torch.FloatTensor(df["rating"].values)
        def __len__(self): return len(self.r)
        def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

    # Use batch_size from config if available (default 1024 if not in dataclass but it usually should be)
    # Checking HPF_PyTorch_Config definition in memory... it likely doesn't have batch_size based on compare_models usage which hardcoded 4096.
    # But best_hyperparams.txt has batch_size: 1024. 
    # If Config doesn't have it, we shouldn't pass it to Config constructor, but we can use it here.
    batch_size = config_dict.get('batch_size', 4096)
    
    loader = torch.utils.data.DataLoader(SimpleDataset(df_shifted), batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for users, items, ratings in loader:
            optimizer.zero_grad()
            loss = model.loss(users, items, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config.epochs} Loss: {total_loss:.4f}")
            
    print(f"Training finished in {time.time() - start_time:.1f}s")
    
    # 4. Save Embeddings
    output_dir = 'data/embeddings/hpf_pytorch'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving embeddings to {output_dir}...")
    model.eval()
    
    user_emb = model.theta.detach().cpu().numpy()
    item_emb = model.beta.detach().cpu().numpy()
    
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
    pred_dir = 'data/predictions/hpf_pytorch'
    os.makedirs(pred_dir, exist_ok=True)

    test_u = test_df["u"].to_numpy()
    test_i = test_df["i"].to_numpy()
    y_true = test_df["rating"].to_numpy()
    
    # Model was trained on shifted ratings (+1)
    y_pred_shifted = model.predict(test_u, test_i)
    y_pred = y_pred_shifted - 1.0
    
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
    parser = argparse.ArgumentParser(description='Train HPF PyTorch')
    parser.add_argument('--dataset_mode', type=str, default='train', 
                        choices=['train', 'train+val', 'full'],
                        help='Which dataset splits to use for training')
    args = parser.parse_args()
    
    train_full_hpf_pytorch(dataset_mode=args.dataset_mode)
