
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import itertools
from src.data.load_data import load_all_splits
from src.models.hpf_pytorch import HPF_PyTorch, HPF_PyTorch_Config
from src.evaluation.metrics import rmse

class RatingDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df["u"].values)
        self.items = torch.LongTensor(df["i"].values)
        self.ratings = torch.FloatTensor(df["rating"].values)
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def run_tuning():
    print("Loading data...")
    train_df, val_df, test_df = load_all_splits()
    
    # Shift ratings by +1
    train_df["rating"] += 1
    val_df["rating"] += 1
    
    # Dataset
    train_dataset = RatingDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    
    n_users = max(train_df["u"].max(), val_df["u"].max(), test_df["u"].max()) + 1
    n_items = max(train_df["i"].max(), val_df["i"].max(), test_df["i"].max()) + 1
    
    # Compute counts for scaling
    user_counts = np.zeros(n_users)
    item_counts = np.zeros(n_items)
    
    u_vals, u_counts = np.unique(train_df["u"], return_counts=True)
    user_counts[u_vals] = u_counts
    
    i_vals, i_counts = np.unique(train_df["i"], return_counts=True)
    item_counts[i_vals] = i_counts
    
    # Grid Search
    param_grid = {
        'n_factors': [20, 50],
        'lr': [0.001, 0.005],
        'a': [0.3, 1.0],
        'a_prime': [1.0, 3.0]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}")
    
    best_rmse = float('inf')
    best_config = None
    
    for i, params in enumerate(combinations):
        print(f"\n--- Run {i+1}/{len(combinations)}: {params} ---")
        
        config = HPF_PyTorch_Config(
            n_factors=params['n_factors'],
            a=params['a'],
            a_prime=params['a_prime'],
            b_prime=1.0,
            c=0.3,
            c_prime=1.0, # Keep symmetric with a_prime? Or fixed? Let's keep fixed for now to reduce space.
            d_prime=1.0,
            lr=params['lr'],
            epochs=10, # Short run for tuning
            verbose=False
        )
        
        model = HPF_PyTorch(n_users, n_items, user_counts, item_counts, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        min_val_rmse = float('inf')
        
        for epoch in range(config.epochs):
            model.train()
            for users, items, ratings in train_loader:
                optimizer.zero_grad()
                loss = model.loss(users, items, ratings)
                loss.backward()
                optimizer.step()
            
            # Eval
            model.eval()
            val_preds = model.predict(val_df["u"].values, val_df["i"].values)
            val_rmse = rmse(val_df["rating"].values - 1, val_preds - 1)
            
            if val_rmse < min_val_rmse:
                min_val_rmse = val_rmse
        
        print(f"Result RMSE: {min_val_rmse:.4f}")
        
        if min_val_rmse < best_rmse:
            best_rmse = min_val_rmse
            best_config = params
            print(f"*** New Best RMSE: {best_rmse:.4f} ***")

    print(f"\nBest Configuration: {best_config}")
    print(f"Best Validation RMSE: {best_rmse:.4f}")

if __name__ == "__main__":
    run_tuning()
