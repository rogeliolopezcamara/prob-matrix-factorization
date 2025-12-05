
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
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

def run_experiment():
    print("Loading data...")
    train_df, val_df, test_df = load_all_splits()
    
    # Shift ratings by +1
    print("Shifting ratings by +1...")
    train_df["rating"] += 1
    val_df["rating"] += 1
    test_df["rating"] += 1
    
    # Dataset and DataLoader
    train_dataset = RatingDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    
    n_users = max(train_df["u"].max(), val_df["u"].max(), test_df["u"].max()) + 1
    n_items = max(train_df["i"].max(), val_df["i"].max(), test_df["i"].max()) + 1
    
    print(f"Users: {n_users}, Items: {n_items}")
    
    # Compute counts for scaling
    user_counts = np.zeros(n_users)
    item_counts = np.zeros(n_items)
    
    u_vals, u_counts = np.unique(train_df["u"], return_counts=True)
    user_counts[u_vals] = u_counts
    
    i_vals, i_counts = np.unique(train_df["i"], return_counts=True)
    item_counts[i_vals] = i_counts
    
    # Config
    config = HPF_PyTorch_Config(
        n_factors=20,
        a=0.3,
        a_prime=3.0,
        b_prime=1.0,
        c=0.3,
        c_prime=3.0,
        d_prime=1.0,
        lr=0.001,
        epochs=50,
        verbose=True
    )
    
    print(f"Config: {config}")
    
    model = HPF_PyTorch(n_users, n_items, user_counts, item_counts, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    print("\nStarting Training...")
    
    best_val_rmse = float("inf")
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        for users, items, ratings in train_loader:
            optimizer.zero_grad()
            loss = model.loss(users, items, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        val_preds = model.predict(val_df["u"].values, val_df["i"].values)
        val_rmse = rmse(val_df["rating"].values - 1, val_preds - 1)
        
        if config.verbose:
            print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f}")
            
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Save best model state if needed
            
    print("\nEvaluating on Test Set...")
    test_preds = model.predict(test_df["u"].values, test_df["i"].values)
    test_rmse = rmse(test_df["rating"].values - 1, test_preds - 1)
    
    train_preds = model.predict(train_df["u"].values, train_df["i"].values)
    train_rmse = rmse(train_df["rating"].values - 1, train_preds - 1)
    
    print(f"\n=== Final RMSEs (Original Scale) ===")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {best_val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    print("\nPrediction Stats (Train):")
    print(pd.Series(train_preds).describe())

if __name__ == "__main__":
    run_experiment()
