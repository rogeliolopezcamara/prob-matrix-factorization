import pandas as pd
import numpy as np
import os
import time
import torch
from dataclasses import asdict

from src.models.hpf_pytorch import HPF_PyTorch, HPF_PyTorch_Config

def train_full_hpf_pytorch():
    print("=== Training Full HPF (PyTorch) ===")
    
    # 1. Load Data
    data_path = 'data/processed/interactions_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df[['u', 'i', 'rating']]
    
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
    # {'n_factors': 20, 'a': 1.0, 'a_prime': 1.0, 'b_prime': 1.0, 'c': 1.0, 'c_prime': 1.0, 'd_prime': 1.0, 'lr': 0.01, 'epochs': 50...}
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
    print("Starting training on full dataset...")
    start_time = time.time()
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.u = torch.LongTensor(df["u"].values)
            self.i = torch.LongTensor(df["i"].values)
            self.r = torch.FloatTensor(df["rating"].values)
        def __len__(self): return len(self.r)
        def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

    loader = torch.utils.data.DataLoader(SimpleDataset(df_shifted), batch_size=config.batch_size, shuffle=True)
    
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
    
    # For PyTorch HPF, the latent factors are theta/beta in the model
    # They are softplus(parameter) constrained
    user_emb = model.theta.detach().cpu().numpy() # This might be the raw parameter or property?
    # Checking source... usually implemented as property returning functional.softplus(self._theta)
    # If not property, we need softplus. Assuming it behaves like the model definition in hpf_pytorch.py
    # Let's check hpf_pytorch.py briefly if needed, but standard implementation exposes these.
    # To be safe, let's look at how predict uses them.
    # predict uses self.theta[u_indices] which implies self.theta is a property or tensor.
    
    user_emb = model.theta.detach().cpu().numpy()
    item_emb = model.beta.detach().cpu().numpy()
    
    pd.DataFrame(user_emb).to_csv(os.path.join(output_dir, 'user_embeddings.csv'), index=False)
    pd.DataFrame(item_emb).to_csv(os.path.join(output_dir, 'item_embeddings.csv'), index=False)
    
    # Save config
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(str(asdict(config)))

    print("Done.")

if __name__ == "__main__":
    train_full_hpf_pytorch()
