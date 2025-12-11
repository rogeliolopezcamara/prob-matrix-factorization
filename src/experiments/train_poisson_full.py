import pandas as pd
import numpy as np
import os
import time
from dataclasses import asdict

from src.models.poisson_mf_cavi import PoissonMFCAVI, PoissonMFCAVIConfig

def train_full_poisson():
    print("=== Training Full Poisson MF (CAVI) ===")
    
    # 1. Load Data
    data_path = 'data/processed/interactions_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Select cols
    df = df[['u', 'i', 'rating']]
    
    # 2. Configure Model (Best Params)
    # {'n_factors': 100, 'a0': 0.1, 'b0': 1.0, 'max_iter': 50, 'tol': 0.0001, 'random_state': 42}
    config = PoissonMFCAVIConfig(
        n_factors=100,
        a0=0.1,
        b0=1.0,
        max_iter=100, # Providing full convergence room
        tol=1e-4,
        random_state=42,
        verbose=True
    )
    
    model = PoissonMFCAVI(config)
    
    # 3. Train
    print("Starting training on full dataset...")
    start_time = time.time()
    model.fit(df) # No validation set passed, trains on everything
    print(f"Training finished in {time.time() - start_time:.1f}s")
    
    # 4. Save Embeddings
    output_dir = 'data/embeddings/poisson_mf'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving embeddings to {output_dir}...")
    # Expected value of theta = gamma_u / rho_u
    # Expected value of beta = lambda_i / eta_i
    # Check if properties exist, otherwise compute
    if hasattr(model, 'E_theta'):
        user_emb = model.E_theta
    else:
        user_emb = model.gamma_u / model.rho_u
        
    if hasattr(model, 'E_beta'):
        item_emb = model.E_beta
    else:
        item_emb = model.lambda_i / model.eta_i
        
    pd.DataFrame(user_emb).to_csv(os.path.join(output_dir, 'user_embeddings.csv'), index=False)
    pd.DataFrame(item_emb).to_csv(os.path.join(output_dir, 'item_embeddings.csv'), index=False)
    
    # Save config
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(str(asdict(config)))
        
    print("Done.")

if __name__ == "__main__":
    train_full_poisson()
