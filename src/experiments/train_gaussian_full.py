import pandas as pd
import numpy as np
import os
import time
from dataclasses import asdict

from src.models.gaussian_mf_cavi_bias import GaussianMFCAVI, GaussianMFCAVIConfig

def train_full_gaussian():
    print("=== Training Full Gaussian MF (CAVI) ===")
    
    # 1. Load Data
    data_path = 'data/processed/interactions_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df[['u', 'i', 'rating']]
    
    # Preprocessing: Center Data
    global_mean = df["rating"].mean()
    print(f"Centering data (Global Mean = {global_mean:.4f})...")
    df_centered = df.copy()
    df_centered["rating"] -= global_mean
    
    # 2. Configure Model (Best Params)
    # {'n_factors': 20, 'sigma2': 0.5, 'eta_theta2': 0.1, 'eta_beta2': 0.01, 'eta_bias2': 0.01, 'max_iter': 100, ...}
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
    print("Starting training on full dataset...")
    start_time = time.time()
    model.fit(df_centered, global_mean=global_mean)
    print(f"Training finished in {time.time() - start_time:.1f}s")
    
    # 4. Save Embeddings
    output_dir = 'data/embeddings/gaussian_mf'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving embeddings to {output_dir}...")
    # Attributes from gaussian_mf_cavi_bias.py: m_theta, m_beta
    
    pd.DataFrame(model.m_theta).to_csv(os.path.join(output_dir, 'user_embeddings.csv'), index=False)
    pd.DataFrame(model.m_beta).to_csv(os.path.join(output_dir, 'item_embeddings.csv'), index=False)
    
    # Save config
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(str(asdict(config)))
        f.write(f"\nglobal_mean: {global_mean}")

    print("Done.")

if __name__ == "__main__":
    train_full_gaussian()
