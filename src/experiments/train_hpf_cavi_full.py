import pandas as pd
import numpy as np
import os
import time
from dataclasses import asdict

from src.models.hpf_cavi import HPF_CAVI, HPF_CAVI_Config

def train_full_hpf_cavi():
    print("=== Training Full HPF (CAVI) ===")
    
    # 1. Load Data
    data_path = 'data/processed/interactions_processed.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df[['u', 'i', 'rating']]
    
    # Preprocessing: Shift Ratings by +1
    print("Shifting ratings by +1 for HPF...")
    df_shifted = df.copy()
    df_shifted["rating"] += 1
    
    # 2. Configure Model (Best Params)
    # {'n_factors': 50, 'a': 1.0, 'a_prime': 1.0, 'b_prime': 1.0, 'c': 1.0, 'c_prime': 1.0, 'd_prime': 1.0, ...}
    config = HPF_CAVI_Config(
        n_factors=50,
        a=1.0,
        a_prime=1.0,
        b_prime=1.0,
        c=1.0,
        c_prime=1.0,
        d_prime=1.0,
        max_iter=100,
        tol=1e-4,
        random_state=42,
        verbose=True
    )
    
    model = HPF_CAVI(config)
    
    # 3. Train
    print("Starting training on full dataset...")
    start_time = time.time()
    model.fit(df_shifted)
    print(f"Training finished in {time.time() - start_time:.1f}s")
    
    # 4. Save Embeddings
    output_dir = 'data/embeddings/hpf_cavi'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving embeddings to {output_dir}...")
    # Gamma user shape/rate -> mean is shape/rate
    # Attributes from hpf_cavi.py:
    # gamma_a_theta, gamma_b_theta (or just use E_theta)
    
    if hasattr(model, 'E_theta'):
        user_emb = model.E_theta
    else:
        user_emb = model.gamma_a_theta / model.gamma_b_theta
        
    if hasattr(model, 'E_beta'):
        item_emb = model.E_beta
    else:
        item_emb = model.gamma_a_beta / model.gamma_b_beta
    
    pd.DataFrame(user_emb).to_csv(os.path.join(output_dir, 'user_embeddings.csv'), index=False)
    pd.DataFrame(item_emb).to_csv(os.path.join(output_dir, 'item_embeddings.csv'), index=False)
    
    # Save config
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(str(asdict(config)))

    print("Done.")

if __name__ == "__main__":
    train_full_hpf_cavi()
