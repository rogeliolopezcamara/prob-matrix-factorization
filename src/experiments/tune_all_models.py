import pandas as pd
import numpy as np
import time
import random
import itertools
from dataclasses import asdict

# Models
from src.models.gaussian_mf_cavi_bias import GaussianMFCAVI, GaussianMFCAVIConfig
from src.models.poisson_mf_cavi import PoissonMFCAVI, PoissonMFCAVIConfig
from src.models.hpf_cavi import HPF_CAVI, HPF_CAVI_Config
from src.models.hpf_pytorch import HPF_PyTorch, HPF_PyTorch_Config
from src.experiments.compare_models import run_gaussian_mf, run_poisson_mf, run_hpf_cavi, run_hpf_pytorch
# Note: we will reimplement evaluation loops here to allow dynamic config injection

from src.evaluation.metrics import rmse
import torch

from src.data.load_data import load_all_splits

def load_data():
    print("Loading Data (using load_all_splits)...")
    train_df, val_df, _ = load_all_splits()
    
    # Subsample for tuning speed (e.g., 50k rows for train, 10k for val)
    print("Subsampling for tuning speed...")
    train_sample = train_df.sample(n=min(50000, len(train_df)), random_state=42)
    val_sample = val_df.sample(n=min(10000, len(val_df)), random_state=42)
    
    return train_sample, val_sample

def tune_gaussian_mf(train_df, val_df, n_trials=10):
    print("\n=== Tuning Gaussian MF (CAVI) ===")
    
    # Preprocessing (Center data)
    global_mean = train_df["rating"].mean()
    train_c = train_df.copy()
    train_c["rating"] -= global_mean
    val_c = val_df.copy()
    val_c["rating"] -= global_mean
    
    # Search Space
    param_grid = {
        'n_factors': [20, 50, 100],
        'sigma2': [0.5, 1.0, 2.0],
        'eta_reg': [0.01, 0.1, 1.0], # Combined for simplicity or separate
    }
    
    best_rmse = float('inf')
    best_config = None
    
    for i in range(n_trials):
        # Sample parameters
        factors = random.choice(param_grid['n_factors'])
        sigma2 = random.choice(param_grid['sigma2'])
        # Sample separate regs
        eta_theta2 = random.choice(param_grid['eta_reg'])
        eta_beta2 = random.choice(param_grid['eta_reg'])
        eta_bias2 = random.choice(param_grid['eta_reg'])
        
        config = GaussianMFCAVIConfig(
            n_factors=factors,
            sigma2=sigma2,
            eta_theta2=eta_theta2,
            eta_beta2=eta_beta2,
            eta_bias2=eta_bias2,
            max_iter=50, # Reduced for tuning speed
            tol=1e-3,
            verbose=False,
            random_state=42
        )
        
        try:
            model = GaussianMFCAVI(config)
            model.fit(train_c, val_df=val_c, global_mean=global_mean)
            val_rmse = model.evaluate_rmse(val_c, global_mean)
            
            print(f"Trial {i+1}/{n_trials}: RMSE={val_rmse:.4f} | factors={factors}, s2={sigma2}, reg={eta_theta2}/{eta_beta2}/{eta_bias2}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_config = config
        except Exception as e:
            print(f"Trial {i+1} failed: {e}")

    print(f"Best Gaussian MF RMSE: {best_rmse:.4f}")
    return best_config

def tune_poisson_mf(train_df, val_df, n_trials=10):
    print("\n=== Tuning Poisson MF (CAVI) ===")
    
    # Search Space
    param_grid = {
        'n_factors': [20, 50, 100],
        'a0': [0.1, 0.3, 1.0],
        'b0': [0.3, 1.0, 3.0]
    }
    
    best_rmse = float('inf')
    best_config = None
    
    for i in range(n_trials):
        factors = random.choice(param_grid['n_factors'])
        a0 = random.choice(param_grid['a0'])
        b0 = random.choice(param_grid['b0'])
        
        config = PoissonMFCAVIConfig(
            n_factors=factors,
            a0=a0,
            b0=b0,
            max_iter=30,
            tol=1e-3,
            verbose=False,
            random_state=42
        )
        
        try:
            model = PoissonMFCAVI(config)
            model.fit(train_df, val_df=val_df)
            val_rmse = model.evaluate_rmse(val_df)
            
            print(f"Trial {i+1}/{n_trials}: RMSE={val_rmse:.4f} | factors={factors}, a0={a0}, b0={b0}")
            
            if val_rmse < best_rmse and not np.isnan(val_rmse):
                best_rmse = val_rmse
                best_config = config
        except Exception as e:
            print(f"Trial {i+1} failed: {e}")

    print(f"Best Poisson MF RMSE: {best_rmse:.4f}")
    return best_config

def tune_hpf_cavi(train_df, val_df, n_trials=10):
    print("\n=== Tuning HPF (CAVI) ===")
    
    # Preprocessing: Shift +1
    train_s = train_df.copy()
    train_s['rating'] += 1
    val_s = val_df.copy()
    val_s['rating'] += 1
    
    # Search Space
    # Focusing on a, a' and c, c' which are shape/rate for user/item gammas
    param_grid = {
        'n_factors': [20, 50],
        'hyper_a': [0.3, 1.0],   # a, c
        'hyper_aprime': [1.0, 3.0, 5.0] # a', c', b', d' usually kept similar or scaled
    }
    
    best_rmse = float('inf')
    best_config = None
    
    for i in range(n_trials):
        factors = random.choice(param_grid['n_factors'])
        a = c = random.choice(param_grid['hyper_a'])
        prime = random.choice(param_grid['hyper_aprime'])
        
        config = HPF_CAVI_Config(
            n_factors=factors,
            a=a, a_prime=prime, b_prime=prime,
            c=c, c_prime=prime, d_prime=prime,
            max_iter=50,
            tol=1e-3,
            verbose=False
        )
        
        try:
            model = HPF_CAVI(config)
            model.fit(train_s, val_df=val_s)
            
            # Eval shifted
            preds = model.predict(val_s["u"].to_numpy(), val_s["i"].to_numpy())
            val_rmse = rmse(val_s["rating"].to_numpy() - 1, preds - 1)
            
            print(f"Trial {i+1}/{n_trials}: RMSE={val_rmse:.4f} | factors={factors}, a={a}, prime={prime}")
            
            if val_rmse < best_rmse and not np.isnan(val_rmse):
                best_rmse = val_rmse
                best_config = config
        except Exception as e:
            print(f"Trial {i+1} failed: {e}")
            
    print(f"Best HPF CAVI RMSE: {best_rmse:.4f}")
    return best_config

def tune_hpf_pytorch(train_df, val_df, n_trials=10):
    print("\n=== Tuning HPF (PyTorch) ===")
    
    train_s = train_df.copy()
    train_s['rating'] += 1
    val_s = val_df.copy()
    val_s['rating'] += 1
    
    n_users = max(train_s["u"].max(), val_s["u"].max()) + 1
    n_items = max(train_s["i"].max(), val_s["i"].max()) + 1
    
    # Counts for inverse scaling (which we assume is helpful based on history)
    user_counts = np.zeros(n_users)
    item_counts = np.zeros(n_items)
    u_vals, u_counts = np.unique(train_s["u"], return_counts=True)
    user_counts[u_vals] = u_counts
    i_vals, i_counts = np.unique(train_s["i"], return_counts=True)
    item_counts[i_vals] = i_counts

    # Search Space
    param_grid = {
        'n_factors': [20, 50],
        'lr': [0.001, 0.005, 0.01],
        'hyper_a': [0.3, 1.0],
        'hyper_prime': [1.0, 3.0]
    }
    
    best_rmse = float('inf')
    best_config = None
    
    # Dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.u = torch.LongTensor(df["u"].values)
            self.i = torch.LongTensor(df["i"].values)
            self.r = torch.FloatTensor(df["rating"].values)
        def __len__(self): return len(self.r)
        def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]
    
    train_loader = torch.utils.data.DataLoader(SimpleDataset(train_s), batch_size=4096, shuffle=True)
    
    for i in range(n_trials):
        factors = random.choice(param_grid['n_factors'])
        lr = random.choice(param_grid['lr'])
        a = c = random.choice(param_grid['hyper_a'])
        prime = random.choice(param_grid['hyper_prime'])
        
        config = HPF_PyTorch_Config(
            n_factors=factors,
            a=a, a_prime=prime, b_prime=prime,
            c=c, c_prime=prime, d_prime=prime,
            lr=lr,
            epochs=20, # Short epochs for tuning
            verbose=False
        )
        
        try:
            model = HPF_PyTorch(n_users, n_items, user_counts, item_counts, config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            
            # Train
            model.train()
            for epoch in range(config.epochs):
                for users, items, ratings in train_loader:
                    optimizer.zero_grad()
                    loss = model.loss(users, items, ratings)
                    loss.backward()
                    optimizer.step()
            
            # Eval
            model.eval()
            preds = model.predict(val_s["u"].values, val_s["i"].values)
            val_rmse = rmse(val_s["rating"].values - 1, preds - 1)
            
            print(f"Trial {i+1}/{n_trials}: RMSE={val_rmse:.4f} | factors={factors}, lr={lr}, a={a}, prime={prime}")
            
            if val_rmse < best_rmse and not np.isnan(val_rmse):
                best_rmse = val_rmse
                best_config = config
        except Exception as e:
            print(f"Trial {i+1} failed: {e}")

    print(f"Best HPF PyTorch RMSE: {best_rmse:.4f}")
    return best_config


def main():
    train_df, val_df = load_data()
    
    best_gaussian = tune_gaussian_mf(train_df, val_df, n_trials=5)
    best_poisson = tune_poisson_mf(train_df, val_df, n_trials=5)
    best_hpf_cavi = tune_hpf_cavi(train_df, val_df, n_trials=5)
    best_hpf_pytorch = tune_hpf_pytorch(train_df, val_df, n_trials=5)
    
    print("\n\n=== TUNING COMPLETE. BEST CONFIGURATIONS ===")
    if best_gaussian: print(f"GaussianMF: {asdict(best_gaussian)}")
    if best_poisson: print(f"PoissonMF: {asdict(best_poisson)}")
    if best_hpf_cavi: print(f"HPF_CAVI: {asdict(best_hpf_cavi)}")
    if best_hpf_pytorch: print(f"HPF_PyTorch: {asdict(best_hpf_pytorch)}")
    
    # Write to file for easy copy-pasting
    with open("best_hyperparams.txt", "w") as f:
        f.write("BEST CONFIGURATIONS\n")
        f.write("===================\n")
        if best_gaussian: f.write(f"GaussianMF: {asdict(best_gaussian)}\n")
        if best_poisson: f.write(f"PoissonMF: {asdict(best_poisson)}\n")
        if best_hpf_cavi: f.write(f"HPF_CAVI: {asdict(best_hpf_cavi)}\n")
        if best_hpf_pytorch: f.write(f"HPF_PyTorch: {asdict(best_hpf_pytorch)}\n")

if __name__ == "__main__":
    main()
