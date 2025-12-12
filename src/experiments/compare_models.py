import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
from dataclasses import asdict

# Data
import ast
import os

# Data
from src.data.load_data import load_all_splits

# Models
from src.models.gaussian_mf_cavi_bias import GaussianMFCAVI, GaussianMFCAVIConfig
from src.models.poisson_mf_cavi import PoissonMFCAVI, PoissonMFCAVIConfig
from src.models.hpf_cavi import HPF_CAVI, HPF_CAVI_Config
from src.models.hpf_pytorch import HPF_PyTorch, HPF_PyTorch_Config

# Evaluator
from src.evaluation.metrics import rmse, macro_mae

def load_best_hyperparams(filepath='best_hyperparams.txt'):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using default hyperparameters.")
        return {}
    
    configs = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="): continue
            
            parts = line.split(":", 1)
            if len(parts) == 2:
                model_name = parts[0].strip()
                config_str = parts[1].strip()
                try:
                    config_dict = ast.literal_eval(config_str)
                    configs[model_name] = config_dict
                except Exception as e:
                    print(f"Error parsing config for {model_name}: {e}")
                    
    print(f"Loaded hyperparameters from {filepath}")
    return configs

def run_gaussian_mf(train_df, val_df, test_df, config_dict=None, verbose=False):
    print("  -> Initializing Gaussian MF (Bias)...", flush=True)
    
    # Preprocessing: Gaussian MF Improved uses Centered Data
    print("     [GaussianMF] Centering data...", flush=True)
    global_mean = train_df["rating"].mean()
    # We train on centered data, but the model expects the dataframe to just have column 'rating'.
    # So we create copies.
    train_centered = train_df.copy()
    train_centered["rating"] -= global_mean
    
    # Validation/Test also used for monitoring in fit(), so center them too if passed
    val_centered = val_df.copy()
    val_centered["rating"] -= global_mean
    
    test_centered = test_df.copy()
    test_centered["rating"] -= global_mean

    # Hyperparameters from tune_all_models.py (loaded via config_dict if available)
    if config_dict:
        print(f"     [GaussianMF] Using loaded config: {config_dict}")
        config = GaussianMFCAVIConfig(**config_dict)
    else:
        print("     [GaussianMF] Using default config")
        config = GaussianMFCAVIConfig(
            n_factors=20,
            sigma2=0.5,
            eta_theta2=0.1,
            eta_beta2=0.01,
            eta_bias2=0.01,
            max_iter=100,
            tol=1e-8,
            random_state=42,
            verbose=verbose
        )
    model = GaussianMFCAVI(config)
    
    print("     [GaussianMF] Starting training (max_iter=100)...", flush=True)
    start_time = time.time()
    # pass global_mean as argument if fit supports it or uses it for init, 
    # but the key is the data passed in.
    # checking gaussian_mf_cavi_bias.py: fit(train_df, val_df=None, global_mean=0.0)
    model.fit(train_centered, val_df=val_centered, global_mean=global_mean)
    train_time = time.time() - start_time
    print(f"     [GaussianMF] Training finished in {train_time:.1f}s", flush=True)
    
    # Evaluate using helper that adds global mean back
    print("     [GaussianMF] Evaluating...", flush=True)
    train_rmse = model.evaluate_rmse(train_centered, global_mean)
    val_rmse = model.evaluate_rmse(val_centered, global_mean)
    test_rmse = model.evaluate_rmse(test_centered, global_mean)
    
    # Calculate Macro MAE
    # We need predictions on the original scale
    train_preds = model.predict(train_centered["u"].to_numpy(), train_centered["i"].to_numpy(), global_mean)
    val_preds = model.predict(val_centered["u"].to_numpy(), val_centered["i"].to_numpy(), global_mean)
    test_preds = model.predict(test_centered["u"].to_numpy(), test_centered["i"].to_numpy(), global_mean)
    
    train_mae = macro_mae(train_df["rating"].to_numpy(), train_preds)
    val_mae = macro_mae(val_df["rating"].to_numpy(), val_preds)
    test_mae = macro_mae(test_df["rating"].to_numpy(), test_preds)
    
    return {
        "Model": "Gaussian MF (CAVI)",
        "Train RMSE": train_rmse,
        "Val RMSE": val_rmse,
        "Test RMSE": test_rmse,
        "Train MacroMAE": train_mae,
        "Val MacroMAE": val_mae,
        "Test MacroMAE": test_mae,
        "Time (s)": train_time,
        "Config": str(asdict(config))
    }

def run_poisson_mf(train_df, val_df, test_df, config_dict=None, verbose=False):
    print("  -> Initializing Poisson MF (CAVI)...", flush=True)
    
    # Hyperparameters from tune_all_models.py (loaded via config_dict if available)
    if config_dict:
        print(f"     [PoissonMF] Using loaded config: {config_dict}")
        config = PoissonMFCAVIConfig(**config_dict)
    else:
        print("     [PoissonMF] Using default config")
        config = PoissonMFCAVIConfig(
            n_factors=100,
            a0=0.1,
            b0=1.0,
            max_iter=50, # Increased slightly
            tol=1e-4,
            random_state=42,
            verbose=verbose,
        )
    
    model = PoissonMFCAVI(config)
    
    print("     [PoissonMF] Starting training...", flush=True)
    start_time = time.time()
    model.fit(train_df, val_df=val_df)
    train_time = time.time() - start_time
    print(f"     [PoissonMF] Training finished in {train_time:.1f}s", flush=True)
    
    train_rmse = model.evaluate_rmse(train_df)
    val_rmse = model.evaluate_rmse(val_df)
    test_rmse = model.evaluate_rmse(test_df)
    
    # Calculate Macro MAE
    train_preds = model.predict(train_df["u"].to_numpy(), train_df["i"].to_numpy())
    val_preds = model.predict(val_df["u"].to_numpy(), val_df["i"].to_numpy())
    test_preds = model.predict(test_df["u"].to_numpy(), test_df["i"].to_numpy())
    
    train_mae = macro_mae(train_df["rating"].to_numpy(), train_preds)
    val_mae = macro_mae(val_df["rating"].to_numpy(), val_preds)
    test_mae = macro_mae(test_df["rating"].to_numpy(), test_preds)
    
    return {
        "Model": "Poisson MF (CAVI)",
        "Train RMSE": train_rmse,
        "Val RMSE": val_rmse,
        "Test RMSE": test_rmse,
        "Train MacroMAE": train_mae,
        "Val MacroMAE": val_mae,
        "Test MacroMAE": test_mae,
        "Time (s)": train_time,
        "Config": str(asdict(config))
    }

def run_hpf_cavi(train_df, val_df, test_df, config_dict=None, verbose=False):
    print("  -> Initializing HPF (CAVI)...", flush=True)
    
    # Preprocessing: Shift ratings by +1 as per run_hpf_cavi.py
    print("     [HPF_CAVI] Shifting ratings...", flush=True)
    train_shifted = train_df.copy()
    train_shifted["rating"] += 1
    val_shifted = val_df.copy()
    val_shifted["rating"] += 1
    test_shifted = test_df.copy()
    test_shifted["rating"] += 1

    # Hyperparameters from tune_all_models.py (loaded via config_dict if available)
    if config_dict:
        print(f"     [HPF_CAVI] Using loaded config: {config_dict}")
        config = HPF_CAVI_Config(**config_dict)
    else:
        print("     [HPF_CAVI] Using default config")
        # Best: {'n_factors': 50, 'a': 1.0, 'a_prime': 1.0...}
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
            verbose=verbose
        )
    model = HPF_CAVI(config)
    
    print("     [HPF_CAVI] Starting training...", flush=True)
    start_time = time.time()
    model.fit(train_shifted, val_df=val_shifted)
    train_time = time.time() - start_time
    print(f"     [HPF_CAVI] Training finished in {train_time:.1f}s", flush=True)
    
    # Evaluate - model predicts on shifted scale, need to shift back for RMSE
    # Evaluate - model predicts on shifted scale, need to shift back for RMSE
    def get_metrics_shifted(model, df):
        preds = model.predict(df["u"].to_numpy(), df["i"].to_numpy())
        rmse_val = rmse(df["rating"].to_numpy() - 1, preds - 1)
        mae_val = macro_mae(df["rating"].to_numpy() - 1, preds - 1)
        return rmse_val, mae_val
        
    train_rmse, train_mae = get_metrics_shifted(model, train_shifted)
    val_rmse, val_mae = get_metrics_shifted(model, val_shifted)
    test_rmse, test_mae = get_metrics_shifted(model, test_shifted)
    
    return {
        "Model": "HPF (CAVI)",
        "Train RMSE": train_rmse,
        "Val RMSE": val_rmse,
        "Test RMSE": test_rmse,
        "Train MacroMAE": train_mae,
        "Val MacroMAE": val_mae,
        "Test MacroMAE": test_mae,
        "Time (s)": train_time,
        "Config": str(asdict(config))
    }

def run_hpf_pytorch(train_df, val_df, test_df, config_dict=None, verbose=False):
    print("  -> Initializing HPF (PyTorch)...", flush=True)
    
    # Preprocessing: Shift ratings by +1 as per run_hpf_pytorch.py
    print("     [HPF_PyTorch] Shifting ratings...", flush=True)
    train_shifted = train_df.copy()
    train_shifted["rating"] += 1
    val_shifted = val_df.copy()
    val_shifted["rating"] += 1
    test_shifted = test_df.copy()
    test_shifted["rating"] += 1
    
    n_users = max(train_df["u"].max(), val_df["u"].max(), test_df["u"].max()) + 1
    n_items = max(train_df["i"].max(), val_df["i"].max(), test_df["i"].max()) + 1
    
    # Compute counts
    user_counts = np.zeros(n_users)
    item_counts = np.zeros(n_items)
    u_vals, u_counts = np.unique(train_df["u"], return_counts=True)
    user_counts[u_vals] = u_counts
    i_vals, i_counts = np.unique(train_df["i"], return_counts=True)
    item_counts[i_vals] = i_counts
    
    # Hyperparameters from tune_all_models.py (loaded via config_dict if available)
    if config_dict:
        # Filter out keys that are not in HPF_PyTorch_Config
        valid_keys = HPF_PyTorch_Config.__annotations__.keys()
        # Also remove 'device' if present but not in Config (or if Config handles it)
        # Checking Config... usually just basic params.
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        print(f"     [HPF_PyTorch] Using loaded config: {filtered_config}")
        config = HPF_PyTorch_Config(**filtered_config)
    else:
        print("     [HPF_PyTorch] Using default config")
        # Best: {'n_factors': 20, 'a': 1.0, 'prime': 1.0, 'lr': 0.01}
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
            verbose=verbose
        )
    
    model = HPF_PyTorch(n_users, n_items, user_counts, item_counts, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # Dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.u = torch.LongTensor(df["u"].values)
            self.i = torch.LongTensor(df["i"].values)
            self.r = torch.FloatTensor(df["rating"].values)
        def __len__(self): return len(self.r)
        def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]

    train_loader = torch.utils.data.DataLoader(SimpleDataset(train_shifted), batch_size=4096, shuffle=True)
    
    print("     [HPF_PyTorch] Starting training (epochs=50)...", flush=True)
    start_time = time.time()
    
    # Train Loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for users, items, ratings in train_loader:
            optimizer.zero_grad()
            loss = model.loss(users, items, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if verbose and (epoch % 5 == 0 or epoch == config.epochs - 1):
             print(f"     [HPF_PyTorch] Epoch {epoch+1}/{config.epochs} Loss: {total_loss:.4f}", flush=True)
    
    train_time = time.time() - start_time
    print(f"     [HPF_PyTorch] Training finished in {train_time:.1f}s", flush=True)
    
    # Predict (Shift back -1)
    model.eval()
    
    def get_metrics(model, df):
        preds = model.predict(df["u"].values, df["i"].values)
        rmse_val = rmse(df["rating"].values - 1, preds - 1)
        mae_val = macro_mae(df["rating"].values - 1, preds - 1)
        return rmse_val, mae_val
        
    train_rmse, train_mae = get_metrics(model, train_shifted)
    val_rmse, val_mae = get_metrics(model, val_shifted)
    test_rmse, test_mae = get_metrics(model, test_shifted)
    
    return {
        "Model": "HPF (PyTorch)",
        "Train RMSE": train_rmse,
        "Val RMSE": val_rmse,
        "Test RMSE": test_rmse,
        "Train MacroMAE": train_mae,
        "Val MacroMAE": val_mae,
        "Test MacroMAE": test_mae,
        "Time (s)": train_time,
        "Config": str(asdict(config))
    }

def plot_results(results_df):
    # Set style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('bmh') # Fallback
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. RMSE Comparison
    rmse_df = results_df[["Model", "Train RMSE", "Val RMSE", "Test RMSE"]].melt(
        id_vars="Model", var_name="Split", value_name="RMSE"
    )
    
    columns = ["Train RMSE", "Val RMSE", "Test RMSE"]
    pivot_rmse = results_df.set_index("Model")[columns]
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    pivot_rmse.plot(kind='bar', ax=axes[0], color=colors, alpha=0.9, width=0.8)
    
    axes[0].set_title("RMSE Comparison (Lower is Better)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("RMSE", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower left')
    
    # Adjust y-limit to show labels well
    y_max = pivot_rmse.max().max()
    axes[0].set_ylim(0, y_max * 1.15)
    
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.3f', padding=3, rotation=0, fontsize=10, fontweight='bold')

    
    # 2. Time Comparison (Now on 3rd slot, let's make 3 plots)
    # Re-layout: 1 row, 3 cols
    plt.close(fig) # close old one
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    columns = ["Train RMSE", "Val RMSE", "Test RMSE"]
    pivot_rmse = results_df.set_index("Model")[columns]
    
    # Colors
    colors_rmse = ['#1f77b4', '#aec7e8', '#ff7f0e']
    
    pivot_rmse.plot(kind='bar', ax=axes[0], color=colors_rmse, alpha=0.9, width=0.8)
    axes[0].set_title("RMSE (Lower is Better)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("RMSE", fontsize=12)
    axes[0].tick_params(axis='x', rotation=0)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    # 2. Macro MAE Comparison
    mae_columns = ["Train MacroMAE", "Val MacroMAE", "Test MacroMAE"]
    pivot_mae = results_df.set_index("Model")[mae_columns]
    
    colors_mae = ['#2ca02c', '#98df8a', '#d62728']
    
    pivot_mae.plot(kind='bar', ax=axes[1], color=colors_mae, alpha=0.9, width=0.8)
    axes[1].set_title("Macro-MAE (Lower is Better)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Macro MAE", fontsize=12)
    axes[1].tick_params(axis='x', rotation=0)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    # 3. Time Comparison
    times = results_df.set_index("Model")["Time (s)"]
    bars = axes[2].bar(times.index, times.values, color='#9467bd', alpha=0.7)
    
    axes[2].set_title("Training Time (Seconds)", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Time (s)", fontsize=12)
    axes[2].tick_params(axis='x', rotation=0)
    
    axes[2].bar_label(bars, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("model_comparison_plots.png", dpi=150)
    print("\nPlots saved to model_comparison_plots.png", flush=True)
    
    # Create a metadata text file for the params
    with open("model_comparison_params.txt", "w") as f:
        for idx, row in results_df.iterrows():
            f.write(f"=== {row['Model']} ===\n")
            f.write(f"{row['Config']}\n\n")
    print("Parameters saved to model_comparison_params.txt", flush=True)

def main():
    print("Loading Data (using load_all_splits)...", flush=True)
    try:
        train_df, val_df, test_df = load_all_splits()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load Hyperparameters
    hyperparams = load_best_hyperparams('best_hyperparams.txt')

    results = []
    
    # 1. Gaussian MF
    try:
        config = hyperparams.get('GaussianMF')
        results.append(run_gaussian_mf(train_df, val_df, test_df, config_dict=config, verbose=True))
    except Exception as e:
        print(f"GaussianMF failed: {e}")
        import traceback
        traceback.print_exc()
        
    # 2. Poisson MF
    try:
        config = hyperparams.get('PoissonMF')
        results.append(run_poisson_mf(train_df, val_df, test_df, config_dict=config, verbose=True))
    except Exception as e:
        print(f"PoissonMF failed: {e}")

    # 3. HPF CAVI
    try:
        config = hyperparams.get('HPF_CAVI')
        results.append(run_hpf_cavi(train_df, val_df, test_df, config_dict=config, verbose=True))
    except Exception as e:
        print(f"HPF CAVI failed: {e}")
        
    # 4. HPF PyTorch
    try:
        config = hyperparams.get('HPF_PyTorch')
        results.append(run_hpf_pytorch(train_df, val_df, test_df, config_dict=config, verbose=True))
    except Exception as e:
        print(f"HPF PyTorch failed: {e}")
        import traceback
        traceback.print_exc()
        
    results_df = pd.DataFrame(results)
    
    print("\n=== FINAL RESULTS ===", flush=True)
    print("\n=== FINAL RESULTS ===", flush=True)
    print(results_df[["Model", "Train RMSE", "Val RMSE", "Test RMSE", "Train MacroMAE", "Val MacroMAE", "Test MacroMAE", "Time (s)"]])
    
    plot_results(results_df)

if __name__ == "__main__":
    main()
