
import numpy as np
import pandas as pd
from src.data.load_data import load_all_splits
from src.models.hpf_cavi import HPF_CAVI, HPF_CAVI_Config
from src.evaluation.metrics import rmse

def run_experiment():
    print("Loading data...")
    train_df, val_df, test_df = load_all_splits()
    
    # Shift ratings by +1 (0-5 -> 1-6)
    print("Shifting ratings by +1...")
    train_df["rating"] += 1
    val_df["rating"] += 1
    test_df["rating"] += 1
    
    # Config
    # Using priors similar to standard HPF defaults or tuned values
    config = HPF_CAVI_Config(
        n_factors=20,
        a=0.3,
        a_prime=1.0,
        b_prime=1.0,
        c=0.3,
        c_prime=1.0,
        d_prime=1.0,
        max_iter=100,
        verbose=True
    )
    
    print(f"Running HPF_CAVI with config: {config}")
    
    model = HPF_CAVI(config)
    model.fit(train_df, val_df) # Pass val_df for monitoring
    
    print("\nEvaluating...")
    
    # Predict on Train
    train_preds = model.predict(train_df["u"].to_numpy(), train_df["i"].to_numpy())
    # Shift back by -1
    train_rmse = rmse(train_df["rating"].to_numpy() - 1, train_preds - 1)
    
    # Predict on Val
    val_preds = model.predict(val_df["u"].to_numpy(), val_df["i"].to_numpy())
    val_rmse = rmse(val_df["rating"].to_numpy() - 1, val_preds - 1)
    
    # Predict on Test
    test_preds = model.predict(test_df["u"].to_numpy(), test_df["i"].to_numpy())
    test_rmse = rmse(test_df["rating"].to_numpy() - 1, test_preds - 1)
    
    print(f"\n=== Final RMSEs (Original Scale) ===")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Check prediction stats
    print("\nPrediction Stats (Train):")
    print(pd.Series(train_preds).describe())

if __name__ == "__main__":
    run_experiment()
