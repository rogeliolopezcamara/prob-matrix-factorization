# src/experiments/run_poisson_mf_extended.py

from src.data.load_data import load_all_splits
from src.models.poisson_mf_extended_cavi import PoissonMFExtendedCAVI, PoissonMFExtendedCAVIConfig


def main():
    # Load raw ratings
    train_df, val_df, test_df = load_all_splits()

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    # Config with tuned hyperparameters from standard Poisson MF
    # We might need to adjust a0/b0 since we now have scalar factors scaling the rate.
    # E[rate] = E[phi] * E[psi] * K * E[theta] * E[beta]
    # If all initialized with a0/b0:
    # E[rate] = (a0/b0)^2 * K * (a0/b0)^2 = K * (a0/b0)^4
    # With K=50, a0=0.3, b0=1.0: 50 * (0.3)^4 = 50 * 0.0081 = 0.4
    # This is too low (mean rating ~4.5).
    # We need higher a0 or lower b0.
    # Try a0=1.0, b0=1.0 -> 50 * 1 = 50 (too high).
    # Try a0=0.6, b0=1.0 -> 50 * 0.1296 = 6.48 (Reasonable).
    
    config = PoissonMFExtendedCAVIConfig(
        n_factors=50,
        a0=0.6,
        b0=1.0,
        max_iter=20,
        tol=1e-4,
        random_state=42,
        verbose=True,
    )

    print("\nRunning Extended Poisson MF with config:")
    print(config)

    model = PoissonMFExtendedCAVI(config=config)

    # Train
    model.fit(train_df, val_df=val_df)

    # Evaluate
    train_rmse = model.evaluate_rmse(train_df)
    val_rmse   = model.evaluate_rmse(val_df)
    test_rmse  = model.evaluate_rmse(test_df)

    print("\n=== Final RMSEs ===")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
