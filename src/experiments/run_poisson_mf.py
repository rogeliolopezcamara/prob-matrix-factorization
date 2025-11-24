# src/experiments/run_poisson_mf.py

from src.data.load_data import load_all_splits
from src.models.poisson_mf_cavi import PoissonMFCAVI, PoissonMFCAVIConfig


def main():
    # Load raw ratings (non-negative)
    # Note: Poisson assumes non-negative integers usually, but works for non-negative floats too.
    # Ratings are 0-5 (or 1-5).
    train_df, val_df, test_df = load_all_splits()

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    
    # Check for negative values just in case
    if (train_df["rating"] < 0).any():
        raise ValueError("Ratings must be non-negative for Poisson MF.")

    config = PoissonMFCAVIConfig(
        n_factors=50,
        a0=0.3,
        b0=1.0,
        max_iter=20,
        tol=1e-4,
        random_state=42,
        verbose=True,
    )

    print("\nRunning Poisson MF with config:")
    print(config)

    model = PoissonMFCAVI(config=config)

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
