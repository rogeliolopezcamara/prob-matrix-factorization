# src/experiments/run_gaussian_mf_improved.py

from src.data.load_data import load_all_splits_centered
from src.models.gaussian_mf_cavi_bias import GaussianMFCAVI, GaussianMFCAVIConfig


def main():
    # Load centered ratings + global mean
    train_df, val_df, test_df, global_mean = load_all_splits_centered()

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print(f"Global rating mean (train): {global_mean:.4f}")

    # Tuned hyperparameters
    config = GaussianMFCAVIConfig(
        n_factors=20,
        sigma2=1.0,       # Trust data moderately (was 2.0)
        eta_theta2=0.1,   # Looser prior (was 0.05)
        eta_beta2=0.1,    # Looser prior (was 0.05)
        eta_bias2=0.1,    # Prior for biases
        max_iter=100,
        tol=1e-8,         # Tighter tolerance to encourage more iterations
        random_state=42,
        verbose=True,
    )

    print("\nRunning with improved config:")
    print(config)

    model = GaussianMFCAVI(config=config)

    # Train with centered ratings
    model.fit(train_df, val_df=val_df, global_mean=global_mean)

    # Evaluate in original 0–5 scale
    train_rmse = model.evaluate_rmse(train_df, global_mean)
    val_rmse   = model.evaluate_rmse(val_df, global_mean)
    test_rmse  = model.evaluate_rmse(test_df, global_mean)

    print("\n=== Final RMSEs (original rating scale) ===")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
