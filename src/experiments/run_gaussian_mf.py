# src/experiments/run_gaussian_mf.py

from src.data.load_data import load_all_splits_centered
from src.models.gaussian_mf_cavi import GaussianMFCAVI, GaussianMFCAVIConfig


def main():
    # Load centered ratings + global mean
    train_df, val_df, test_df, global_mean = load_all_splits_centered()

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print(f"Global rating mean (train): {global_mean:.4f}")

    config = GaussianMFCAVIConfig(
        n_factors=20,
        sigma2=2.0,
        eta_theta2=0.05,
        eta_beta2=0.05,
        max_iter=100,
        tol=1e-6,
        random_state=42,
        verbose=True,
    )

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