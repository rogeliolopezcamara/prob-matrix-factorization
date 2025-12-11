# src/experiments/run_gaussian_mf_best_k.py

from src.data.load_data import load_all_splits_centered
from src.models.gaussian_mf_cavi import GaussianMFCAVI, GaussianMFCAVIConfig
from src.evaluation.metrics import GaussianLogPredictiveLikelihood
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Load centered ratings + global mean
    train_df, val_df, test_df, global_mean = load_all_splits_centered()

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)
    print(f"Global rating mean (train): {global_mean:.4f}")

    # Initialize vectors
    train_rmse_vector = []
    val_rmse_vector = []
    test_rmse_vector = []
    ll_vector = []

    # max number of latent dimensions
    K=60
    grid = range(2,K+1)

    for k in grid:
        config = GaussianMFCAVIConfig(
            n_factors=k,
            sigma2=2.0,
            eta_theta2=0.05,
            eta_beta2=0.05,
            max_iter=100,
            tol=1e-6,
            random_state=42,
            verbose=False,
        )

        print(f"\nRunning Gaussian Factorization with k={k}:")
        print(config)

        model = GaussianMFCAVI(config=config)

        # Train
        model.fit(train_df, val_df=val_df)

        # Evaluate
        train_rmse = model.evaluate_rmse(train_df, global_mean)
        val_rmse   = model.evaluate_rmse(val_df, global_mean)
        test_rmse  = model.evaluate_rmse(test_df, global_mean)
        ll = GaussianLogPredictiveLikelihood(test_df,model.m_theta,model.m_beta,config.sigma2)

        # Update vectors
        train_rmse_vector.append(train_rmse)
        val_rmse_vector.append(val_rmse)
        test_rmse_vector.append(test_rmse)
        ll_vector.append(ll)

        print("\n=== Final RMSEs ===")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")

        print("\n=== Log Predictive Likelihood ===")
        print(f"Total LPL: {ll:.4f}")

    print('\n===Highest Log Predictive Likelihood===')
    print(f"k = {grid[np.argmax(ll_vector)]}")

    # plot RMSE by split
    arrays = [train_rmse_vector, val_rmse_vector, test_rmse_vector]
    names  = ["train", "validation", "test"]

    for y, name in zip(arrays, names):
        plt.plot(grid, y, label=name)

    plt.legend()
    plt.title("Gaussian Factorization RMSE by Split")
    plt.xlabel("K")
    plt.savefig("GF_RMSE.png", dpi=300)
    plt.show()

    # log predictive likelihood
    plt.plot(grid, ll_vector)

    # add labels + title (optional)
    plt.xlabel("K")
    plt.ylabel("Log Preditive Likelihood")
    plt.title("Gaussian Factorization Log Predictive Likelihood")

    # save to file
    plt.savefig("GF_LPL.png", dpi=300, bbox_inches="tight")

    # show on screen
    plt.show()


if __name__ == "__main__":
    main()