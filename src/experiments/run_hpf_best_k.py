# src/experiments/run_hpf_best_k.py

from src.data.load_data import load_all_splits
from src.models.hpf_cavi import HPF_CAVI, HPF_CAVI_Config
from src.evaluation.metrics import PoissonLogPredictiveLikelihood
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():

    # Load raw ratings (non-negative)
    # Note: Poisson assumes non-negative integers usually, but works for non-negative floats too.
    # Ratings are 0-5 (or 1-5).
    train_df, val_df, test_df = load_all_splits()

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    # Initialize vectors
    train_rmse_vector = []
    val_rmse_vector = []
    test_rmse_vector = []
    ll_vector = []

    # Check for negative values just in case
    if (train_df["rating"] < 0).any():
        raise ValueError("Ratings must be non-negative for Poisson MF.")

    # max number of latent dimensions
    K=60
    grid = range(2,K+1)

    for k in grid:
        config = HPF_CAVI_Config(
        n_factors=k,
        a=0.3,
        a_prime=1.0,
        b_prime=1.0,
        c=0.3,
        c_prime=1.0,
        d_prime=1.0,
        max_iter=100,
        verbose=False
        )   

        print(f"\nRunning HPF with k={k}:")
        print(config)

        model = HPF_CAVI(config)

        # Train
        model.fit(train_df, val_df=val_df)

        # Evaluate
        train_rmse = model.evaluate_rmse(train_df)
        val_rmse   = model.evaluate_rmse(val_df)
        test_rmse  = model.evaluate_rmse(test_df)
        ll = PoissonLogPredictiveLikelihood(test_df,model.E_theta,model.E_beta)

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
    plt.title("HPF RMSE by Split")
    plt.xlabel("K")
    plt.savefig("HPF_RMSE.png", dpi=300)
    plt.show()

    # plot log predictive likelihood
    plt.plot(grid, ll_vector)

    plt.xlabel("K")
    plt.ylabel("Log Preditive Likelihood")
    plt.title("HPF Log Predictive Likelihood")

    # save to file
    plt.savefig("HPF_LPL.png", dpi=300, bbox_inches="tight")

    # show on screen
    plt.show()

if __name__ == "__main__":
    main()
