# src/models/gaussian_mf_cavi_bias.py

import numpy as np
from dataclasses import dataclass

from src.evaluation.metrics import rmse


@dataclass
class GaussianMFCAVIConfig:
    n_factors: int = 10          # K (latent dimension)
    sigma2: float = 1.0          # observation noise variance σ²
    eta_theta2: float = 1.0      # prior variance for user factors η_θ²
    eta_beta2: float = 1.0       # prior variance for item factors η_β²
    eta_bias2: float = 1.0       # prior variance for biases η_b²
    max_iter: int = 20           # maximum CAVI iterations
    tol: float = 1e-3            # tolerance for validation RMSE improvement
    random_state: int = 42
    verbose: bool = True


class GaussianMFCAVI:
    """
    Gaussian Matrix Factorization with mean-field VI (CAVI updates).
    Model: r_ij ~ N(mu + b_i + b_j + theta_i^T beta_j, sigma^2)
    """

    def __init__(self, config: GaussianMFCAVIConfig):
        self.config = config

        # Will be set during fit
        self.n_users = None
        self.n_items = None
        self.m_theta = None    # shape (n_users, K)
        self.V_theta = None    # shape (n_users, K, K)
        self.m_beta = None     # shape (n_items, K)
        self.V_beta = None     # shape (n_items, K, K)
        
        # Biases
        self.m_user_bias = None # shape (n_users,)
        self.m_item_bias = None # shape (n_items,)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _infer_dimensions(self, train_df):
        self.n_users = int(train_df["u"].max()) + 1
        self.n_items = int(train_df["i"].max()) + 1
        if self.config.verbose:
            print(f"Inferred n_users={self.n_users}, n_items={self.n_items}")

    def _initialize_variational_params(self):
        rng = np.random.default_rng(self.config.random_state)
        K = self.config.n_factors

        # Means initialized with small Gaussian noise
        self.m_theta = 0.1 * rng.standard_normal((self.n_users, K))
        self.m_beta = 0.1 * rng.standard_normal((self.n_items, K))

        # Biases initialized to 0
        self.m_user_bias = np.zeros(self.n_users)
        self.m_item_bias = np.zeros(self.n_items)

        # Covariances initialized as identity matrices
        I_K = np.eye(K)
        self.V_theta = np.tile(I_K[None, :, :], (self.n_users, 1, 1))
        self.V_beta = np.tile(I_K[None, :, :], (self.n_items, 1, 1))

    def _build_index_lists(self, user_ids, item_ids, n_users, n_items):
        """
        Build lists of indices for Ω_i and Ω^j:
        - user_to_obs[i] = indices t where user_ids[t] == i
        - item_to_obs[j] = indices t where item_ids[t] == j
        """
        user_to_obs = [[] for _ in range(n_users)]
        item_to_obs = [[] for _ in range(n_items)]

        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            user_to_obs[u].append(idx)
            item_to_obs[i].append(idx)

        # Convert to numpy arrays for faster indexing
        user_to_obs = [np.array(idx_list, dtype=int) for idx_list in user_to_obs]
        item_to_obs = [np.array(idx_list, dtype=int) for idx_list in item_to_obs]

        return user_to_obs, item_to_obs

    # ------------------------------------------------------------------
    # CAVI core
    # ------------------------------------------------------------------
    def fit(self, train_df, val_df=None, global_mean=0.0):
        """
        Run CAVI on the training data.

        Args:
            train_df: DataFrame with columns ['u', 'i', 'rating'].
            val_df: optional DataFrame for monitoring validation RMSE.
        """
        # 1) Dimensions and initialization
        self.global_mean = global_mean
        self._infer_dimensions(train_df)
        self._initialize_variational_params()

        K = self.config.n_factors
        sigma2 = self.config.sigma2
        eta_theta2 = self.config.eta_theta2
        eta_beta2 = self.config.eta_beta2
        eta_bias2 = self.config.eta_bias2

        I_K = np.eye(K)

        # Training data arrays
        user_ids = train_df["u"].to_numpy(dtype=int)
        item_ids = train_df["i"].to_numpy(dtype=int)
        ratings = train_df["rating"].to_numpy(dtype=float)

        # Build Ω_i and Ω^j
        user_to_obs, item_to_obs = self._build_index_lists(
            user_ids, item_ids, self.n_users, self.n_items
        )

        prev_val_rmse = None

        # 2) CAVI iterations
        for it in range(1, self.config.max_iter + 1):
            if self.config.verbose:
                print(f"\nCAVI iteration {it}/{self.config.max_iter}")

            # ----------------------------------------------------------
            # Update q(θ_i) for all users i
            # ----------------------------------------------------------
            for i in range(self.n_users):
                idx = user_to_obs[i]
                if idx.size == 0:
                    continue  # no observations for this user

                j_idx = item_ids[idx]          # items rated by user i
                x_ij = ratings[idx]            # ratings (centered)
                
                # Subtract biases: effective rating = r_ij - b_i - b_j
                # Note: b_i is constant for this user, but we are updating theta_i
                # The residual for theta is (r_ij - b_i - b_j)
                b_i = self.m_user_bias[i]
                b_j = self.m_item_bias[j_idx]
                residual = x_ij - b_i - b_j

                beta_means = self.m_beta[j_idx]      # shape (n_i, K)
                beta_covs = self.V_beta[j_idx]       # shape (n_i, K, K)

                # E_q[β_j β_j^T] = V_βj + m_βj m_βj^T
                E_bbT = beta_covs + np.einsum(
                    "nk,nl->nkl", beta_means, beta_means
                )  # (n_i, K, K)

                S = E_bbT.sum(axis=0)  # Σ_j E[β_j β_j^T], shape (K, K)

                precision = (1.0 / eta_theta2) * I_K + (1.0 / sigma2) * S
                V_i = np.linalg.inv(precision)

                # Σ_j E[β_j] * residual
                weighted_sum = (beta_means * residual[:, None]).sum(axis=0)  # (K,)
                m_i = (1.0 / sigma2) * V_i @ weighted_sum

                self.m_theta[i] = m_i
                self.V_theta[i] = V_i

            # ----------------------------------------------------------
            # Update q(β_j) for all items j
            # ----------------------------------------------------------
            for j in range(self.n_items):
                idx = item_to_obs[j]
                if idx.size == 0:
                    continue  # no observations for this item

                u_idx = user_ids[idx]          # users who rated item j
                x_ij = ratings[idx]            # ratings
                
                # Subtract biases
                b_i = self.m_user_bias[u_idx]
                b_j = self.m_item_bias[j]
                residual = x_ij - b_i - b_j

                theta_means = self.m_theta[u_idx]    # shape (n_j, K)
                theta_covs = self.V_theta[u_idx]     # shape (n_j, K, K)

                # E_q[θ_i θ_i^T] = V_θi + m_θi m_θi^T
                E_ttT = theta_covs + np.einsum(
                    "nk,nl->nkl", theta_means, theta_means
                )  # (n_j, K, K)

                S = E_ttT.sum(axis=0)  # Σ_i E[θ_i θ_i^T], shape (K, K)

                precision = (1.0 / eta_beta2) * I_K + (1.0 / sigma2) * S
                V_j = np.linalg.inv(precision)

                # Σ_i E[θ_i] * residual
                weighted_sum = (theta_means * residual[:, None]).sum(axis=0)  # (K,)
                m_j = (1.0 / sigma2) * V_j @ weighted_sum

                self.m_beta[j] = m_j
                self.V_beta[j] = V_j
                
            # ----------------------------------------------------------
            # Update q(b_i) for all users i
            # ----------------------------------------------------------
            for i in range(self.n_users):
                idx = user_to_obs[i]
                if idx.size == 0:
                    continue
                
                j_idx = item_ids[idx]
                x_ij = ratings[idx]
                
                # Target for b_i: r_ij - b_j - theta_i^T beta_j
                b_j = self.m_item_bias[j_idx]
                
                # Interaction term E[theta_i^T beta_j] = m_theta_i^T m_beta_j
                theta_i = self.m_theta[i] # (K,)
                beta_j = self.m_beta[j_idx] # (n_i, K)
                interaction = beta_j @ theta_i # (n_i,)
                
                residual = x_ij - b_j - interaction
                
                # Precision lambda_bi = 1/eta_bias2 + N_i/sigma2
                N_i = len(idx)
                prec = (1.0 / eta_bias2) + (N_i / sigma2)
                var = 1.0 / prec
                
                # Mean mu_bi = var/sigma2 * sum(residual)
                mean = (var / sigma2) * residual.sum()
                
                self.m_user_bias[i] = mean

            # ----------------------------------------------------------
            # Update q(b_j) for all items j
            # ----------------------------------------------------------
            for j in range(self.n_items):
                idx = item_to_obs[j]
                if idx.size == 0:
                    continue
                
                u_idx = user_ids[idx]
                x_ij = ratings[idx]
                
                # Target for b_j: r_ij - b_i - theta_i^T beta_j
                b_i = self.m_user_bias[u_idx]
                
                # Interaction term
                theta_i = self.m_theta[u_idx] # (n_j, K)
                beta_j = self.m_beta[j] # (K,)
                interaction = theta_i @ beta_j # (n_j,)
                
                residual = x_ij - b_i - interaction
                
                # Precision lambda_bj = 1/eta_bias2 + N_j/sigma2
                N_j = len(idx)
                prec = (1.0 / eta_bias2) + (N_j / sigma2)
                var = 1.0 / prec
                
                # Mean mu_bj = var/sigma2 * sum(residual)
                mean = (var / sigma2) * residual.sum()
                
                self.m_item_bias[j] = mean

            # ----------------------------------------------------------
            # Monitor validation RMSE (optional)
            # ----------------------------------------------------------
            if val_df is not None:
                val_rmse = self.evaluate_rmse(val_df, self.global_mean)
                if self.config.verbose:
                    print(f"Validation RMSE: {val_rmse:.4f}")

                if prev_val_rmse is not None:
                    improvement = prev_val_rmse - val_rmse
                    if self.config.verbose:
                        print(f"Improvement: {improvement:.6f}")

                    if improvement >= 0 and improvement < self.config.tol:
                        if self.config.verbose:
                            print("Early stopping: small improvement on validation.")
                        break

                prev_val_rmse = val_rmse

        return self

    # ------------------------------------------------------------------
    # Prediction and evaluation
    # ------------------------------------------------------------------
    def predict(self, user_ids, item_ids, global_mean=0.0):
        """
        Predict ratings:
            centered_pred = b_i + b_j + m_θi^T m_βj
            final_pred = centered_pred + global_mean
        """

        user_ids = np.asarray(user_ids, dtype=int)
        item_ids = np.asarray(item_ids, dtype=int)

        valid_mask = (user_ids < self.n_users) & (item_ids < self.n_items)

        preds = np.zeros(len(user_ids), dtype=float)

        if np.any(valid_mask):
            u_valid = user_ids[valid_mask]
            i_valid = item_ids[valid_mask]
            
            theta = self.m_theta[u_valid]
            beta = self.m_beta[i_valid]
            b_i = self.m_user_bias[u_valid]
            b_j = self.m_item_bias[i_valid]
            
            preds[valid_mask] = b_i + b_j + np.sum(theta * beta, axis=1)

        return preds + global_mean

    def evaluate_rmse(self, df, global_mean):
        """
        Compute RMSE on original rating scale.
        Ignores unseen users/items.
        """
        mask = (df["u"] < self.n_users) & (df["i"] < self.n_items)
        df = df[mask]

        if df.empty:
            print("Warning: No valid (u,i) pairs.")
            return np.nan

        y_true = df["rating"].to_numpy(dtype=float) + global_mean
        y_pred = self.predict(df["u"].to_numpy(), df["i"].to_numpy(), global_mean)

        return rmse(y_true, y_pred)