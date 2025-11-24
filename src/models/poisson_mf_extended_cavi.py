# src/models/poisson_mf_extended_cavi.py

import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.evaluation.metrics import rmse

@dataclass
class PoissonMFExtendedCAVIConfig:
    n_factors: int = 20
    a0: float = 0.3
    b0: float = 1.0
    max_iter: int = 100
    tol: Optional[float] = 1e-4
    random_state: int = 42
    verbose: bool = True

class PoissonMFExtendedCAVI:
    """
    Extended Poisson Matrix Factorization with mean-field VI (CAVI updates).
    Model: x_ij ~ Poisson(phi_u * psi_i * (theta_u^T beta_i))
    """

    def __init__(self, config: PoissonMFExtendedCAVIConfig):
        self.config = config
        
        self.n_users = None
        self.n_items = None
        
        # Latent Factors (K-dim)
        self.a_theta = None
        self.b_theta = None
        self.a_beta = None
        self.b_beta = None
        
        # Scalar Factors (1-dim)
        self.a_phi = None
        self.b_phi = None
        self.a_psi = None
        self.b_psi = None
        
        # Expectations
        self.E_theta = None
        self.E_beta = None
        self.E_phi = None
        self.E_psi = None

    def _infer_dimensions(self, train_df):
        self.n_users = int(train_df["u"].max()) + 1
        self.n_items = int(train_df["i"].max()) + 1
        if self.config.verbose:
            print(f"Inferred n_users={self.n_users}, n_items={self.n_items}")

    def _initialize_variational_params(self):
        rng = np.random.default_rng(self.config.random_state)
        K = self.config.n_factors
        
        # Initialize shape 'a' with a0 + noise
        self.a_theta = self.config.a0 + rng.gamma(1.0, 0.1, size=(self.n_users, K))
        self.a_beta = self.config.a0 + rng.gamma(1.0, 0.1, size=(self.n_items, K))
        self.a_phi = self.config.a0 + rng.gamma(1.0, 0.1, size=self.n_users)
        self.a_psi = self.config.a0 + rng.gamma(1.0, 0.1, size=self.n_items)
        
        # Initialize rate 'b' with b0
        self.b_theta = self.config.b0 * np.ones((self.n_users, K))
        self.b_beta = self.config.b0 * np.ones((self.n_items, K))
        self.b_phi = self.config.b0 * np.ones(self.n_users)
        self.b_psi = self.config.b0 * np.ones(self.n_items)
        
        # Compute initial expectations
        self.E_theta = self.a_theta / self.b_theta
        self.E_beta = self.a_beta / self.b_beta
        self.E_phi = self.a_phi / self.b_phi
        self.E_psi = self.a_psi / self.b_psi

    def _build_index_lists(self, user_ids, item_ids, n_users, n_items):
        user_to_obs = [[] for _ in range(n_users)]
        item_to_obs = [[] for _ in range(n_items)]

        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            user_to_obs[u].append(idx)
            item_to_obs[i].append(idx)

        user_to_obs = [np.array(idx_list, dtype=int) for idx_list in user_to_obs]
        item_to_obs = [np.array(idx_list, dtype=int) for idx_list in item_to_obs]

        return user_to_obs, item_to_obs

    def fit(self, train_df, val_df=None):
        self._infer_dimensions(train_df)
        self._initialize_variational_params()
        
        user_ids = train_df["u"].to_numpy(dtype=int)
        item_ids = train_df["i"].to_numpy(dtype=int)
        ratings = train_df["rating"].to_numpy(dtype=float)
        
        user_to_obs, item_to_obs = self._build_index_lists(
            user_ids, item_ids, self.n_users, self.n_items
        )
        
        prev_val_rmse = None
        
        for it in range(1, self.config.max_iter + 1):
            if self.config.verbose:
                print(f"\nCAVI iteration {it}/{self.config.max_iter}")
            
            # ----------------------------------------------------------
            # Update User Parameters (theta, phi)
            # ----------------------------------------------------------
            for u in range(self.n_users):
                idx = user_to_obs[u]
                if idx.size == 0:
                    # Priors
                    self.a_theta[u] = self.config.a0
                    self.b_theta[u] = self.config.b0
                    self.a_phi[u] = self.config.a0
                    self.b_phi[u] = self.config.b0
                    continue
                
                i_idx = item_ids[idx]
                x_ui = ratings[idx]
                
                # Gather item expectations
                beta_subset = self.E_beta[i_idx] # (N_u, K)
                psi_subset = self.E_psi[i_idx]   # (N_u,)
                
                # Current user expectations
                theta_u = self.E_theta[u] # (K,)
                phi_u = self.E_phi[u]     # scalar
                
                # Expected rate: phi_u * psi_i * (theta_u^T beta_i)
                dot_prod = beta_subset @ theta_u # (N_u,)
                rate_est = phi_u * psi_subset * dot_prod
                rate_est[rate_est < 1e-10] = 1e-10
                
                # --- Update theta_u ---
                # Allocation for theta: x_ui * (phi_u * psi_i * beta_ik * theta_uk) / rate_est
                # = x_ui * (beta_ik * theta_uk) / dot_prod
                alloc_theta = (x_ui[:, None] / dot_prod[:, None]) * beta_subset * theta_u[None, :]
                self.a_theta[u] = self.config.a0 + np.sum(alloc_theta, axis=0)
                
                # b_theta = b0 + sum_i E[psi_i] * E[beta_i]
                # Weighted sum of betas by psi
                weighted_beta = beta_subset * psi_subset[:, None] # (N_u, K)
                self.b_theta[u] = self.config.b0 + np.sum(weighted_beta, axis=0)
                
                # --- Update phi_u ---
                # Allocation for phi: x_ui * (phi_u * ...) / rate_est = x_ui
                # a_phi = a0 + sum_i x_ui
                self.a_phi[u] = self.config.a0 + np.sum(x_ui)
                
                # b_phi = b0 + sum_i E[psi_i] * (E[theta_u]^T E[beta_i])
                # = sum_i psi_i * dot_prod_new (using updated theta? No, standard CAVI uses old)
                # But we can use updated theta for faster convergence (Gauss-Seidel)
                # Let's use current E_theta (from previous step or updated? Standard is coordinate ascent, so use updated)
                # Update E_theta first
                self.E_theta[u] = self.a_theta[u] / self.b_theta[u]
                theta_u_new = self.E_theta[u]
                
                dot_prod_new = beta_subset @ theta_u_new
                self.b_phi[u] = self.config.b0 + np.sum(psi_subset * dot_prod_new)
                
                # Update E_phi
                self.E_phi[u] = self.a_phi[u] / self.b_phi[u]

            # ----------------------------------------------------------
            # Update Item Parameters (beta, psi)
            # ----------------------------------------------------------
            for i in range(self.n_items):
                idx = item_to_obs[i]
                if idx.size == 0:
                    self.a_beta[i] = self.config.a0
                    self.b_beta[i] = self.config.b0
                    self.a_psi[i] = self.config.a0
                    self.b_psi[i] = self.config.b0
                    continue
                
                u_idx = user_ids[idx]
                x_ui = ratings[idx]
                
                # Gather user expectations
                theta_subset = self.E_theta[u_idx] # (N_i, K)
                phi_subset = self.E_phi[u_idx]     # (N_i,)
                
                # Current item expectations
                beta_i = self.E_beta[i] # (K,)
                psi_i = self.E_psi[i]   # scalar
                
                # Expected rate
                dot_prod = theta_subset @ beta_i
                rate_est = phi_subset * psi_i * dot_prod
                rate_est[rate_est < 1e-10] = 1e-10
                
                # --- Update beta_i ---
                # Allocation: x_ui * (theta_uk * beta_ik) / dot_prod
                alloc_beta = (x_ui[:, None] / dot_prod[:, None]) * theta_subset * beta_i[None, :]
                self.a_beta[i] = self.config.a0 + np.sum(alloc_beta, axis=0)
                
                # b_beta = b0 + sum_u E[phi_u] * E[theta_u]
                weighted_theta = theta_subset * phi_subset[:, None]
                self.b_beta[i] = self.config.b0 + np.sum(weighted_theta, axis=0)
                
                # Update E_beta
                self.E_beta[i] = self.a_beta[i] / self.b_beta[i]
                beta_i_new = self.E_beta[i]
                
                # --- Update psi_i ---
                # a_psi = a0 + sum_u x_ui
                self.a_psi[i] = self.config.a0 + np.sum(x_ui)
                
                # b_psi = b0 + sum_u E[phi_u] * (E[theta_u]^T E[beta_i])
                dot_prod_new = theta_subset @ beta_i_new
                self.b_psi[i] = self.config.b0 + np.sum(phi_subset * dot_prod_new)
                
                # Update E_psi
                self.E_psi[i] = self.a_psi[i] / self.b_psi[i]

            # ----------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------
            if val_df is not None:
                val_rmse = self.evaluate_rmse(val_df)
                if self.config.verbose:
                    print(f"Validation RMSE: {val_rmse:.4f}")
                
                if prev_val_rmse is not None:
                    improvement = prev_val_rmse - val_rmse
                    if self.config.verbose:
                        print(f"Improvement: {improvement:.6f}")
                    
                    if self.config.tol is not None and improvement < self.config.tol:
                        if self.config.verbose:
                            print("Early stopping.")
                        break
                prev_val_rmse = val_rmse
                
        return self

    def predict(self, user_ids, item_ids):
        user_ids = np.asarray(user_ids, dtype=int)
        item_ids = np.asarray(item_ids, dtype=int)
        
        preds = np.zeros(len(user_ids))
        
        valid_mask = (user_ids < self.n_users) & (item_ids < self.n_items)
        
        if np.any(valid_mask):
            u_valid = user_ids[valid_mask]
            i_valid = item_ids[valid_mask]
            
            theta = self.E_theta[u_valid]
            beta = self.E_beta[i_valid]
            phi = self.E_phi[u_valid]
            psi = self.E_psi[i_valid]
            
            dot_prod = np.sum(theta * beta, axis=1)
            preds[valid_mask] = phi * psi * dot_prod
            
        return preds

    def evaluate_rmse(self, df):
        y_true = df["rating"].to_numpy()
        y_pred = self.predict(df["u"].to_numpy(), df["i"].to_numpy())
        return rmse(y_true, y_pred)
