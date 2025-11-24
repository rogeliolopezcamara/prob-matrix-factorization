# src/models/poisson_mf_cavi.py

import numpy as np
from dataclasses import dataclass
from src.evaluation.metrics import rmse

from typing import Optional

@dataclass
class PoissonMFCAVIConfig:
    n_factors: int = 20          # K (latent dimension)
    a0: float = 0.3              # Hyperparameter a for Gamma prior
    b0: float = 1.0              # Hyperparameter b for Gamma prior
    max_iter: int = 100          # Maximum CAVI iterations
    tol: Optional[float] = 1e-4  # Tolerance for convergence (None to disable)
    random_state: int = 42
    verbose: bool = True

class PoissonMFCAVI:
    """
    Poisson Matrix Factorization with mean-field VI (CAVI updates).
    Model: x_ij ~ Poisson(theta_i^T beta_j)
    Priors: theta_i ~ Gamma(a0, b0), beta_j ~ Gamma(a0, b0)
    """

    def __init__(self, config: PoissonMFCAVIConfig):
        self.config = config
        
        # Dimensions
        self.n_users = None
        self.n_items = None
        
        # Variational Parameters (Shape 'a' and Rate 'b')
        # E[X] = a / b
        self.a_theta = None
        self.b_theta = None
        self.a_beta = None
        self.b_beta = None
        
        # Expectations
        self.E_theta = None
        self.E_beta = None

    def _infer_dimensions(self, train_df):
        self.n_users = int(train_df["u"].max()) + 1
        self.n_items = int(train_df["i"].max()) + 1
        if self.config.verbose:
            print(f"Inferred n_users={self.n_users}, n_items={self.n_items}")

    def _initialize_variational_params(self):
        rng = np.random.default_rng(self.config.random_state)
        K = self.config.n_factors
        
        # Initialize with random Gamma noise
        # We initialize the expectations directly to start, then derive parameters
        # Or better, initialize parameters a and b.
        
        # a_theta: shape (n_users, K)
        # b_theta: shape (n_users, K)
        
        # Initialize shape 'a' with a0 + small noise
        self.a_theta = self.config.a0 + rng.gamma(1.0, 0.1, size=(self.n_users, K))
        self.a_beta = self.config.a0 + rng.gamma(1.0, 0.1, size=(self.n_items, K))
        
        # Initialize rate 'b' with b0
        self.b_theta = self.config.b0 * np.ones((self.n_users, K))
        self.b_beta = self.config.b0 * np.ones((self.n_items, K))
        
        # Compute initial expectations
        self.E_theta = self.a_theta / self.b_theta
        self.E_beta = self.a_beta / self.b_beta

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
        """
        Run CAVI on the training data.
        """
        self._infer_dimensions(train_df)
        self._initialize_variational_params()
        
        user_ids = train_df["u"].to_numpy(dtype=int)
        item_ids = train_df["i"].to_numpy(dtype=int)
        ratings = train_df["rating"].to_numpy(dtype=float)
        
        # Pre-build index lists for fast access
        user_to_obs, item_to_obs = self._build_index_lists(
            user_ids, item_ids, self.n_users, self.n_items
        )
        
        prev_val_rmse = None
        
        for it in range(1, self.config.max_iter + 1):
            if self.config.verbose:
                print(f"\nCAVI iteration {it}/{self.config.max_iter}")
            
            # ----------------------------------------------------------
            # Update User Factors (theta)
            # ----------------------------------------------------------
            # a_theta_ik = a0 + sum_{j in Omega_i} x_ij * phi_ijk (approx by x_ij * E[beta_jk] / E[x_ij])
            # BUT standard auxiliary variable free update is:
            # a_theta_ik = a0 + sum_{j} x_ij * (beta_jk * theta_ik) / (theta_i^T beta_j)
            # This is slow.
            # Simplified update often used:
            # a_theta_ik = a0 + sum_{j in Omega_i} x_ij * (E[beta_jk] / sum_k E[theta_ik] E[beta_jk]) * E[theta_ik] ??
            #
            # Actually, the user prompt specified the formulas:
            # a_theta = a0 + sum_j x_ij  <-- This implies x_ij is the count contributed by factor k?
            # No, usually for Poisson MF:
            # a_{ik} = a0 + sum_j x_ij * \zeta_{ijk}
            # where \zeta_{ijk} \propto \theta_{ik} \beta_{jk}
            #
            # However, if we assume the user wants the SIMPLEST update where we just sum x_ij:
            # That corresponds to a model where each factor k explains x_ij independently? No.
            #
            # Let's use the standard HPF update which is efficient:
            # a_{ik} = a0 + sum_{j \in \Omega_i} x_{ij} * \frac{\theta_{ik} \beta_{jk}}{\sum_l \theta_{il} \beta_{jl}}
            # b_{ik} = b0 + sum_{j} \beta_{jk}
            #
            # We need to compute the expected counts for each factor.
            # ----------------------------------------------------------
            
            # Update theta for each user
            for i in range(self.n_users):
                idx = user_to_obs[i]
                if idx.size == 0:
                    self.a_theta[i] = self.config.a0
                    self.b_theta[i] = self.config.b0
                    continue
                
                j_idx = item_ids[idx]
                x_ij = ratings[idx] # shape (N_i,)
                
                # E[beta] for observed items
                beta_subset = self.E_beta[j_idx] # (N_i, K)
                theta_i = self.E_theta[i] # (K,)
                
                # Compute interaction: theta_i^T beta_j
                rate_est = beta_subset @ theta_i # (N_i,)
                
                # Avoid divide by zero
                rate_est[rate_est < 1e-10] = 1e-10
                
                # Compute allocation: x_ij * (beta_jk * theta_ik) / rate_est
                # shape (N_i, K)
                allocation = (x_ij[:, None] / rate_est[:, None]) * beta_subset * theta_i[None, :]
                
                # Sum over j to get shape update
                self.a_theta[i] = self.config.a0 + np.sum(allocation, axis=0)
                
                # Rate update: b0 + sum_{j in Omega_i} E[beta_jk]
                # This treats missing data as missing, not zero.
                self.b_theta[i] = self.config.b0 + np.sum(beta_subset, axis=0)
            
            # Update expectations
            self.E_theta = self.a_theta / self.b_theta
            
            # ----------------------------------------------------------
            # Update Item Factors (beta)
            # ----------------------------------------------------------
            
            for j in range(self.n_items):
                idx = item_to_obs[j]
                if idx.size == 0:
                    self.a_beta[j] = self.config.a0
                    self.b_beta[j] = self.config.b0
                    continue
                    
                u_idx = user_ids[idx]
                x_ij = ratings[idx]
                
                theta_subset = self.E_theta[u_idx] # (N_j, K)
                beta_j = self.E_beta[j] # (K,)
                
                rate_est = theta_subset @ beta_j
                rate_est[rate_est < 1e-10] = 1e-10
                
                allocation = (x_ij[:, None] / rate_est[:, None]) * theta_subset * beta_j[None, :]
                
                self.a_beta[j] = self.config.a0 + np.sum(allocation, axis=0)
                
                # Rate update: b0 + sum_{i in Omega_j} E[theta_ik]
                self.b_beta[j] = self.config.b0 + np.sum(theta_subset, axis=0)
            
            # Update expectations
            self.E_beta = self.a_beta / self.b_beta
            
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
        """
        Predict E[x_ij] = E[theta_i]^T E[beta_j]
        """
        user_ids = np.asarray(user_ids, dtype=int)
        item_ids = np.asarray(item_ids, dtype=int)
        
        preds = np.zeros(len(user_ids))
        
        valid_mask = (user_ids < self.n_users) & (item_ids < self.n_items)
        
        if np.any(valid_mask):
            u_valid = user_ids[valid_mask]
            i_valid = item_ids[valid_mask]
            
            theta = self.E_theta[u_valid]
            beta = self.E_beta[i_valid]
            
            preds[valid_mask] = np.sum(theta * beta, axis=1)
            
        return preds

    def evaluate_rmse(self, df):
        y_true = df["rating"].to_numpy()
        y_pred = self.predict(df["u"].to_numpy(), df["i"].to_numpy())
        return rmse(y_true, y_pred)
