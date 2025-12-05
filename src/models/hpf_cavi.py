
import numpy as np
from dataclasses import dataclass
from src.evaluation.metrics import rmse
from typing import Optional

@dataclass
class HPF_CAVI_Config:
    n_factors: int = 20
    a: float = 0.3              # Shape for theta
    a_prime: float = 0.3        # Shape for xi (user rate prior)
    b_prime: float = 1.0        # Rate for xi
    c: float = 0.3              # Shape for beta
    c_prime: float = 0.3        # Shape for eta (item rate prior)
    d_prime: float = 1.0        # Rate for eta
    max_iter: int = 100
    tol: Optional[float] = 1e-4
    random_state: int = 42
    verbose: bool = True

class HPF_CAVI:
    """
    Hierarchical Poisson Factorization with CAVI updates on OBSERVED data only.
    
    Model:
        x_ui ~ Poisson(theta_u^T beta_i)
        theta_uk ~ Gamma(a, xi_u)
        xi_u ~ Gamma(a_prime, b_prime)
        beta_ik ~ Gamma(c, eta_i)
        eta_i ~ Gamma(c_prime, d_prime)
    """
    def __init__(self, config: HPF_CAVI_Config):
        self.config = config
        self.n_users = None
        self.n_items = None
        
        # Variational Parameters
        # Theta (Users)
        self.gamma_a_theta = None # Shape (N, K)
        self.gamma_b_theta = None # Rate (N, K)
        
        # Beta (Items)
        self.gamma_a_beta = None # Shape (M, K)
        self.gamma_b_beta = None # Rate (M, K)
        
        # Xi (User Priors)
        self.gamma_a_xi = None # Shape (N,)
        self.gamma_b_xi = None # Rate (N,)
        
        # Eta (Item Priors)
        self.gamma_a_eta = None # Shape (M,)
        self.gamma_b_eta = None # Rate (M,)
        
        # Expectations
        self.E_theta = None
        self.E_beta = None
        self.E_xi = None
        self.E_eta = None

    def _infer_dimensions(self, train_df):
        self.n_users = int(train_df["u"].max()) + 1
        self.n_items = int(train_df["i"].max()) + 1
        if self.config.verbose:
            print(f"Inferred n_users={self.n_users}, n_items={self.n_items}")

    def _initialize(self):
        rng = np.random.default_rng(self.config.random_state)
        K = self.config.n_factors
        N = self.n_users
        M = self.n_items
        
        # Initialize Theta (Users)
        self.gamma_a_theta = self.config.a + rng.gamma(1.0, 0.1, size=(N, K))
        self.gamma_b_theta = self.config.b_prime + rng.gamma(1.0, 0.1, size=(N, K)) # Initial guess
        
        # Initialize Beta (Items)
        self.gamma_a_beta = self.config.c + rng.gamma(1.0, 0.1, size=(M, K))
        self.gamma_b_beta = self.config.d_prime + rng.gamma(1.0, 0.1, size=(M, K)) # Initial guess
        
        # Initialize Xi (User Priors)
        self.gamma_a_xi = self.config.a_prime + K * self.config.a
        self.gamma_b_xi = self.config.b_prime * np.ones(N)
        
        # Initialize Eta (Item Priors)
        self.gamma_a_eta = self.config.c_prime + K * self.config.c
        self.gamma_b_eta = self.config.d_prime * np.ones(M)
        
        # Compute Expectations
        self._update_expectations()

    def _update_expectations(self):
        self.E_theta = self.gamma_a_theta / self.gamma_b_theta
        self.E_beta = self.gamma_a_beta / self.gamma_b_beta
        self.E_xi = self.gamma_a_xi / self.gamma_b_xi
        self.E_eta = self.gamma_a_eta / self.gamma_b_eta

    def _build_index_lists(self, user_ids, item_ids):
        user_to_obs = [[] for _ in range(self.n_users)]
        item_to_obs = [[] for _ in range(self.n_items)]

        for idx, (u, i) in enumerate(zip(user_ids, item_ids)):
            user_to_obs[u].append(idx)
            item_to_obs[i].append(idx)

        user_to_obs = [np.array(idx_list, dtype=int) for idx_list in user_to_obs]
        item_to_obs = [np.array(idx_list, dtype=int) for idx_list in item_to_obs]
        return user_to_obs, item_to_obs

    def fit(self, train_df, val_df=None):
        self._infer_dimensions(train_df)
        self._initialize()
        
        user_ids = train_df["u"].to_numpy(dtype=int)
        item_ids = train_df["i"].to_numpy(dtype=int)
        ratings = train_df["rating"].to_numpy(dtype=float)
        
        user_to_obs, item_to_obs = self._build_index_lists(user_ids, item_ids)
        
        prev_val_rmse = None
        
        for it in range(1, self.config.max_iter + 1):
            if self.config.verbose:
                print(f"\nHPF_CAVI iteration {it}/{self.config.max_iter}")
            
            # --- Update Theta (Users) ---
            for u in range(self.n_users):
                idx = user_to_obs[u]
                if idx.size == 0:
                    # No observations, revert to prior
                    self.gamma_a_theta[u] = self.config.a
                    self.gamma_b_theta[u] = self.E_xi[u]
                    continue
                
                i_indices = item_ids[idx]
                x_ui = ratings[idx]
                
                beta_subset = self.E_beta[i_indices] # (N_u, K)
                theta_u = self.E_theta[u] # (K,)
                
                rate_est = beta_subset @ theta_u
                rate_est[rate_est < 1e-10] = 1e-10
                
                # Allocation: x_ui * (theta_uk * beta_ik) / rate_est
                allocation = (x_ui[:, None] / rate_est[:, None]) * beta_subset * theta_u[None, :]
                
                # Shape: a + sum_i E[z_uik]
                self.gamma_a_theta[u] = self.config.a + np.sum(allocation, axis=0)
                
                # Rate: E[xi_u] + sum_{i in Omega_u} E[beta_ik]
                # CRITICAL: Sum only over OBSERVED items for this user
                self.gamma_b_theta[u] = self.E_xi[u] + np.sum(beta_subset, axis=0)
            
            self._update_expectations() # Update E[theta]
            
            # --- Update Xi (User Priors) ---
            # Shape is constant: a' + K*a
            # Rate: b' + sum_k E[theta_uk]
            self.gamma_b_xi = self.config.b_prime + np.sum(self.E_theta, axis=1)
            self._update_expectations() # Update E[xi]
            
            # --- Update Beta (Items) ---
            for i in range(self.n_items):
                idx = item_to_obs[i]
                if idx.size == 0:
                    self.gamma_a_beta[i] = self.config.c
                    self.gamma_b_beta[i] = self.E_eta[i]
                    continue
                
                u_indices = user_ids[idx]
                x_ui = ratings[idx]
                
                theta_subset = self.E_theta[u_indices] # (N_i, K)
                beta_i = self.E_beta[i] # (K,)
                
                rate_est = theta_subset @ beta_i
                rate_est[rate_est < 1e-10] = 1e-10
                
                allocation = (x_ui[:, None] / rate_est[:, None]) * theta_subset * beta_i[None, :]
                
                # Shape: c + sum_u E[z_uik]
                self.gamma_a_beta[i] = self.config.c + np.sum(allocation, axis=0)
                
                # Rate: E[eta_i] + sum_{u in Omega_i} E[theta_uk]
                # CRITICAL: Sum only over OBSERVED users for this item
                self.gamma_b_beta[i] = self.E_eta[i] + np.sum(theta_subset, axis=0)
            
            self._update_expectations() # Update E[beta]
            
            # --- Update Eta (Item Priors) ---
            # Shape is constant: c' + K*c
            # Rate: d' + sum_k E[beta_ik]
            self.gamma_b_eta = self.config.d_prime + np.sum(self.E_beta, axis=1)
            self._update_expectations() # Update E[eta]
            
            # --- Evaluation ---
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
            
            preds[valid_mask] = np.sum(theta * beta, axis=1)
            
        return preds

    def evaluate_rmse(self, df):
        y_true = df["rating"].to_numpy()
        y_pred = self.predict(df["u"].to_numpy(), df["i"].to_numpy())
        return rmse(y_true, y_pred)
