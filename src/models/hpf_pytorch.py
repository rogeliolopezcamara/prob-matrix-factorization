
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from src.evaluation.metrics import rmse
import numpy as np

@dataclass
class HPF_PyTorch_Config:
    n_factors: int = 20
    a: float = 0.3
    a_prime: float = 1.0
    b_prime: float = 1.0
    c: float = 0.3
    c_prime: float = 1.0
    d_prime: float = 1.0
    lr: float = 0.001
    batch_size: int = 1024
    epochs: int = 20
    device: str = "cpu"
    verbose: bool = True

class HPF_PyTorch(nn.Module):
    def __init__(self, n_users, n_items, user_counts, item_counts, config: HPF_PyTorch_Config):
        super().__init__()
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        self.K = config.n_factors
        
        # Buffers for scaling
        # We add 1e-6 to avoid division by zero if any count is 0 (shouldn't happen in train)
        self.register_buffer("user_scale", 1.0 / (torch.tensor(user_counts, dtype=torch.float32) + 1e-6))
        self.register_buffer("item_scale", 1.0 / (torch.tensor(item_counts, dtype=torch.float32) + 1e-6))
        
        # Latent Factors (Unconstrained, will use Softplus)
        # Theta (Users): Shape (N, K)
        self.theta_uncons = nn.Parameter(torch.randn(n_users, self.K) * 0.1)
        
        # Beta (Items): Shape (M, K)
        self.beta_uncons = nn.Parameter(torch.randn(n_items, self.K) * 0.1)
        
        # User Rate Xi: Shape (N,)
        self.xi_uncons = nn.Parameter(torch.randn(n_users) * 0.1)
        
        # Item Rate Eta: Shape (M,)
        self.eta_uncons = nn.Parameter(torch.randn(n_items) * 0.1)
        
    @property
    def theta(self):
        return F.softplus(self.theta_uncons)
    
    @property
    def beta(self):
        return F.softplus(self.beta_uncons)
    
    @property
    def xi(self):
        return F.softplus(self.xi_uncons)
    
    @property
    def eta(self):
        return F.softplus(self.eta_uncons)

    def forward(self, user_ids, item_ids):
        theta_u = self.theta[user_ids]
        beta_i = self.beta[item_ids]
        return (theta_u * beta_i).sum(dim=1)

    def loss(self, user_ids, item_ids, ratings):
        # 1. Likelihood: Poisson(x | theta^T beta)
        # P(x|lambda) = lambda^x e^-lambda / x!
        # log P = x log lambda - lambda - log x!
        # We minimize negative log likelihood.
        # We can ignore log x! as it's constant wrt parameters.
        
        preds = self.forward(user_ids, item_ids)
        # Add epsilon to avoid log(0)
        preds = torch.clamp(preds, min=1e-6)
        
        # NLL = sum(lambda - x log lambda)
        nll = (preds - ratings * torch.log(preds)).sum()
        
        # 2. Priors (Regularization)
        # We need to sum log priors for ALL parameters, not just the batch.
        # However, for SGD, we usually scale the prior term by batch_size / total_size
        # OR we can just add the prior terms for the batch indices?
        # Standard practice for embedding regularization is to add it only for the batch.
        # But for the hierarchical structure (xi, eta), it's tricky.
        # Let's compute full prior loss and scale it? No, that's expensive.
        # Let's approximate by scaling.
        
        # Actually, for MAP with SGD, we typically add weight decay.
        # Here we have specific Gamma priors.
        # log Gamma(x | a, b) = a log b - log Gamma(a) + (a-1) log x - b x
        # We minimize - log P.
        # Term depends on x: - (a-1) log x + b x
        
        theta = self.theta[user_ids]
        beta = self.beta[item_ids]
        xi = self.xi[user_ids]
        eta = self.eta[item_ids]
        
        # Prior on Theta: Gamma(a, xi)
        # We treat xi as fixed for this term? No, xi is also learned.
        # But xi is user-specific.
        # In the batch, we have a subset of users.
        # The prior for theta_u depends on xi_u.
        # Loss_theta = sum_{u in batch} [ - (a-1) log theta_u + xi_u * theta_u ]
        # Note: We ignore constant terms wrt theta (a log xi - log Gamma(a)) if we are just optimizing theta?
        # No, we optimize xi too, so we need the terms involving xi.
        # Term involving xi in log P(theta|xi): a log xi - log Gamma(a)
        # So full term for u: a log xi_u - log Gamma(a) + (a-1) log theta_u - xi_u * theta_u
        # Neg Log Prior: - a log xi_u + log Gamma(a) - (a-1) log theta_u + xi_u * theta_u
        
        a = self.config.a
        c = self.config.c
        
        # We need to be careful about scaling. If a user appears multiple times in a batch,
        # we shouldn't penalize them multiple times?
        # Ideally we iterate over unique users in batch.
        
        # Prior Theta
        # Sum over K factors
        # - a log xi + xi * theta - (a-1) log theta
        # SCALE by user_scale[user_ids]
        
        u_scale = self.user_scale[user_ids] # (Batch,)
        i_scale = self.item_scale[item_ids] # (Batch,)
        
        # We compute the prior term for each user in the batch
        # And multiply by their scale factor.
        # Note: We don't need unique users anymore. We iterate over the batch.
        # If user u appears k times in batch, we add (prior_u * scale_u) k times.
        # Total sum over epoch = sum_{batches} sum_{u in batch} prior_u * (1/N_u)
        # = sum_{u} prior_u * (1/N_u) * (count of u in epoch)
        # = sum_{u} prior_u * (1/N_u) * N_u = sum_{u} prior_u.
        # This is exactly what we want!
        
        theta_batch = self.theta[user_ids] # (Batch, K)
        xi_batch = self.xi[user_ids]       # (Batch,)
        
        # Per-user prior loss (sum over K)
        loss_theta_batch = torch.sum(
            - a * torch.log(xi_batch.unsqueeze(1)) 
            + xi_batch.unsqueeze(1) * theta_batch 
            - (a - 1) * torch.log(theta_batch),
            dim=1
        ) # (Batch,)
        
        prior_theta = (loss_theta_batch * u_scale).sum()
        
        # Prior Beta
        beta_batch = self.beta[item_ids] # (Batch, K)
        eta_batch = self.eta[item_ids]   # (Batch,)
        
        loss_beta_batch = torch.sum(
            - c * torch.log(eta_batch.unsqueeze(1))
            + eta_batch.unsqueeze(1) * beta_batch
            - (c - 1) * torch.log(beta_batch),
            dim=1
        ) # (Batch,)
        
        prior_beta = (loss_beta_batch * i_scale).sum()
        
        # Prior Xi: Gamma(a', b')
        # - (a'-1) log xi + b' xi
        loss_xi_batch = (
            - (self.config.a_prime - 1) * torch.log(xi_batch)
            + self.config.b_prime * xi_batch
        )
        prior_xi = (loss_xi_batch * u_scale).sum()
        
        # Prior Eta: Gamma(c', d')
        loss_eta_batch = (
            - (self.config.c_prime - 1) * torch.log(eta_batch)
            + self.config.d_prime * eta_batch
        )
        prior_eta = (loss_eta_batch * i_scale).sum()
        
        # Total Loss
        loss = nll + prior_theta + prior_beta + prior_xi + prior_eta
        return loss

    def predict(self, user_ids, item_ids):
        # Handle numpy or torch inputs
        if isinstance(user_ids, np.ndarray):
            user_ids = torch.from_numpy(user_ids).long().to(self.theta.device)
        if isinstance(item_ids, np.ndarray):
            item_ids = torch.from_numpy(item_ids).long().to(self.beta.device)
            
        with torch.no_grad():
            preds = self.forward(user_ids, item_ids)
        return preds.cpu().numpy()
