"""Item Response Theory model with variational inference for movie recommendations."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from tqdm import tqdm


@dataclass
class IRTConfig:
    """Configuration for IRT model."""

    n_factors: int = 20  # Latent dimensionality K
    prior_scale_start: float = 1.0  # σ_1 for first dimension
    prior_scale_end: float = 0.1  # σ_K for last dimension
    noise_std: float = 1.0  # Observation noise σ
    learning_rate: float = 0.01
    batch_size: int = 10000
    n_epochs: int = 20
    device: str = "cpu"


class IRTModel(nn.Module):
    """IRT latent factor model with variational inference.

    Generative model:
        θ_u ~ N(0, Σ_user)  for each user u
        β_m ~ N(0, Σ_item)  for each movie m
        b_u ~ N(0, σ_b²)    user bias
        b_m ~ N(0, σ_b²)    movie bias
        r_{u,m} ~ N(θ_u · β_m + b_u + b_m, σ²)

    Uses non-symmetric priors (decreasing variance per dimension) to break
    rotational invariance in the latent space.
    """

    def __init__(self, n_users: int, n_items: int, config: IRTConfig):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.config = config
        self.n_factors = config.n_factors

        # Non-symmetric prior scales (decreasing variance per dimension)
        # This breaks rotational invariance
        prior_scales = torch.linspace(
            config.prior_scale_start, config.prior_scale_end, config.n_factors
        )
        self.register_buffer("prior_scales", prior_scales)

        # Variational parameters for user factors: q(θ_u) = N(μ_θ, diag(σ_θ²))
        self.user_mu = nn.Parameter(torch.randn(n_users, config.n_factors) * 0.1)
        self.user_log_std = nn.Parameter(torch.zeros(n_users, config.n_factors) - 1.0)

        # Variational parameters for item factors: q(β_m) = N(μ_β, diag(σ_β²))
        self.item_mu = nn.Parameter(torch.randn(n_items, config.n_factors) * 0.1)
        self.item_log_std = nn.Parameter(torch.zeros(n_items, config.n_factors) - 1.0)

        # Variational parameters for biases
        self.user_bias_mu = nn.Parameter(torch.zeros(n_users))
        self.user_bias_log_std = nn.Parameter(torch.zeros(n_users) - 1.0)

        self.item_bias_mu = nn.Parameter(torch.zeros(n_items))
        self.item_bias_log_std = nn.Parameter(torch.zeros(n_items) - 1.0)

        # Global mean rating (learned)
        self.global_mean = nn.Parameter(torch.tensor(3.5))

        # Log observation noise (learned)
        self.log_noise_std = nn.Parameter(torch.tensor(math.log(config.noise_std)))

    @property
    def noise_std(self) -> torch.Tensor:
        return torch.exp(self.log_noise_std)

    def _kl_divergence_normal(
        self, mu: torch.Tensor, log_std: torch.Tensor, prior_scale: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence from q(z) = N(mu, exp(2*log_std)) to p(z) = N(0, prior_scale²).

        KL(q||p) = log(σ_p/σ_q) + (σ_q² + μ²)/(2σ_p²) - 1/2
        """
        var_q = torch.exp(2 * log_std)
        var_p = prior_scale**2
        kl = (
            torch.log(prior_scale)
            - log_std
            + (var_q + mu**2) / (2 * var_p)
            - 0.5
        )
        return kl.sum()

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all variational parameters."""
        kl = 0.0

        # KL for user factors (with non-symmetric prior)
        kl += self._kl_divergence_normal(
            self.user_mu, self.user_log_std, self.prior_scales
        )

        # KL for item factors (with non-symmetric prior)
        kl += self._kl_divergence_normal(
            self.item_mu, self.item_log_std, self.prior_scales
        )

        # KL for user biases (standard prior, scale=1)
        kl += self._kl_divergence_normal(
            self.user_bias_mu, self.user_bias_log_std, torch.tensor(1.0)
        )

        # KL for item biases (standard prior, scale=1)
        kl += self._kl_divergence_normal(
            self.item_bias_mu, self.item_bias_log_std, torch.tensor(1.0)
        )

        return kl

    def sample_factors(
        self, user_idx: torch.Tensor, item_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample latent factors using reparameterization trick."""
        # User factors
        user_mu = self.user_mu[user_idx]
        user_std = torch.exp(self.user_log_std[user_idx])
        user_factors = user_mu + user_std * torch.randn_like(user_mu)

        # Item factors
        item_mu = self.item_mu[item_idx]
        item_std = torch.exp(self.item_log_std[item_idx])
        item_factors = item_mu + item_std * torch.randn_like(item_mu)

        # User biases
        user_bias_mu = self.user_bias_mu[user_idx]
        user_bias_std = torch.exp(self.user_bias_log_std[user_idx])
        user_biases = user_bias_mu + user_bias_std * torch.randn_like(user_bias_mu)

        # Item biases
        item_bias_mu = self.item_bias_mu[item_idx]
        item_bias_std = torch.exp(self.item_bias_log_std[item_idx])
        item_biases = item_bias_mu + item_bias_std * torch.randn_like(item_bias_mu)

        return user_factors, item_factors, user_biases, item_biases

    def predict_rating(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """Predict ratings for given user-item pairs.

        Args:
            user_idx: User indices
            item_idx: Item indices
            sample: If True, sample from variational distribution; else use means
        """
        if sample:
            user_factors, item_factors, user_biases, item_biases = self.sample_factors(
                user_idx, item_idx
            )
        else:
            user_factors = self.user_mu[user_idx]
            item_factors = self.item_mu[item_idx]
            user_biases = self.user_bias_mu[user_idx]
            item_biases = self.item_bias_mu[item_idx]

        # Dot product of factors + biases + global mean
        pred = (user_factors * item_factors).sum(dim=1)
        pred = pred + user_biases + item_biases + self.global_mean

        return pred

    def log_likelihood(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        ratings: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Compute log likelihood of ratings under the model.

        Uses Monte Carlo estimate with n_samples.
        """
        total_ll = 0.0
        for _ in range(n_samples):
            pred = self.predict_rating(user_idx, item_idx, sample=True)
            # Gaussian log likelihood
            ll = -0.5 * ((ratings - pred) / self.noise_std) ** 2
            ll = ll - torch.log(self.noise_std) - 0.5 * math.log(2 * math.pi)
            total_ll = total_ll + ll.sum()

        return total_ll / n_samples

    def elbo(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        ratings: torch.Tensor,
        n_total_ratings: int,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Compute Evidence Lower Bound (ELBO).

        ELBO = E_q[log p(r|θ,β)] - KL(q||p)

        For mini-batch training, we scale the likelihood by n_total/n_batch.
        """
        batch_size = len(ratings)
        scale = n_total_ratings / batch_size

        # Scaled log likelihood
        ll = self.log_likelihood(user_idx, item_idx, ratings, n_samples) * scale

        # KL divergence (not scaled - computed once per batch)
        kl = self.kl_divergence()

        return ll - kl

    def predict_with_uncertainty(
        self, user_idx: int, item_idx: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict ratings with uncertainty estimates.

        Returns:
            means: Expected ratings
            stds: Standard deviations (uncertainty)
        """
        with torch.no_grad():
            user_idx_t = torch.tensor([user_idx] * len(item_idx))

            # Get variational parameters
            user_mu = self.user_mu[user_idx_t]
            user_var = torch.exp(2 * self.user_log_std[user_idx_t])
            item_mu = self.item_mu[item_idx]
            item_var = torch.exp(2 * self.item_log_std[item_idx])

            user_bias_mu = self.user_bias_mu[user_idx_t]
            item_bias_mu = self.item_bias_mu[item_idx]

            # Expected rating: E[θ·β] = μ_θ · μ_β (since θ and β independent)
            mean_rating = (user_mu * item_mu).sum(dim=1)
            mean_rating = mean_rating + user_bias_mu + item_bias_mu + self.global_mean

            # Variance of rating: Var[θ·β] = Σ_k (σ_θ[k]² σ_β[k]² + σ_θ[k]² μ_β[k]² + μ_θ[k]² σ_β[k]²)
            var_rating = (user_var * item_var).sum(dim=1)
            var_rating = var_rating + (user_var * item_mu**2).sum(dim=1)
            var_rating = var_rating + (user_mu**2 * item_var).sum(dim=1)

            # Add observation noise and bias uncertainty
            var_rating = var_rating + self.noise_std**2
            var_rating = var_rating + torch.exp(2 * self.user_bias_log_std[user_idx_t])
            var_rating = var_rating + torch.exp(2 * self.item_bias_log_std[item_idx])

            std_rating = torch.sqrt(var_rating)

            return mean_rating.numpy(), std_rating.numpy()


class IRTTrainer:
    """Trainer for IRT model using stochastic variational inference."""

    def __init__(self, model: IRTModel, config: IRTConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.history = {"elbo": [], "epoch": []}

    def _prepare_batches(
        self, ratings_coo: sparse.coo_matrix
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Prepare mini-batches from sparse ratings matrix."""
        n_ratings = ratings_coo.nnz
        indices = np.random.permutation(n_ratings)

        batches = []
        for start in range(0, n_ratings, self.config.batch_size):
            end = min(start + self.config.batch_size, n_ratings)
            batch_idx = indices[start:end]

            user_idx = torch.tensor(ratings_coo.row[batch_idx], dtype=torch.long)
            item_idx = torch.tensor(ratings_coo.col[batch_idx], dtype=torch.long)
            ratings = torch.tensor(ratings_coo.data[batch_idx], dtype=torch.float32)

            batches.append((user_idx, item_idx, ratings))

        return batches

    def train_epoch(
        self, ratings_coo: sparse.coo_matrix, n_samples: int = 1
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        n_total = ratings_coo.nnz
        batches = self._prepare_batches(ratings_coo)

        total_elbo = 0.0
        for user_idx, item_idx, ratings in batches:
            self.optimizer.zero_grad()

            elbo = self.model.elbo(user_idx, item_idx, ratings, n_total, n_samples)
            loss = -elbo  # Minimize negative ELBO

            loss.backward()
            self.optimizer.step()

            total_elbo += elbo.item()

        return total_elbo / len(batches)

    def fit(
        self,
        ratings_matrix: sparse.csr_matrix,
        verbose: bool = True,
    ) -> "IRTTrainer":
        """Fit the model using stochastic variational inference.

        Args:
            ratings_matrix: Sparse user-item rating matrix (CSR format)
            verbose: Print progress
        """
        # Convert to COO for efficient batch sampling
        ratings_coo = ratings_matrix.tocoo()

        if verbose:
            print(f"Training IRT model: {self.model.n_users:,} users, "
                  f"{self.model.n_items:,} items, {ratings_coo.nnz:,} ratings")
            print(f"Config: K={self.config.n_factors}, lr={self.config.learning_rate}, "
                  f"batch_size={self.config.batch_size}")

        epoch_iter = range(self.config.n_epochs)
        if verbose:
            epoch_iter = tqdm(epoch_iter, desc="Training")

        for epoch in epoch_iter:
            avg_elbo = self.train_epoch(ratings_coo)
            self.history["elbo"].append(avg_elbo)
            self.history["epoch"].append(epoch)

            if verbose:
                epoch_iter.set_postfix({"ELBO": f"{avg_elbo:.2f}"})

        return self


def initialize_with_svd(
    model: IRTModel,
    ratings_matrix: sparse.csr_matrix,
    n_components: Optional[int] = None,
) -> None:
    """Initialize model factors using truncated SVD (warm start).

    This provides a better starting point than random initialization.
    """
    from scipy.sparse.linalg import svds

    n_components = n_components or model.n_factors

    # Center the ratings (subtract global mean)
    data = ratings_matrix.copy()
    global_mean = data.data.mean()
    data.data = data.data - global_mean

    # Compute truncated SVD
    # Note: svds returns components in ascending order of singular values
    U, s, Vt = svds(data.astype(np.float32), k=min(n_components, min(data.shape) - 1))

    # Reverse to get descending order
    U = U[:, ::-1]
    s = s[::-1]
    Vt = Vt[::-1, :]

    # Scale by sqrt of singular values
    sqrt_s = np.sqrt(s)
    user_factors = U * sqrt_s
    item_factors = (Vt.T * sqrt_s).T

    # Pad if n_components < n_factors
    if n_components < model.n_factors:
        pad_u = np.zeros((model.n_users, model.n_factors - n_components))
        pad_i = np.zeros((model.n_factors - n_components, model.n_items))
        user_factors = np.hstack([user_factors, pad_u])
        item_factors = np.vstack([item_factors, pad_i])

    # Initialize model parameters
    with torch.no_grad():
        model.user_mu.copy_(torch.tensor(user_factors, dtype=torch.float32))
        model.item_mu.copy_(torch.tensor(item_factors.T, dtype=torch.float32))
        model.global_mean.copy_(torch.tensor(global_mean, dtype=torch.float32))

    print(f"Initialized with SVD: global_mean={global_mean:.3f}, "
          f"top singular values={s[:5]}")


def fit_new_user(
    model: IRTModel,
    user_ratings: dict[int, float],
    n_iter: int = 100,
    lr: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit latent factors for a new user given their ratings.

    This is used for cold-start: a new user with ratings but no trained factors.

    Args:
        model: Trained IRT model
        user_ratings: Dict mapping item_idx -> rating
        n_iter: Number of optimization iterations
        lr: Learning rate

    Returns:
        user_mu: Estimated user factor means
        user_log_std: Estimated user factor log-stds
    """
    model.eval()

    # Initialize new user's variational parameters
    user_mu = nn.Parameter(torch.zeros(model.n_factors))
    user_log_std = nn.Parameter(torch.zeros(model.n_factors) - 1.0)
    user_bias_mu = nn.Parameter(torch.tensor(0.0))
    user_bias_log_std = nn.Parameter(torch.tensor(-1.0))

    item_indices = torch.tensor(list(user_ratings.keys()), dtype=torch.long)
    ratings = torch.tensor(list(user_ratings.values()), dtype=torch.float32)

    optimizer = torch.optim.Adam([user_mu, user_log_std, user_bias_mu, user_bias_log_std], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()

        # Sample user factors
        user_std = torch.exp(user_log_std)
        user_factors = user_mu + user_std * torch.randn_like(user_mu)
        user_bias = user_bias_mu + torch.exp(user_bias_log_std) * torch.randn(1)

        # Get item factors (fixed from trained model)
        with torch.no_grad():
            item_factors = model.item_mu[item_indices]
            item_biases = model.item_bias_mu[item_indices]

        # Predict
        pred = (user_factors * item_factors).sum(dim=1)
        pred = pred + user_bias + item_biases + model.global_mean

        # Log likelihood
        ll = -0.5 * ((ratings - pred) / model.noise_std) ** 2
        ll = ll.sum()

        # KL divergence for user factors
        kl = model._kl_divergence_normal(user_mu.unsqueeze(0), user_log_std.unsqueeze(0), model.prior_scales)
        kl = kl + model._kl_divergence_normal(user_bias_mu.unsqueeze(0), user_bias_log_std.unsqueeze(0), torch.tensor(1.0))

        # ELBO
        elbo = ll - kl
        loss = -elbo

        loss.backward()
        optimizer.step()

    return user_mu.detach(), user_log_std.detach(), user_bias_mu.detach()
