#!/usr/bin/env python3
"""Update user latent factors (θ) from pairwise comparisons and new ratings.

This script:
1. Loads the main IRT model (β fixed)
2. Loads or creates user θ checkpoint
3. Loads new comparisons and ratings since last update
4. Optimizes θ using information-weighted likelihood
5. Saves updated θ checkpoint with new watermarks
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from src.irt_model import IRTModel, IRTConfig
from src.elicitation import ComparisonLogger


# Default uncertainty prior (in rating scale units, ~1-10)
DEFAULT_UNCERTAINTY = 1.0

# Bits of information in a rating (for normalization)
RATING_INFO_BITS = 2.5


@dataclass
class UserCheckpoint:
    """User's latent factor checkpoint."""

    theta_mu: torch.Tensor
    theta_log_std: torch.Tensor
    bias_mu: float
    bias_log_std: float
    comparisons_watermark: int
    ratings_watermark: Optional[str]
    n_comparisons_used: int
    n_ratings_used: int

    def save(self, path: str) -> None:
        """Save checkpoint to disk."""
        torch.save({
            "theta_mu": self.theta_mu,
            "theta_log_std": self.theta_log_std,
            "bias_mu": self.bias_mu,
            "bias_log_std": self.bias_log_std,
            "comparisons_watermark": self.comparisons_watermark,
            "ratings_watermark": self.ratings_watermark,
            "n_comparisons_used": self.n_comparisons_used,
            "n_ratings_used": self.n_ratings_used,
        }, path)
        print(f"Saved user checkpoint to {path}")

    @classmethod
    def load(cls, path: str) -> "UserCheckpoint":
        """Load checkpoint from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            theta_mu=data["theta_mu"],
            theta_log_std=data["theta_log_std"],
            bias_mu=data["bias_mu"],
            bias_log_std=data["bias_log_std"],
            comparisons_watermark=data["comparisons_watermark"],
            ratings_watermark=data["ratings_watermark"],
            n_comparisons_used=data["n_comparisons_used"],
            n_ratings_used=data["n_ratings_used"],
        )


def load_main_model(model_path: str) -> tuple[IRTModel, dict]:
    """Load the main IRT model with fixed item factors."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    config = checkpoint["config"]
    model_state = checkpoint["model_state"]
    movie_ids = checkpoint["movie_ids"]
    user_ids = checkpoint["user_ids"]

    n_users = len(user_ids)
    n_items = len(movie_ids)

    irt_config = IRTConfig(**config)
    model = IRTModel(n_users, n_items, irt_config)
    model.load_state_dict(model_state)
    model.eval()

    # Build movieId -> item_idx mapping
    movieid_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    print(f"Loaded model: {n_items:,} items, K={irt_config.n_factors}")

    return model, movieid_to_idx


def load_tconst_to_item_idx(
    links_path: str, movieid_to_idx: dict
) -> dict[str, int]:
    """Build tconst -> item_idx mapping."""
    import pandas as pd

    links_df = pd.read_csv(links_path)
    links_df["tconst"] = "tt" + links_df["imdbId"].astype(str).str.zfill(7)
    links_df["item_idx"] = links_df["movieId"].map(movieid_to_idx)
    links_df = links_df.dropna(subset=["item_idx"])
    links_df["item_idx"] = links_df["item_idx"].astype(int)

    return dict(zip(links_df["tconst"], links_df["item_idx"]))


def load_user_ratings(
    ratings_path: str, tconst_to_idx: dict
) -> tuple[dict[int, float], int]:
    """Load original IMDb ratings, return item_idx -> rating dict."""
    import pandas as pd

    df = pd.read_csv(ratings_path)

    # Find columns
    tconst_col = None
    rating_col = None
    for col in ["tconst", "imdb_id", "Const"]:
        if col in df.columns:
            tconst_col = col
            break
    for col in ["rating", "Your Rating"]:
        if col in df.columns:
            rating_col = col
            break

    if tconst_col is None or rating_col is None:
        raise ValueError("Cannot find tconst/rating columns in user ratings")

    item_ratings = {}
    for _, row in df.iterrows():
        tconst = row[tconst_col]
        if tconst in tconst_to_idx:
            item_idx = tconst_to_idx[tconst]
            rating = row[rating_col]
            # Scale from 1-10 to MovieLens 0.5-5 scale
            scaled_rating = (rating - 1) / 9 * 4.5 + 0.5
            item_ratings[item_idx] = scaled_rating

    return item_ratings, len(df)


def fit_user_with_comparisons(
    model: IRTModel,
    imdb_ratings: dict[int, float],
    elicitation_ratings: list[dict],
    comparisons: list[dict],
    tconst_to_idx: dict,
    init_checkpoint: Optional[UserCheckpoint] = None,
    n_iter: int = 200,
    lr: float = 0.05,
    verbose: bool = True,
) -> UserCheckpoint:
    """Fit user factors using ratings + comparisons with information weighting.

    Args:
        model: Trained IRT model (item factors fixed)
        imdb_ratings: Original ratings {item_idx: rating} (MovieLens scale)
        elicitation_ratings: New ratings from elicitation (1-10 scale)
        comparisons: Pairwise comparisons
        tconst_to_idx: Mapping from tconst to item index
        init_checkpoint: Previous checkpoint to initialize from
        n_iter: Optimization iterations
        lr: Learning rate
        verbose: Print progress

    Returns:
        Updated UserCheckpoint
    """
    model.eval()

    # Initialize user parameters
    if init_checkpoint is not None:
        user_mu = nn.Parameter(init_checkpoint.theta_mu.clone())
        user_log_std = nn.Parameter(init_checkpoint.theta_log_std.clone())
        user_bias_mu = nn.Parameter(torch.tensor(init_checkpoint.bias_mu))
        user_bias_log_std = nn.Parameter(torch.tensor(init_checkpoint.bias_log_std))
    else:
        user_mu = nn.Parameter(torch.zeros(model.n_factors))
        user_log_std = nn.Parameter(torch.zeros(model.n_factors) - 1.0)
        user_bias_mu = nn.Parameter(torch.tensor(0.0))
        user_bias_log_std = nn.Parameter(torch.tensor(-1.0))

    # Prepare IMDb ratings
    imdb_item_indices = torch.tensor(list(imdb_ratings.keys()), dtype=torch.long)
    imdb_rating_values = torch.tensor(list(imdb_ratings.values()), dtype=torch.float32)
    imdb_weights = torch.ones(len(imdb_ratings))  # Uniform weight for IMDb

    # Prepare elicitation ratings
    elic_item_indices = []
    elic_rating_values = []
    elic_weights = []

    for r in elicitation_ratings:
        tconst = r["tconst"]
        if tconst not in tconst_to_idx:
            continue
        item_idx = tconst_to_idx[tconst]
        rating = r["rating"]
        # Scale from 1-10 to MovieLens 0.5-5
        scaled_rating = (rating - 1) / 9 * 4.5 + 0.5
        # Weight by model uncertainty (or default)
        uncertainty = r.get("model_uncertainty") or DEFAULT_UNCERTAINTY
        # Normalize: higher uncertainty = more informative observation
        weight = min(uncertainty, 2.0)  # Cap to avoid extreme weights

        elic_item_indices.append(item_idx)
        elic_rating_values.append(scaled_rating)
        elic_weights.append(weight)

    if elic_item_indices:
        elic_item_indices = torch.tensor(elic_item_indices, dtype=torch.long)
        elic_rating_values = torch.tensor(elic_rating_values, dtype=torch.float32)
        elic_weights = torch.tensor(elic_weights, dtype=torch.float32)
    else:
        elic_item_indices = torch.tensor([], dtype=torch.long)
        elic_rating_values = torch.tensor([], dtype=torch.float32)
        elic_weights = torch.tensor([], dtype=torch.float32)

    # Prepare comparisons
    comp_item_a = []
    comp_item_b = []
    comp_choice = []  # 1 if chose A, 0 if chose B
    comp_weights = []

    for c in comparisons:
        tconst_a = c.get("movie_a", {}).get("tconst")
        tconst_b = c.get("movie_b", {}).get("tconst")
        choice = c.get("choice")

        if tconst_a not in tconst_to_idx or tconst_b not in tconst_to_idx:
            continue
        if choice not in ["a", "b"]:
            continue

        idx_a = tconst_to_idx[tconst_a]
        idx_b = tconst_to_idx[tconst_b]

        # Weight by entropy (stored in sampling info)
        entropy = c.get("sampling", {}).get("entropy") or DEFAULT_UNCERTAINTY
        # Normalize to rating scale: entropy is 0-1 bits, rating is ~2.5 bits
        weight = entropy / RATING_INFO_BITS

        comp_item_a.append(idx_a)
        comp_item_b.append(idx_b)
        comp_choice.append(1.0 if choice == "a" else 0.0)
        comp_weights.append(weight)

    if comp_item_a:
        comp_item_a = torch.tensor(comp_item_a, dtype=torch.long)
        comp_item_b = torch.tensor(comp_item_b, dtype=torch.long)
        comp_choice = torch.tensor(comp_choice, dtype=torch.float32)
        comp_weights = torch.tensor(comp_weights, dtype=torch.float32)
    else:
        comp_item_a = torch.tensor([], dtype=torch.long)
        comp_item_b = torch.tensor([], dtype=torch.long)
        comp_choice = torch.tensor([], dtype=torch.float32)
        comp_weights = torch.tensor([], dtype=torch.float32)

    if verbose:
        print(f"Fitting user factors:")
        print(f"  IMDb ratings: {len(imdb_ratings)}")
        print(f"  Elicitation ratings: {len(elic_item_indices)}")
        print(f"  Comparisons: {len(comp_item_a)}")

    # Optimizer
    optimizer = torch.optim.Adam(
        [user_mu, user_log_std, user_bias_mu, user_bias_log_std], lr=lr
    )

    # Training loop
    for iteration in range(n_iter):
        optimizer.zero_grad()

        # Sample user factors (reparameterization trick)
        user_std = torch.exp(user_log_std)
        user_factors = user_mu + user_std * torch.randn_like(user_mu)
        user_bias = user_bias_mu + torch.exp(user_bias_log_std) * torch.randn(1)

        total_ll = torch.tensor(0.0)

        # --- IMDb rating likelihood ---
        if len(imdb_item_indices) > 0:
            with torch.no_grad():
                item_factors = model.item_mu[imdb_item_indices]
                item_biases = model.item_bias_mu[imdb_item_indices]

            pred = (user_factors * item_factors).sum(dim=1)
            pred = pred + user_bias + item_biases + model.global_mean

            ll_imdb = -0.5 * ((imdb_rating_values - pred) / model.noise_std) ** 2
            ll_imdb = (ll_imdb * imdb_weights).sum()
            total_ll = total_ll + ll_imdb

        # --- Elicitation rating likelihood ---
        if len(elic_item_indices) > 0:
            with torch.no_grad():
                item_factors = model.item_mu[elic_item_indices]
                item_biases = model.item_bias_mu[elic_item_indices]

            pred = (user_factors * item_factors).sum(dim=1)
            pred = pred + user_bias + item_biases + model.global_mean

            ll_elic = -0.5 * ((elic_rating_values - pred) / model.noise_std) ** 2
            ll_elic = (ll_elic * elic_weights).sum()
            total_ll = total_ll + ll_elic

        # --- Comparison likelihood (Bradley-Terry) ---
        if len(comp_item_a) > 0:
            with torch.no_grad():
                factors_a = model.item_mu[comp_item_a]
                factors_b = model.item_mu[comp_item_b]
                biases_a = model.item_bias_mu[comp_item_a]
                biases_b = model.item_bias_mu[comp_item_b]

            pred_a = (user_factors * factors_a).sum(dim=1)
            pred_a = pred_a + user_bias + biases_a + model.global_mean
            pred_b = (user_factors * factors_b).sum(dim=1)
            pred_b = pred_b + user_bias + biases_b + model.global_mean

            # P(A > B) = sigmoid(pred_a - pred_b)
            log_p_a = torch.nn.functional.logsigmoid(pred_a - pred_b)
            log_p_b = torch.nn.functional.logsigmoid(pred_b - pred_a)

            # Bradley-Terry log likelihood
            ll_comp = comp_choice * log_p_a + (1 - comp_choice) * log_p_b
            ll_comp = (ll_comp * comp_weights).sum()
            total_ll = total_ll + ll_comp

        # --- KL divergence (regularization) ---
        kl = model._kl_divergence_normal(
            user_mu.unsqueeze(0), user_log_std.unsqueeze(0), model.prior_scales
        )
        kl = kl + model._kl_divergence_normal(
            user_bias_mu.unsqueeze(0),
            user_bias_log_std.unsqueeze(0),
            torch.tensor(1.0),
        )

        # ELBO
        elbo = total_ll - kl
        loss = -elbo

        loss.backward()
        optimizer.step()

        if verbose and (iteration + 1) % 50 == 0:
            print(f"  Iteration {iteration + 1}: ELBO = {elbo.item():.2f}")

    # Compute final watermarks
    logger = ComparisonLogger()
    final_comp_watermark = logger.get_max_comparison_number()
    final_rating_watermark = logger.get_max_rating_timestamp()

    n_comparisons = (
        (init_checkpoint.n_comparisons_used if init_checkpoint else 0)
        + len(comp_item_a)
    )
    n_ratings = len(imdb_ratings) + len(elic_item_indices)

    return UserCheckpoint(
        theta_mu=user_mu.detach(),
        theta_log_std=user_log_std.detach(),
        bias_mu=user_bias_mu.item(),
        bias_log_std=user_bias_log_std.item(),
        comparisons_watermark=final_comp_watermark,
        ratings_watermark=final_rating_watermark,
        n_comparisons_used=n_comparisons,
        n_ratings_used=n_ratings,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Update user factors from comparisons and ratings"
    )
    parser.add_argument(
        "--model",
        default="models/irt_v1.pt",
        help="Path to trained IRT model",
    )
    parser.add_argument(
        "--user-checkpoint",
        default="models/user_theta.pt",
        help="Path to user theta checkpoint",
    )
    parser.add_argument(
        "--user-ratings",
        default="data/user_ratings.csv",
        help="Path to original IMDb ratings",
    )
    parser.add_argument(
        "--links",
        default="data/ml-25m/links.csv",
        help="Path to MovieLens links.csv",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=200,
        help="Optimization iterations (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if no new data",
    )

    args = parser.parse_args()

    # Load model
    model, movieid_to_idx = load_main_model(args.model)

    # Build tconst mapping
    tconst_to_idx = load_tconst_to_item_idx(args.links, movieid_to_idx)
    print(f"Loaded {len(tconst_to_idx):,} tconst mappings")

    # Load original ratings
    imdb_ratings, n_imdb = load_user_ratings(args.user_ratings, tconst_to_idx)
    print(f"Loaded {len(imdb_ratings)} IMDb ratings (of {n_imdb} total)")

    # Load existing checkpoint if exists
    checkpoint_path = Path(args.user_checkpoint)
    init_checkpoint = None
    if checkpoint_path.exists():
        init_checkpoint = UserCheckpoint.load(args.user_checkpoint)
        print(f"Loaded existing checkpoint:")
        print(f"  Comparisons watermark: {init_checkpoint.comparisons_watermark}")
        print(f"  Ratings watermark: {init_checkpoint.ratings_watermark}")

    # Get new comparisons and ratings
    logger = ComparisonLogger()

    comp_watermark = init_checkpoint.comparisons_watermark if init_checkpoint else 0
    new_comparisons = logger.get_comparisons_after(comp_watermark)

    rating_watermark = init_checkpoint.ratings_watermark if init_checkpoint else None
    new_ratings = logger.get_ratings_after(rating_watermark)

    print(f"New comparisons: {len(new_comparisons)}")
    print(f"New elicitation ratings: {len(new_ratings)}")

    if not new_comparisons and not new_ratings and not args.force:
        print("No new data. Use --force to refit anyway.")
        return

    # Fit user factors
    checkpoint = fit_user_with_comparisons(
        model=model,
        imdb_ratings=imdb_ratings,
        elicitation_ratings=new_ratings,
        comparisons=new_comparisons if args.force else new_comparisons,
        tconst_to_idx=tconst_to_idx,
        init_checkpoint=init_checkpoint,
        n_iter=args.n_iter,
        lr=args.lr,
        verbose=True,
    )

    # Save checkpoint
    checkpoint.save(args.user_checkpoint)

    print(f"\nUser bias: {checkpoint.bias_mu:.2f}")
    print(f"Comparisons used: {checkpoint.n_comparisons_used}")
    print(f"Ratings used: {checkpoint.n_ratings_used}")


if __name__ == "__main__":
    main()
