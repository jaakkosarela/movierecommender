"""Thompson sampling for IRT model recommendations.

Implements squared log-vote shrinkage and soft IMDb floor to balance
exploration with quality control.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.irt_model import IRTModel


@dataclass
class ThompsonConfig:
    """Configuration for Thompson sampling."""

    reference_votes: int = 50000
    """Vote count at which uncertainty is fully trusted."""

    imdb_tolerance: float = 3.0
    """Allow predictions up to this many points above IMDb."""

    imdb_penalty_weight: float = 0.5
    """Weight for penalizing divergence from IMDb."""

    seed: Optional[int] = None
    """Random seed for reproducibility."""


@dataclass
class UserPredictions:
    """Predictions for a user on candidate items."""

    candidate_items: list[int]
    """Item indices for candidates (not yet rated)."""

    pred_mean: np.ndarray
    """Mean predicted rating for each candidate."""

    pred_std: np.ndarray
    """Standard deviation of predicted rating."""

    votes: np.ndarray
    """IMDb vote count for each candidate."""

    imdb_ratings: np.ndarray
    """IMDb average rating for each candidate."""


def compute_user_predictions(
    model: IRTModel,
    user_mu: torch.Tensor,
    user_log_std: torch.Tensor,
    user_bias: torch.Tensor,
    rated_items: set[int],
    movie_idx_to_id: dict[int, int],
    imdb_df: pd.DataFrame,
) -> UserPredictions:
    """Compute predictions for all unrated items.

    Args:
        model: Trained IRT model
        user_mu: User latent factor means
        user_log_std: User latent factor log-stds
        user_bias: User bias term
        rated_items: Set of item indices already rated
        movie_idx_to_id: Mapping from item index to MovieLens movieId
        imdb_df: IMDb ratings DataFrame indexed by tconst

    Returns:
        UserPredictions with means, stds, and metadata
    """
    model.eval()
    candidate_items = [i for i in range(model.n_items) if i not in rated_items]

    with torch.no_grad():
        candidate_idx = torch.tensor(candidate_items, dtype=torch.long)
        item_factors = model.item_mu[candidate_idx]
        item_biases = model.item_bias_mu[candidate_idx]
        item_log_stds = model.item_log_std[candidate_idx]

        # Mean prediction
        pred_mean = (
            (user_mu * item_factors).sum(dim=1)
            + user_bias
            + item_biases
            + model.global_mean
        )

        # Variance calculation
        user_var = torch.exp(2 * user_log_std)
        item_var = torch.exp(2 * item_log_stds)
        pred_var = (user_var * item_var).sum(dim=1)
        pred_var = pred_var + (user_var * item_factors**2).sum(dim=1)
        pred_var = pred_var + (user_mu**2 * item_var).sum(dim=1)
        pred_var = pred_var + model.noise_std**2
        pred_std = torch.sqrt(pred_var)

    pred_mean = pred_mean.numpy()
    pred_std = pred_std.numpy()

    # Get vote counts and IMDb ratings
    # Note: imdb_df should be indexed by tconst, and we need movieid_to_tconst mapping
    # For now, we pass imdb_df as (tconst -> row) and assume caller provides mapping
    votes = np.full(len(candidate_items), 100.0)
    imdb_ratings = np.full(len(candidate_items), 5.0)

    return UserPredictions(
        candidate_items=candidate_items,
        pred_mean=pred_mean,
        pred_std=pred_std,
        votes=votes,
        imdb_ratings=imdb_ratings,
    )


def add_imdb_metadata(
    predictions: UserPredictions,
    movie_idx_to_id: dict[int, int],
    movieid_to_tconst: dict[int, str],
    imdb_df: pd.DataFrame,
) -> None:
    """Add IMDb vote counts and ratings to predictions (in-place).

    Args:
        predictions: UserPredictions to update
        movie_idx_to_id: Mapping from item index to MovieLens movieId
        movieid_to_tconst: Mapping from movieId to IMDb tconst
        imdb_df: IMDb ratings DataFrame indexed by tconst
    """
    for i, item_idx in enumerate(predictions.candidate_items):
        movie_id = movie_idx_to_id[item_idx]
        tconst = movieid_to_tconst.get(movie_id)
        if tconst and tconst in imdb_df.index:
            predictions.votes[i] = imdb_df.loc[tconst, "numVotes"]
            predictions.imdb_ratings[i] = imdb_df.loc[tconst, "averageRating"]


def thompson_sample(
    predictions: UserPredictions,
    config: ThompsonConfig,
) -> np.ndarray:
    """Apply Thompson sampling with shrinkage and IMDb floor.

    Args:
        predictions: User predictions with means, stds, and metadata
        config: Thompson sampling configuration

    Returns:
        Final scores for each candidate item
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    # Squared log-vote shrinkage
    log_votes = np.log(np.maximum(predictions.votes, 10))
    log_reference = np.log(config.reference_votes)
    shrink_factor = np.clip((log_votes / log_reference) ** 2, 0, 1)
    effective_std = predictions.pred_std * shrink_factor

    # Thompson sampling
    z = np.random.randn(len(predictions.pred_mean))
    thompson_score = predictions.pred_mean + effective_std * z

    # Soft IMDb floor
    divergence = predictions.pred_mean - predictions.imdb_ratings
    penalty = np.maximum(0, divergence - config.imdb_tolerance) * config.imdb_penalty_weight
    final_score = thompson_score - penalty

    return final_score


@dataclass
class Recommendation:
    """A single movie recommendation."""

    rank: int
    item_idx: int
    movie_id: int
    title: str
    score: float
    pred_mean: float
    pred_std: float
    imdb_rating: float
    votes: int


def generate_recommendations(
    predictions: UserPredictions,
    config: ThompsonConfig,
    movie_idx_to_id: dict[int, int],
    movies_df: pd.DataFrame,
    top_n: int = 30,
    genre_filter: Optional[list[str]] = None,
) -> list[Recommendation]:
    """Generate ranked recommendations using Thompson sampling.

    Args:
        predictions: User predictions with means, stds, and metadata
        config: Thompson sampling configuration
        movie_idx_to_id: Mapping from item index to MovieLens movieId
        movies_df: Movies DataFrame indexed by movieId
        top_n: Number of recommendations to return
        genre_filter: If provided, only include movies matching ALL these genres

    Returns:
        List of Recommendation objects, sorted by score
    """
    scores = thompson_sample(predictions, config)
    sorted_idx = np.argsort(scores)[::-1]

    # Build genre mask if filtering
    if genre_filter:
        genre_filter_lower = [g.lower() for g in genre_filter]
        valid_mask = np.ones(len(predictions.candidate_items), dtype=bool)
        for i, item_idx in enumerate(predictions.candidate_items):
            movie_id = movie_idx_to_id[item_idx]
            if movie_id in movies_df.index:
                genres_str = movies_df.loc[movie_id, "genres"]
                if pd.isna(genres_str):
                    valid_mask[i] = False
                else:
                    movie_genres = [g.lower() for g in genres_str.split("|")]
                    # Movie must have ALL requested genres
                    if not all(g in movie_genres for g in genre_filter_lower):
                        valid_mask[i] = False
            else:
                valid_mask[i] = False
        # Filter sorted indices
        sorted_idx = [i for i in sorted_idx if valid_mask[i]]

    recommendations = []
    for rank, i in enumerate(sorted_idx[:top_n]):
        item_idx = predictions.candidate_items[i]
        movie_id = movie_idx_to_id[item_idx]

        title = (
            movies_df.loc[movie_id, "title"]
            if movie_id in movies_df.index
            else f"Movie {movie_id}"
        )

        recommendations.append(
            Recommendation(
                rank=rank + 1,
                item_idx=item_idx,
                movie_id=movie_id,
                title=title,
                score=scores[i],
                pred_mean=predictions.pred_mean[i],
                pred_std=predictions.pred_std[i],
                imdb_rating=predictions.imdb_ratings[i],
                votes=int(predictions.votes[i]),
            )
        )

    return recommendations


def print_recommendations(
    recommendations: list[Recommendation],
    show_details: bool = False,
) -> None:
    """Print recommendations to stdout.

    Args:
        recommendations: List of recommendations to print
        show_details: If True, show pred_mean and pred_std
    """
    print("=" * 100)
    print(f"TOP {len(recommendations)} RECOMMENDATIONS")
    print("=" * 100)

    if show_details:
        print(f"\n{'Rank':<4} {'Title':<45} {'Score':>6} {'Mean':>6} {'Std':>5} {'IMDb':>5} {'Votes':>10}")
        print("-" * 95)
    else:
        print(f"\n{'Rank':<4} {'Title':<50} {'Score':>6} {'IMDb':>5} {'Votes':>10}")
        print("-" * 85)

    for rec in recommendations:
        title = rec.title[:43] + "..." if len(rec.title) > 45 else rec.title

        if show_details:
            print(
                f"{rec.rank:<4} {title:<45} {rec.score:>6.2f} "
                f"{rec.pred_mean:>6.2f} {rec.pred_std:>5.2f} "
                f"{rec.imdb_rating:>5.1f} {rec.votes:>10,}"
            )
        else:
            title = rec.title[:48] + "..." if len(rec.title) > 50 else rec.title
            print(
                f"{rec.rank:<4} {title:<50} {rec.score:>6.2f} "
                f"{rec.imdb_rating:>5.1f} {rec.votes:>10,}"
            )

    print("-" * (95 if show_details else 85))

    # Vote distribution
    votes = [r.votes for r in recommendations]
    print(f"\nVote distribution:")
    print(f"  <1K:     {sum(1 for v in votes if v < 1000)}")
    print(f"  1K-10K:  {sum(1 for v in votes if 1000 <= v < 10000)}")
    print(f"  10K-50K: {sum(1 for v in votes if 10000 <= v < 50000)}")
    print(f"  >50K:    {sum(1 for v in votes if v >= 50000)}")
