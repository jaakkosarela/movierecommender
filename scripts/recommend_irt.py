#!/usr/bin/env python3
"""Generate recommendations using IRT model with Thompson sampling."""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data_loader import RecommenderData
from src.irt_model import IRTConfig, IRTModel, fit_new_user
from src.recommendation import (
    ThompsonConfig,
    compute_user_predictions,
    add_imdb_metadata,
    generate_recommendations,
    print_recommendations,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate recommendations from trained IRT model"
    )
    parser.add_argument(
        "--model", type=str, default="models/irt_v1.pt", help="Path to trained model"
    )
    parser.add_argument(
        "--user-checkpoint",
        type=str,
        default="models/user_theta.pt",
        help="Path to user theta checkpoint",
    )
    parser.add_argument(
        "--top-n", type=int, default=30, help="Number of recommendations"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    # Thompson sampling parameters
    parser.add_argument(
        "--reference-votes",
        type=int,
        default=50000,
        help="Vote count at which uncertainty is fully trusted",
    )
    parser.add_argument(
        "--imdb-tolerance",
        type=float,
        default=3.0,
        help="Allow predictions up to this many points above IMDb",
    )
    parser.add_argument(
        "--imdb-penalty-weight",
        type=float,
        default=0.5,
        help="Weight for penalizing divergence from IMDb",
    )
    # Filtering options
    parser.add_argument(
        "--genre",
        type=str,
        action="append",
        dest="genres",
        help="Filter by genre (can specify multiple, e.g., --genre Thriller --genre Action)",
    )
    parser.add_argument(
        "--list-genres",
        action="store_true",
        help="List all available genres and exit",
    )
    # Output options
    parser.add_argument(
        "--show-details", action="store_true", help="Show prediction mean and std"
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data = RecommenderData().load_all(verbose=False)
    data.load_imdb_metadata()

    # List genres if requested
    if args.list_genres:
        all_genres = set()
        for genres_str in data.movies["genres"].dropna():
            for g in genres_str.split("|"):
                if g and g != "(no genres listed)":
                    all_genres.add(g)
        print("\nAvailable genres:")
        for genre in sorted(all_genres):
            print(f"  {genre}")
        return

    movie_id_to_idx = {mid: idx for idx, mid in enumerate(data.movie_ids)}
    movie_idx_to_id = dict(enumerate(data.movie_ids))

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, weights_only=False)
    config = IRTConfig(**checkpoint["config"])
    model = IRTModel(data.matrix.shape[0], data.matrix.shape[1], config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Build rated items set (needed for filtering predictions)
    user_ratings_dict = {}
    for _, row in data.user_ratings.iterrows():
        movie_id = int(row["movieId"])
        if movie_id in movie_id_to_idx:
            user_ratings_dict[movie_id_to_idx[movie_id]] = float(row["rating"])

    # Load user factors from checkpoint or fit from scratch
    user_checkpoint_path = Path(args.user_checkpoint)
    if user_checkpoint_path.exists():
        print(f"Loading user checkpoint from {args.user_checkpoint}...")
        user_ckpt = torch.load(args.user_checkpoint, map_location="cpu", weights_only=False)
        user_mu = user_ckpt["theta_mu"]
        user_log_std = user_ckpt["theta_log_std"]
        user_bias = torch.tensor(user_ckpt["bias_mu"])
        n_comp = user_ckpt.get("n_comparisons_used", 0)
        n_ratings = user_ckpt.get("n_ratings_used", 0)
        print(f"Loaded: {n_ratings} ratings, {n_comp} comparisons, bias={user_bias.item():.2f}")
    else:
        print(f"No checkpoint found. Fitting from {len(user_ratings_dict)} ratings...")
        user_mu, user_log_std, user_bias = fit_new_user(
            model, user_ratings_dict, n_iter=200, lr=0.1
        )

    # Compute predictions
    imdb_df = data.imdb_ratings.set_index("tconst")
    movies_df = data.movies.set_index("movieId")

    predictions = compute_user_predictions(
        model=model,
        user_mu=user_mu,
        user_log_std=user_log_std,
        user_bias=user_bias,
        rated_items=set(user_ratings_dict.keys()),
        movie_idx_to_id=movie_idx_to_id,
        imdb_df=imdb_df,
    )

    # Add IMDb metadata
    add_imdb_metadata(
        predictions=predictions,
        movie_idx_to_id=movie_idx_to_id,
        movieid_to_tconst=data.movieid_to_tconst,
        imdb_df=imdb_df,
    )

    # Configure Thompson sampling
    thompson_config = ThompsonConfig(
        reference_votes=args.reference_votes,
        imdb_tolerance=args.imdb_tolerance,
        imdb_penalty_weight=args.imdb_penalty_weight,
        seed=args.seed,
    )

    # Genre filter info
    if args.genres:
        print(f"Filtering by genres: {', '.join(args.genres)}")

    # Generate and print recommendations
    recommendations = generate_recommendations(
        predictions=predictions,
        config=thompson_config,
        movie_idx_to_id=movie_idx_to_id,
        movies_df=movies_df,
        top_n=args.top_n,
        genre_filter=args.genres,
    )

    print_recommendations(recommendations, show_details=args.show_details)


if __name__ == "__main__":
    main()
