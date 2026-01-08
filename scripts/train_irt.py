#!/usr/bin/env python3
"""Train IRT model on MovieLens data and generate recommendations."""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data_loader import RecommenderData
from src.irt_model import (
    IRTConfig,
    IRTModel,
    IRTTrainer,
    fit_new_user,
    initialize_with_svd,
)
from src.recommendation import (
    ThompsonConfig,
    compute_user_predictions,
    add_imdb_metadata,
    generate_recommendations,
    print_recommendations,
)


def main():
    parser = argparse.ArgumentParser(description="Train IRT recommendation model")
    parser.add_argument(
        "--n-factors", type=int, default=20, help="Number of latent factors"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=10000, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--no-svd-init", action="store_true", help="Skip SVD initialization"
    )
    parser.add_argument(
        "--top-n", type=int, default=50, help="Number of recommendations to show"
    )
    parser.add_argument(
        "--save-model", type=str, default=None, help="Path to save trained model"
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="Path to load pre-trained model"
    )
    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    data = RecommenderData().load_all()

    # Build index mappings: movie_ids array maps index -> movieId
    # We need movieId -> index for lookup
    movie_idx_to_id = dict(enumerate(data.movie_ids))  # {0: movieId_0, 1: movieId_1, ...}
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(data.movie_ids)}  # {movieId_0: 0, ...}

    if args.load_model:
        print(f"\nLoading pre-trained model from {args.load_model}...")
        checkpoint = torch.load(args.load_model)
        config = IRTConfig(**checkpoint["config"])
        model = IRTModel(data.matrix.shape[0], data.matrix.shape[1], config)
        model.load_state_dict(checkpoint["model_state"])
    else:
        # Configure model
        config = IRTConfig(
            n_factors=args.n_factors,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        # Create model
        print("\n" + "=" * 60)
        print("Creating IRT model...")
        print("=" * 60)
        model = IRTModel(
            n_users=data.matrix.shape[0],
            n_items=data.matrix.shape[1],
            config=config,
        )

        # Initialize with SVD
        if not args.no_svd_init:
            print("\nInitializing with SVD...")
            initialize_with_svd(model, data.matrix)

        # Train
        print("\n" + "=" * 60)
        print("Training model...")
        print("=" * 60)
        trainer = IRTTrainer(model, config)
        trainer.fit(data.matrix)

        # Save model if requested
        if args.save_model:
            print(f"\nSaving model to {args.save_model}...")
            torch.save(
                {
                    "config": {
                        "n_factors": config.n_factors,
                        "prior_scale_start": config.prior_scale_start,
                        "prior_scale_end": config.prior_scale_end,
                        "noise_std": config.noise_std,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "n_epochs": config.n_epochs,
                    },
                    "model_state": model.state_dict(),
                    "movie_ids": data.movie_ids.tolist(),
                    "user_ids": data.user_ids.tolist(),
                },
                args.save_model,
            )

    # Fit new user
    print("\n" + "=" * 60)
    print("Fitting user factors...")
    print("=" * 60)

    # Convert user ratings to item indices
    user_ratings_dict = {}
    for _, row in data.user_ratings.iterrows():
        movie_id = int(row["movieId"])
        if movie_id in movie_id_to_idx:
            item_idx = movie_id_to_idx[movie_id]
            user_ratings_dict[item_idx] = float(row["rating"])

    print(f"User has {len(user_ratings_dict)} ratings mapped to MovieLens")

    user_mu, user_log_std, user_bias = fit_new_user(
        model, user_ratings_dict, n_iter=200, lr=0.1
    )

    print(f"User bias: {user_bias.item():.3f}")
    print(f"User factor norms: {user_mu.norm().item():.3f}")

    # Generate recommendations
    print("\n" + "=" * 60)
    print("Generating recommendations...")
    print("=" * 60)

    # Load IMDb metadata
    print("\nLoading IMDb metadata for enrichment...")
    data.load_imdb_metadata()

    imdb_df = data.imdb_ratings.set_index("tconst")
    movies_df = data.movies.set_index("movieId")

    # Compute predictions using recommendation module
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

    # Use Thompson sampling for recommendations
    thompson_config = ThompsonConfig(
        reference_votes=50000,
        imdb_tolerance=3.0,
        imdb_penalty_weight=0.5,
    )

    recommendations = generate_recommendations(
        predictions=predictions,
        config=thompson_config,
        movie_idx_to_id=movie_idx_to_id,
        movies_df=movies_df,
        top_n=args.top_n,
    )

    print_recommendations(recommendations, show_details=True)

    # Show user's rated movies for context
    print(f"\n\nUser's highest rated movies (for context):")
    print("-" * 60)
    user_top = data.user_ratings.nlargest(10, "rating")
    for _, row in user_top.iterrows():
        print(f"  {row['rating']:.0f}/10  {row.get('Title', row['tconst'])}")


if __name__ == "__main__":
    main()
