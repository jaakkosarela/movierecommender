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
    parser.add_argument(
        "--mine",
        action="store_true",
        help="Show your rated movies ranked by model prediction (instead of recommendations)",
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

    imdb_df = data.imdb_ratings.set_index("tconst")
    movies_df = data.movies.set_index("movieId")

    # --mine mode: show rated movies/series ranked by prediction or rating
    if args.mine:
        from src.elicitation import ComparisonLogger, MovieSearcher
        import pandas as pd

        logger = ComparisonLogger()
        searcher = MovieSearcher()

        # Load comparison-based rating estimates first (newest ratings take precedence)
        rating_events = logger.load_ratings()
        rating_estimates = {}
        for event in rating_events:
            rating_estimates[event["tconst"]] = event["rating"]

        # Collect all items: rated movies in model, rated series, items from comparisons
        # Each item has: title, predicted, orig_rating, new_rating, type
        all_items = {}

        # 1. Rated movies in model (have predictions with uncertainty)
        with torch.no_grad():
            user_var = torch.exp(2 * user_log_std)

            for item_idx, orig_rating in user_ratings_dict.items():
                movie_id = movie_idx_to_id[item_idx]
                tconst = data.movieid_to_tconst.get(movie_id)
                if not tconst:
                    continue

                item_mu = model.item_mu[item_idx]
                item_var = torch.exp(2 * model.item_log_std[item_idx])

                # Mean prediction
                pred = (user_mu * item_mu).sum() + user_bias + model.item_bias_mu[item_idx] + model.global_mean
                pred_scaled = (pred.item() - 0.5) / 4.5 * 9 + 1  # MovieLens to 1-10

                # Uncertainty (propagate variance)
                var_rating = (user_var * item_var).sum()
                var_rating = var_rating + (user_var * item_mu**2).sum()
                var_rating = var_rating + (user_mu**2 * item_var).sum()
                std_scaled = (torch.sqrt(var_rating).item()) / 4.5 * 9

                title = movies_df.loc[movie_id, "title"] if movie_id in movies_df.index else f"Movie {movie_id}"

                all_items[tconst] = {
                    "title": title,
                    "predicted": pred_scaled,
                    "uncertainty": std_scaled,
                    "orig_rating": orig_rating,
                    "new_rating": rating_estimates.get(tconst),
                    "type": "",
                }

        # 2. Rated series and movies not in model (from raw user_ratings.csv)
        raw_ratings = pd.read_csv("data/user_ratings.csv")

        # Find column names
        tconst_col = next((c for c in ["Const", "tconst"] if c in raw_ratings.columns), None)
        rating_col = next((c for c in ["Your Rating", "rating"] if c in raw_ratings.columns), None)

        if tconst_col and rating_col:
            for _, row in raw_ratings.iterrows():
                tconst = row[tconst_col]
                if not tconst or tconst in all_items:
                    continue

                orig_rating = float(row[rating_col])

                # Look up in IMDb
                movie = searcher.get_by_tconst(tconst)
                if movie:
                    type_label = f"[{movie.type_label()}]" if movie.type_label() else ""
                    all_items[tconst] = {
                        "title": f"{movie.title} ({movie.year or '?'})",
                        "predicted": None,
                        "uncertainty": None,
                        "orig_rating": orig_rating,
                        "new_rating": rating_estimates.get(tconst),
                        "type": type_label,
                    }

        # 3. Items from comparisons only (no original rating)
        comparisons = logger.load_comparisons()
        for comp in comparisons:
            for key in ["movie_a", "movie_b"]:
                movie_data = comp.get(key, {})
                tconst = movie_data.get("tconst")
                if tconst and tconst not in all_items:
                    title = movie_data.get("title", tconst)
                    year = movie_data.get("year")
                    new_rating = rating_estimates.get(tconst)
                    if new_rating:
                        all_items[tconst] = {
                            "title": f"{title} ({year or '?'})",
                            "predicted": None,
                            "uncertainty": None,
                            "orig_rating": None,
                            "new_rating": new_rating,
                            "type": "[new]",
                        }

        # Sort by best available rating: pred > new > orig
        def sort_rating(item):
            return item["predicted"] or item["new_rating"] or item["orig_rating"] or 0

        items_list = list(all_items.values())
        items_list.sort(key=lambda x: -sort_rating(x))

        # ANSI codes for bold
        BOLD = "\033[1m"
        RESET = "\033[0m"

        # Thresholds for highlighting
        HIGH_UNCERTAINTY = 2.0  # ± above this is considered high
        BIG_DIFF = 1.5  # |diff| above this is considered big

        print(f"\nYour top {args.top_n} movies/series:")
        print("=" * 100)
        print(f"\n{'Rank':<4} {'Title':<42} {'Pred':>6} {'±':>5} {'New':>6} {'Orig':>6} {'Diff':>6}")
        print("-" * 90)

        for i, item in enumerate(items_list[:args.top_n], 1):
            title = item["title"]
            if item["type"]:
                title = f"{title} {item['type']}"
            title = title[:40] + "..." if len(title) > 42 else title

            # Effective rating for diff calculation
            eff_rating = item["new_rating"] or item["orig_rating"]
            uncertainty = item.get("uncertainty")
            diff = None

            if item["predicted"] is not None:
                pred_str = f"{item['predicted']:.1f}"
                if eff_rating:
                    diff = item["predicted"] - eff_rating
                    diff_str = f"{diff:+.1f}"
                else:
                    diff_str = "-"
            else:
                pred_str = "-"
                diff_str = "-"

            unc_str = f"{uncertainty:.1f}" if uncertainty is not None else "-"
            new_str = f"{item['new_rating']:.1f}" if item["new_rating"] else "-"
            orig_str = f"{item['orig_rating']:.1f}" if item["orig_rating"] else "-"

            # Check if line should be bold
            should_bold = False
            if uncertainty is not None and uncertainty >= HIGH_UNCERTAINTY:
                should_bold = True
            if diff is not None and abs(diff) >= BIG_DIFF:
                should_bold = True

            line = f"{i:<4} {title:<42} {pred_str:>6} {unc_str:>5} {new_str:>6} {orig_str:>6} {diff_str:>6}"
            if should_bold:
                print(f"{BOLD}{line}{RESET}")
            else:
                print(line)

        print("-" * 90)

        # Summary
        n_with_pred = sum(1 for x in items_list if x["predicted"] is not None)
        n_with_new = sum(1 for x in items_list if x["new_rating"] is not None)
        n_without = len(items_list) - n_with_pred
        print(f"\nTotal: {len(items_list)} items ({n_with_pred} with predictions, {n_with_new} with calibrated ratings)")
        return

    # Compute predictions

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
