#!/usr/bin/env python3
"""Analyze IRT model factors - show top/bottom movies for each factor."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.irt_model import IRTConfig, IRTModel


def main():
    parser = argparse.ArgumentParser(description="Analyze factor loadings")
    parser.add_argument(
        "--model", type=str, default="models/irt_v2_k50.pt", help="Path to trained model"
    )
    parser.add_argument(
        "--n-factors", type=int, default=20, help="Number of factors to show"
    )
    parser.add_argument(
        "--n-movies", type=int, default=5, help="Movies per factor (top and bottom)"
    )
    parser.add_argument(
        "--min-votes", type=int, default=10000, help="Minimum IMDb votes to include"
    )
    parser.add_argument(
        "--user-only", action="store_true", help="Only show user's rated movies"
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, weights_only=False)
    config = IRTConfig(**checkpoint["config"])
    movie_ids = checkpoint["movie_ids"]  # MovieLens movieIds in order
    user_ids = checkpoint["user_ids"]

    n_users = len(user_ids)
    n_items = len(movie_ids)
    model = IRTModel(n_users, n_items, config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Build movieId -> idx mapping
    movieid_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    # Load MovieLens links (movieId -> tconst)
    print("Loading MovieLens links...")
    links_df = pd.read_csv("data/ml-25m/links.csv")
    links_df["tconst"] = "tt" + links_df["imdbId"].astype(str).str.zfill(7)
    movieid_to_tconst = dict(zip(links_df["movieId"], links_df["tconst"]))
    tconst_to_movieid = {v: k for k, v in movieid_to_tconst.items()}

    # Load IMDb ratings for vote filtering
    print("Loading IMDb ratings...")
    imdb_df = pd.read_csv(
        "data/title.ratings.tsv",
        sep="\t",
        usecols=["tconst", "averageRating", "numVotes"],
    )
    imdb_ratings = imdb_df.set_index("tconst")

    # Load movie titles
    print("Loading movie titles...")
    movies_df = pd.read_csv("data/ml-25m/movies.csv")
    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    # Load user ratings
    user_ratings_df = pd.read_csv("data/user_ratings.csv")
    tconst_col = next((c for c in ["Const", "tconst"] if c in user_ratings_df.columns), None)
    if tconst_col:
        user_tconsts = set(user_ratings_df[tconst_col].dropna())
    else:
        user_tconsts = set()

    # Build item info with vote counts
    print("Building item info...")
    item_info = []
    for idx, movie_id in enumerate(movie_ids):
        tconst = movieid_to_tconst.get(movie_id)
        title = movieid_to_title.get(movie_id, f"Movie {movie_id}")

        votes = 0
        imdb_rating = None
        if tconst and tconst in imdb_ratings.index:
            votes = int(imdb_ratings.loc[tconst, "numVotes"])
            imdb_rating = float(imdb_ratings.loc[tconst, "averageRating"])

        is_rated = tconst in user_tconsts if tconst else False

        item_info.append({
            "idx": idx,
            "movie_id": movie_id,
            "tconst": tconst,
            "title": title,
            "votes": votes,
            "imdb_rating": imdb_rating,
            "is_rated": is_rated,
        })

    # Filter by vote count (unless user-only mode)
    if args.user_only:
        valid_items = [i for i in item_info if i["is_rated"]]
        print(f"Showing only user's {len(valid_items)} rated movies")
    else:
        valid_items = [i for i in item_info if i["votes"] >= args.min_votes]
        print(f"Filtered to {len(valid_items)} movies with >= {args.min_votes} votes")

    valid_indices = {i["idx"] for i in valid_items}

    # Get factor loadings
    item_mu = model.item_mu.detach()  # [n_items, K]
    prior_scales = model.prior_scales.detach()  # [K]

    # Sort factors by prior scale (most important first)
    factor_order = torch.argsort(prior_scales, descending=True)

    print()
    print("=" * 100)
    print(f"TOP {args.n_factors} FACTORS (by prior scale)")
    print("=" * 100)

    for rank, factor_idx in enumerate(factor_order[:args.n_factors]):
        k = factor_idx.item()
        prior = prior_scales[k].item()

        # Get loadings for this factor
        loadings = item_mu[:, k]

        # Sort all items by loading
        sorted_indices = torch.argsort(loadings, descending=True)

        # Filter to valid items
        top_items = []
        bottom_items = []

        for idx in sorted_indices:
            idx = idx.item()
            if idx in valid_indices:
                top_items.append(idx)
                if len(top_items) >= args.n_movies:
                    break

        for idx in reversed(sorted_indices.tolist()):
            if idx in valid_indices:
                bottom_items.append(idx)
                if len(bottom_items) >= args.n_movies:
                    break

        # Print factor header
        print()
        print(f"Factor {k} (prior={prior:.3f})")
        print("-" * 100)

        # Print top movies (highest loading)
        print(f"  HIGH β[{k}] (positive loading):")
        for idx in top_items:
            info = item_info[idx]
            loading = loadings[idx].item()
            rated_marker = " [YOU]" if info["is_rated"] else ""
            imdb_str = f"IMDb {info['imdb_rating']:.1f}" if info["imdb_rating"] else ""
            votes_str = f"{info['votes']//1000}K" if info["votes"] >= 1000 else str(info["votes"])
            print(f"    β={loading:+.3f}  {info['title'][:50]:<50} ({votes_str}, {imdb_str}){rated_marker}")

        print()

        # Print bottom movies (lowest loading)
        print(f"  LOW β[{k}] (negative loading):")
        for idx in bottom_items:
            info = item_info[idx]
            loading = loadings[idx].item()
            rated_marker = " [YOU]" if info["is_rated"] else ""
            imdb_str = f"IMDb {info['imdb_rating']:.1f}" if info["imdb_rating"] else ""
            votes_str = f"{info['votes']//1000}K" if info["votes"] >= 1000 else str(info["votes"])
            print(f"    β={loading:+.3f}  {info['title'][:50]:<50} ({votes_str}, {imdb_str}){rated_marker}")

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
