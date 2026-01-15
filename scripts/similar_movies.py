#!/usr/bin/env python3
"""Find movies similar to a given movie based on factor loadings."""

import argparse

import pandas as pd
import torch

from src.elicitation import MovieSearcher, ModelInterface


def select_movie(searcher: MovieSearcher, query: str, year: int | None = None) -> str | None:
    """Interactive movie selection (same as elicit_preferences rate)."""
    current_limit = 10
    max_limit = 50
    year_filter = year

    while True:
        year_str = f" ({year_filter})" if year_filter else ""
        print(f"Searching for \"{query}\"{year_str}...")

        results = searcher.search(query, limit=current_limit, year=year_filter)

        if not results:
            if year_filter:
                print(f"No movies found matching \"{query}\" from {year_filter}.")
            else:
                print("No movies found.")
            return None

        # Show results
        print()
        for i, result in enumerate(results, 1):
            m = result.movie
            ml_status = "[in model]" if m.movielens_id else "[NOT IN MODEL]"
            type_label = f"[{m.type_label()}]" if m.type_label() else ""
            print(f"  [{i}] {m.title} ({m.year or '?'}) {type_label} {ml_status}".rstrip())
            if m.genres:
                print(f"      {', '.join(m.genres)}")

        # Get selection
        print()
        more_available = current_limit < max_limit and len(results) == current_limit
        if more_available:
            prompt = f"Select [1-{len(results)}, m=more, y=add year filter, q=quit]: "
        else:
            prompt = f"Select [1-{len(results)}, y=add year filter, q=quit]: "

        try:
            selection = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if selection == "q":
            return None
        elif selection == "m" and more_available:
            current_limit = min(current_limit + 10, max_limit)
            print(f"\nShowing up to {current_limit} results...")
            continue
        elif selection == "y":
            try:
                year_input = input("Enter year: ").strip()
                year_filter = int(year_input)
                current_limit = 10
                continue
            except (ValueError, EOFError, KeyboardInterrupt):
                print("Invalid year.")
                continue

        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(results):
                print("Invalid selection.")
                continue
            selected = results[idx].movie
            if not selected.movielens_id:
                print(f"Warning: {selected.title} is not in the model, can't find similar movies.")
                return None
            return selected.tconst
        except ValueError:
            print("Invalid selection.")
            continue


def main():
    parser = argparse.ArgumentParser(description="Find similar movies based on factor loadings")
    parser.add_argument("movie", help="Movie title to search for")
    parser.add_argument(
        "--year", type=int, help="Filter by release year"
    )
    parser.add_argument(
        "--model", type=str, default="models/irt_v2_k50.pt", help="Path to trained model"
    )
    parser.add_argument(
        "--n", type=int, default=20, help="Number of similar movies to show"
    )
    parser.add_argument(
        "--min-votes", type=int, default=10000, help="Minimum IMDb votes"
    )
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean"], default="cosine",
        help="Similarity metric"
    )
    args = parser.parse_args()

    # Initialize
    print("Loading model and data...")
    model_interface = ModelInterface(model_path=args.model)
    searcher = MovieSearcher()

    # Force load model and mappings
    _ = model_interface.get_predictions_for_rated_movies()

    # Select movie
    print()
    target_tconst = select_movie(searcher, args.movie, args.year)
    if not target_tconst:
        return

    target_movie = searcher.get_by_tconst(target_tconst)
    print(f"\nFinding movies similar to: {target_movie.title} ({target_movie.year})")

    # Get target's factor loadings
    irt_model = model_interface._model
    target_idx = model_interface._tconst_to_item_idx[target_tconst]
    target_beta = irt_model.item_mu[target_idx].detach()  # [K]

    # Load IMDb ratings for vote filtering
    print("Loading IMDb ratings...")
    imdb_df = pd.read_csv(
        "data/title.ratings.tsv",
        sep="\t",
        usecols=["tconst", "averageRating", "numVotes"],
    )
    imdb_ratings = imdb_df.set_index("tconst")

    # Load movie titles
    movies_df = pd.read_csv("data/ml-25m/movies.csv")
    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    # Get all item factors
    item_mu = irt_model.item_mu.detach()  # [n_items, K]
    movie_ids = model_interface._movie_ids

    # Build movieId -> tconst mapping
    links_df = pd.read_csv("data/ml-25m/links.csv")
    links_df["tconst"] = "tt" + links_df["imdbId"].astype(str).str.zfill(7)
    movieid_to_tconst = dict(zip(links_df["movieId"], links_df["tconst"]))

    # Compute similarities
    print("Computing similarities...")
    if args.metric == "cosine":
        # Normalize vectors
        target_norm = target_beta / target_beta.norm()
        item_norms = item_mu / item_mu.norm(dim=1, keepdim=True)
        similarities = (item_norms @ target_norm).numpy()
    else:  # euclidean
        distances = ((item_mu - target_beta) ** 2).sum(dim=1).sqrt().numpy()
        similarities = -distances  # negate so higher = more similar

    # Build results with filtering
    results = []
    for idx, movie_id in enumerate(movie_ids):
        if idx == target_idx:
            continue  # skip self

        tconst = movieid_to_tconst.get(movie_id)
        title = movieid_to_title.get(movie_id, f"Movie {movie_id}")

        # Vote filter
        votes = 0
        imdb_rating = None
        if tconst and tconst in imdb_ratings.index:
            votes = int(imdb_ratings.loc[tconst, "numVotes"])
            imdb_rating = float(imdb_ratings.loc[tconst, "averageRating"])

        if votes < args.min_votes:
            continue

        # Check if user rated
        is_rated = tconst in model_interface._user_ratings if model_interface._user_ratings else False

        results.append({
            "idx": idx,
            "movie_id": movie_id,
            "tconst": tconst,
            "title": title,
            "similarity": similarities[idx],
            "votes": votes,
            "imdb_rating": imdb_rating,
            "is_rated": is_rated,
        })

    # Sort by similarity
    results.sort(key=lambda x: -x["similarity"])

    # Print results
    print()
    print("=" * 100)
    print(f"TOP {args.n} MOVIES SIMILAR TO: {target_movie.title} ({target_movie.year})")
    print(f"Metric: {args.metric}, Min votes: {args.min_votes:,}")
    print("=" * 100)
    print()

    metric_label = "Cos" if args.metric == "cosine" else "Dist"
    print(f"{'Rank':<5} {metric_label:<8} {'Title':<55} {'Votes':>8} {'IMDb':>6}")
    print("-" * 90)

    for rank, r in enumerate(results[:args.n], 1):
        rated_marker = " [YOU]" if r["is_rated"] else ""
        votes_str = f"{r['votes']//1000}K" if r['votes'] >= 1000 else str(r['votes'])
        imdb_str = f"{r['imdb_rating']:.1f}" if r["imdb_rating"] else "-"
        sim_str = f"{r['similarity']:.3f}"

        title = r["title"][:52]
        if r["is_rated"]:
            title = title[:49] + "..."

        print(f"{rank:<5} {sim_str:<8} {title:<55} {votes_str:>8} {imdb_str:>6}{rated_marker}")

    print("-" * 90)

    # Also show target's top factor loadings
    print()
    print(f"Factor profile for {target_movie.title}:")
    abs_beta = torch.abs(target_beta)
    top_factors = abs_beta.argsort(descending=True)[:10]
    prior_scales = irt_model.prior_scales.detach()

    for factor_idx in top_factors:
        k = factor_idx.item()
        print(f"  Factor {k:2d}: β={target_beta[k].item():+.3f} (prior={prior_scales[k].item():.3f})")


if __name__ == "__main__":
    main()
