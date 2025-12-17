#!/usr/bin/env python3
"""Get movie recommendations based on your ratings."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import RecommenderData
from src.similarity import find_similar_users_vectorized, get_user_ratings_dict
from src.recommender import Recommender


def main():
    parser = argparse.ArgumentParser(description="Get movie recommendations")
    parser.add_argument("-n", type=int, default=5, help="Number of recommendations (default: 5)")
    parser.add_argument("--pool", type=int, default=100, help="Candidate pool size (default: 100)")
    parser.add_argument("--min-overlap", type=int, default=10, help="Min overlapping movies for similarity (default: 10)")
    parser.add_argument("--weighting", choices=["linear", "uniform", "squared", "softmax"], default="linear")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show loading progress")
    args = parser.parse_args()

    # Load data
    data = RecommenderData()
    data.load_all(verbose=args.verbose)

    # Get user ratings and find similar users
    user_dict = get_user_ratings_dict(data.user_ratings)
    similar = find_similar_users_vectorized(
        user_dict, data.matrix, data.movie_to_idx,
        min_overlap=args.min_overlap, top_k=50
    )

    if not similar:
        print("Could not find similar users. Try lowering --min-overlap.")
        return 1

    # Get recommendations
    recommender = Recommender(data)
    recs = recommender.sample_recommendations(
        user_dict, similar,
        n=args.n,
        pool_size=args.pool,
        weighting=args.weighting,
    )

    # Print recommendations
    print("Do watch the following movie:")
    for rec in recs:
        title = rec["title"]
        genres = rec["genres"].replace("|", ", ")
        print(f"  {title}")
        print(f"    {genres}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
