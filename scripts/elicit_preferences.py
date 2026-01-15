#!/usr/bin/env python3
"""Preference elicitation CLI.

Two modes:
- calibrate: System asks about movies where model predictions differ from user ratings
- rate: User rates a movie they watched via pairwise comparisons
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch

from src.elicitation import (
    Movie,
    Comparison,
    Session,
    ComparisonChoice,
    RatingEstimate,
    ModelPrediction,
    UserRatings,
    SamplingInfo,
    ComparisonLogger,
    MaxEntropySampler,
    AdaptiveBinarySearchSampler,
    DiscrepancySampler,
    FactorUncertaintySampler,
    MovieSearcher,
    ModelInterface,
)


def print_header(title: str, subtitle: str = "") -> None:
    """Print a formatted header."""
    width = 65
    print()
    print("=" * width)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * width)
    print()


def print_comparison(round_num: int, total: str, movie_a: Movie, movie_b: Movie) -> None:
    """Print a comparison prompt."""
    print(f"Round {round_num} of {total}")
    print("-" * 65)
    type_a = f" [{movie_a.type_label()}]" if movie_a.type_label() else ""
    print(f"  [A] {movie_a.title} ({movie_a.year or '?'}){type_a}")
    if movie_a.genres:
        print(f"      {', '.join(movie_a.genres)}")
    print()
    type_b = f" [{movie_b.type_label()}]" if movie_b.type_label() else ""
    print(f"  [B] {movie_b.title} ({movie_b.year or '?'}){type_b}")
    if movie_b.genres:
        print(f"      {', '.join(movie_b.genres)}")
    print("-" * 65)


def get_choice() -> ComparisonChoice | str | None:
    """Get user's choice (a/b), skip (s), or quit (q).

    Returns:
        ComparisonChoice.A or .B for choice
        "skip" to skip this pair
        None to quit
    """
    while True:
        try:
            response = input("\nWhich do you prefer? [a/b/s=skip/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if response == "a":
            return ComparisonChoice.A
        elif response == "b":
            return ComparisonChoice.B
        elif response == "s":
            return "skip"
        elif response == "q":
            return None
        else:
            print("Please enter 'a', 'b', 's' to skip, or 'q' to quit")


def run_calibrate(args: argparse.Namespace) -> None:
    """Run calibration mode."""
    strategy_map = {
        "entropy": "max_entropy",
        "discrepancy": "discrepancy",
        "uncertainty": "factor_uncertainty",
    }
    strategy_name = strategy_map.get(args.strategy, args.strategy)
    print_header(
        "PREFERENCE CALIBRATION",
        f"Model: {args.model} | Strategy: {strategy_name} | Rounds: {args.n_rounds}"
    )

    # Initialize components
    model = ModelInterface(model_path=args.model)
    logger = ComparisonLogger()
    searcher = MovieSearcher()

    # Get model info
    model_info = model.get_model_info()
    print(f"Loaded model: {model_info.version}")

    # Get predictions for rated movies (anchors)
    predictions = model.get_predictions_for_rated_movies()
    print(f"Predictions for {len(predictions)} rated movies (anchors)")

    # Build Movie objects for rated movies
    rated_movies = []
    for tconst in predictions:
        movie = searcher.get_by_tconst(tconst)
        if movie:
            rated_movies.append(movie)

    # Build predicted ratings dict (just mean, for sampler)
    predicted_ratings = {tc: pred[0] for tc, pred in predictions.items()}

    # Find rated series (items not in model) to include as calibration targets
    all_rated_tconsts = model.get_rated_tconsts()
    target_items = []
    target_ratings = {}

    for tconst in all_rated_tconsts:
        if tconst in predictions:
            continue  # Already an anchor
        # This is a series or movie not in model
        item = searcher.get_by_tconst(tconst)
        if item and item.title_type in ("tvSeries", "tvMiniSeries"):
            user_rating = model.get_user_rating(tconst)
            if user_rating is not None:
                target_items.append(item)
                target_ratings[tconst] = user_rating

    if target_items:
        print(f"Including {len(target_items)} rated series as calibration targets")

    # Only exclude pairs that haven't been used for factor updates yet
    # (pairs before the watermark have been incorporated - could be asked again)
    user_checkpoint_path = Path("models/user_theta.pt")
    if user_checkpoint_path.exists():
        user_ckpt = torch.load(user_checkpoint_path, map_location="cpu", weights_only=False)
        watermark = user_ckpt.get("comparisons_watermark", 0)
    else:
        watermark = 0

    # Get only comparisons after the watermark (not yet incorporated)
    unprocessed_comparisons = logger.get_comparisons_after(watermark)
    exclude_pairs = set()
    for comp in unprocessed_comparisons:
        tc_a = comp.get("movie_a", {}).get("tconst")
        tc_b = comp.get("movie_b", {}).get("tconst")
        if tc_a and tc_b:
            exclude_pairs.add((tc_a, tc_b))
            exclude_pairs.add((tc_b, tc_a))  # both directions

    if exclude_pairs:
        print(f"Excluding {len(exclude_pairs) // 2} unprocessed pairs (run update_factors to incorporate)")

    # Initialize sampler based on strategy
    if args.strategy == "discrepancy":
        # For discrepancy mode, we need actual ratings
        actual_ratings = {}
        for tconst in predictions:
            rating = model.get_user_rating(tconst)
            if rating is not None:
                actual_ratings[tconst] = rating

        sampler = DiscrepancySampler(
            rated_movies=rated_movies,
            predicted_ratings=predicted_ratings,
            actual_ratings=actual_ratings,
            exclude_pairs=exclude_pairs,
            min_discrepancy=1.0,  # Items with |pred - actual| >= 1.0
        )
        print(f"Discrepancy mode: targeting items where model is confidently wrong")

    elif args.strategy == "uncertainty":
        # Factor uncertainty mode - needs model internals
        actual_ratings = {}
        for tconst in predictions:
            rating = model.get_user_rating(tconst)
            if rating is not None:
                actual_ratings[tconst] = rating

        # Access model internals for factor-based sampling
        irt_model = model._model
        user_mu, user_log_std, user_bias = model._user_factors

        sampler = FactorUncertaintySampler(
            rated_movies=rated_movies,
            predicted_ratings=predicted_ratings,
            actual_ratings=actual_ratings,
            item_mu=irt_model.item_mu,
            tconst_to_idx=model._tconst_to_item_idx,
            user_log_std=user_log_std,
            prior_scales=irt_model.prior_scales,
            exclude_pairs=exclude_pairs,
        )
        print(f"Factor uncertainty mode: targeting pairs that reduce θ uncertainty")

    else:
        # Default: max entropy
        sampler = MaxEntropySampler(
            rated_movies=rated_movies,
            predicted_ratings=predicted_ratings,
            exclude_pairs=exclude_pairs,
            target_items=target_items,
            target_ratings=target_ratings,
        )

    # Create session
    session = Session(
        use_case="calibrate",
        started_at=datetime.now(),
        model_version=model_info.version,
        sampling_strategy=strategy_name,
    )

    print(f"\nSession: {session.id}")
    n_targets = len(rated_movies) + len(target_items)
    if target_items:
        print(f"Starting calibration with {len(rated_movies)} movies + {len(target_items)} series...")
    else:
        print(f"Starting calibration with {len(rated_movies)} rated movies...")
    print()

    # Run comparison loop
    round_num = 0
    skipped = 0
    while round_num < args.n_rounds:
        pair = sampler.sample_pair()
        if pair is None:
            print("No more pairs available.")
            break

        round_num += 1
        print_comparison(
            round_num,
            str(args.n_rounds),
            pair.movie_a,
            pair.movie_b,
        )

        prob = pair.prediction.prob_a_wins
        choice = get_choice()
        if choice is None:
            print("\nSession ended early.")
            break

        if choice == "skip":
            # Mark pair as used but don't log, don't count toward rounds
            sampler.mark_used(pair.movie_a.tconst, pair.movie_b.tconst)
            round_num -= 1  # Don't count skipped pairs
            skipped += 1
            print("Skipped.")
            print()
            continue

        # Get user's actual ratings
        rating_a = model.get_user_rating(pair.movie_a.tconst)
        rating_b = model.get_user_rating(pair.movie_b.tconst)

        # Create comparison record
        comparison = Comparison(
            movie_a=pair.movie_a,
            movie_b=pair.movie_b,
            choice=choice,
            timestamp=datetime.now(),
            session_id=session.id,
            round_num=round_num,
            use_case="calibrate",
            model_prediction=ModelPrediction(
                prob_a_wins=pair.prediction.prob_a_wins,
                entropy=pair.prediction.entropy,
                rating_a=pair.prediction.rating_a,
                rating_b=pair.prediction.rating_b,
                model_version=model_info.version,
            ),
            user_ratings=UserRatings(rating_a=rating_a, rating_b=rating_b),
            sampling=SamplingInfo(
                strategy=strategy_name,
                entropy=pair.prediction.entropy,
            ),
        )

        # Log comparison
        logger.log_comparison(comparison)
        session.comparisons.append(comparison)
        session.n_comparisons += 1

        # Mark pair as used
        sampler.mark_used(pair.movie_a.tconst, pair.movie_b.tconst)

        # Feedback
        winner = "A" if choice == ComparisonChoice.A else "B"
        correct = (choice == ComparisonChoice.A and prob > 0.5) or (choice == ComparisonChoice.B and prob < 0.5)
        status = "aligned" if correct else "surprised"
        print(f"Logged: chose {winner}, model was {status}")
        print()

    # Finalize session
    session.ended_at = datetime.now()
    logger.log_session(session)

    print("=" * 65)
    skip_msg = f" ({skipped} skipped)" if skipped else ""
    print(f"Session complete! {session.n_comparisons} comparisons logged.{skip_msg}")
    print(f"Session ID: {session.id}")
    print("=" * 65)


def run_rate(args: argparse.Namespace) -> None:
    """Run rating mode."""
    query = args.movie
    print_header(
        "RATE A MOVIE",
        f"~{args.max_rounds} comparisons to estimate your rating"
    )

    # Initialize components
    model = ModelInterface(model_path=args.model)
    logger = ComparisonLogger()
    searcher = MovieSearcher()

    # Search for the movie
    year_filter = getattr(args, 'year', None)
    year_str = f" ({year_filter})" if year_filter else ""
    print(f"Searching for \"{query}\"{year_str}...")

    # Search with increasing limits until user finds their movie
    current_limit = 10
    max_limit = 50
    results = []
    target_movie = None

    while target_movie is None:
        results = searcher.search(query, limit=current_limit, year=year_filter)

        if not results:
            if year_filter:
                print(f"No movies found matching \"{query}\" from {year_filter}.")
            else:
                print("No movies found.")
            return

        # Show results
        print()
        for i, result in enumerate(results, 1):
            m = result.movie
            ml_status = "[in model]" if m.movielens_id else ""
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
            return

        if selection == "q":
            return
        elif selection == "m" and more_available:
            current_limit = min(current_limit + 10, max_limit)
            print(f"\nShowing up to {current_limit} results...")
            continue
        elif selection == "y":
            try:
                year_input = input("Enter year: ").strip()
                year_filter = int(year_input)
                print(f"\nSearching for \"{query}\" ({year_filter})...")
                current_limit = 10  # Reset limit
                continue
            except (ValueError, EOFError, KeyboardInterrupt):
                print("Invalid year.")
                continue

        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(results):
                print("Invalid selection.")
                continue
            target_movie = results[idx].movie
        except ValueError:
            print("Invalid selection.")
            continue
    print(f"\nSelected: {target_movie.title} ({target_movie.year})")

    # Check if already rated
    model_info = model.get_model_info()
    existing_rating = model.get_user_rating(target_movie.tconst)
    if existing_rating is not None:
        print(f"You already rated this: {existing_rating}/10")
        try:
            response = input("Re-rate anyway? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if response != "y":
            return

    # Check if in model
    model_pred = None
    model_uncertainty = None
    if model.is_in_model(target_movie.tconst):
        pred = model.get_prediction(target_movie.tconst)
        if pred:
            model_pred, model_uncertainty = pred
            print(f"Model prediction: {model_pred:.1f} ± {model_uncertainty:.1f}")

    # Get anchor movies (exclude target to avoid self-comparison)
    # Use model predictions as anchor ratings, with uncertainties
    predictions = model.get_predictions_for_rated_movies()

    # Also include comparison-rated items that are in the model
    comparison_ratings = logger.get_current_ratings()

    anchor_movies = []
    anchor_ratings = {}  # tconst -> model prediction (calibrated rating)
    anchor_uncertainties = {}  # tconst -> model uncertainty

    # Process all items with model predictions
    for tconst, (pred_mean, pred_std) in predictions.items():
        if tconst == target_movie.tconst:
            continue  # don't compare against itself

        movie = searcher.get_by_tconst(tconst)
        if movie:
            anchor_movies.append(movie)
            # Use comparison-calibrated rating if available, else model prediction
            if tconst in comparison_ratings:
                anchor_ratings[tconst] = comparison_ratings[tconst]
                # Lower uncertainty for explicitly calibrated items
                anchor_uncertainties[tconst] = pred_std * 0.5
            else:
                anchor_ratings[tconst] = pred_mean
                anchor_uncertainties[tconst] = pred_std

    # Add comparison-rated items that are in model but not in original ratings
    for tconst, rating in comparison_ratings.items():
        if tconst == target_movie.tconst:
            continue
        if tconst in anchor_ratings:
            continue  # already added
        # Check if in model
        pred = model.get_prediction(tconst)
        if pred is not None:
            movie = searcher.get_by_tconst(tconst)
            if movie:
                pred_mean, pred_std = pred
                anchor_movies.append(movie)
                anchor_ratings[tconst] = rating  # Use calibrated rating
                anchor_uncertainties[tconst] = pred_std * 0.5

    print(f"Using {len(anchor_movies)} anchors (model predictions + calibrated ratings)")

    # Initialize sampler
    sampler = AdaptiveBinarySearchSampler(
        target_movie=target_movie,
        anchor_movies=anchor_movies,
        anchor_ratings=anchor_ratings,
        model_prediction=model_pred,
        anchor_uncertainties=anchor_uncertainties,
    )

    # Create session
    session = Session(
        use_case="rate",
        started_at=datetime.now(),
        model_version=model_info.version,
        sampling_strategy="adaptive_binary_search",
        target_movie=target_movie,
    )

    print(f"\nSession: {session.id}")
    print()

    # Run comparison loop
    round_num = 0
    skipped = 0
    while round_num < args.max_rounds and not sampler.is_converged():
        pair = sampler.sample_pair()
        if pair is None:
            break

        round_num += 1
        print_comparison(
            round_num,
            f"~{args.max_rounds}",
            target_movie,
            pair.movie_b,
        )

        choice = get_choice()
        if choice is None:
            print("\nSession ended early.")
            break

        if choice == "skip":
            # Mark anchor as used but don't update bounds
            sampler._used_anchors.add(pair.movie_b.tconst)
            round_num -= 1
            skipped += 1
            print("Skipped.")
            print()
            continue

        # Update sampler
        target_preferred = (choice == ComparisonChoice.A)
        anchor_rating = anchor_ratings[pair.movie_b.tconst]
        sampler.update(target_preferred, pair.movie_b.tconst, anchor_rating)

        # Get current estimate
        estimate, low, high = sampler.get_estimate()

        # Create comparison record
        comparison = Comparison(
            movie_a=target_movie,
            movie_b=pair.movie_b,
            choice=choice,
            timestamp=datetime.now(),
            session_id=session.id,
            round_num=round_num,
            use_case="rate",
            model_prediction=ModelPrediction(
                prob_a_wins=pair.prediction.prob_a_wins,
                entropy=pair.prediction.entropy,
                rating_a=model_pred,
                rating_b=anchor_rating,
                model_version=model_info.version,
            ),
            user_ratings=UserRatings(rating_a=None, rating_b=anchor_rating),
            sampling=SamplingInfo(
                strategy="adaptive_binary_search",
                entropy=pair.prediction.entropy,
            ),
            target_movie=target_movie,
            rating_estimate=RatingEstimate(
                rating=estimate,
                confidence_low=low,
                confidence_high=high,
                n_comparisons=round_num,
            ),
        )

        logger.log_comparison(comparison)
        session.comparisons.append(comparison)
        session.n_comparisons += 1

        print(f"Current estimate: {estimate:.1f} [{low:.1f} - {high:.1f}]")
        print()

    # Final estimate
    estimate, low, high = sampler.get_estimate()
    session.rating_estimate = RatingEstimate(
        rating=estimate,
        confidence_low=low,
        confidence_high=high,
        n_comparisons=session.n_comparisons,
    )

    # Finalize session
    session.ended_at = datetime.now()
    logger.log_session(session)

    print("=" * 65)
    print(f"Rating estimate: {estimate:.1f} [{low:.1f} - {high:.1f}]")
    print()

    try:
        save = input("Save this rating? [y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        save = "n"

    if save == "y":
        logger.log_rating(
            tconst=target_movie.tconst,
            title=target_movie.title,
            year=target_movie.year,
            rating=estimate,
            confidence_low=low,
            confidence_high=high,
            n_comparisons=session.n_comparisons,
            session_id=session.id,
            model_uncertainty=model_uncertainty,  # None if not in model
        )
        print(f"Rating saved: {target_movie.title} = {estimate:.1f}")
    else:
        print("Rating not saved.")

    print(f"\nSession ID: {session.id}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Preference elicitation CLI")
    parser.add_argument(
        "--model",
        default="models/irt_v2_k50.pt",
        help="Path to trained model",
    )
    subparsers = parser.add_subparsers(dest="command", help="Mode")

    # Calibrate subcommand
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate model to your preferences")
    cal_parser.add_argument(
        "--n-rounds",
        type=int,
        default=20,
        help="Number of comparisons (default: 20)",
    )
    cal_parser.add_argument(
        "--strategy",
        choices=["entropy", "discrepancy", "uncertainty"],
        default="entropy",
        help="Sampling strategy: entropy (model uncertain), discrepancy (model wrong), or uncertainty (reduce θ variance)",
    )

    # Rate subcommand
    rate_parser = subparsers.add_parser("rate", help="Rate a movie via pairwise comparisons")
    rate_parser.add_argument(
        "movie",
        help="Movie title to search for",
    )
    rate_parser.add_argument(
        "--year",
        type=int,
        help="Filter by release year (e.g., --year 1987)",
    )
    rate_parser.add_argument(
        "--max-rounds",
        type=int,
        default=7,
        help="Maximum comparisons (default: 7)",
    )

    args = parser.parse_args()

    if args.command == "calibrate":
        run_calibrate(args)
    elif args.command == "rate":
        run_rate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
