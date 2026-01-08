"""Sampling strategies for preference elicitation."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .schemas import Movie, ModelPrediction


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def binary_entropy(p: float) -> float:
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


@dataclass
class CandidatePair:
    """A candidate pair for comparison with model prediction."""

    movie_a: Movie
    movie_b: Movie
    prediction: ModelPrediction


class BaseSampler(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def sample_pair(self) -> Optional[CandidatePair]:
        """Sample a pair of movies for comparison."""
        pass


class MaxEntropySampler(BaseSampler):
    """Sample pairs where model is most uncertain (max entropy).

    For calibration: compare movies from user's rated set where
    P(A > B) is closest to 0.5.

    Supports two modes:
    1. Symmetric: both A and B from same pool (movies in model)
    2. Asymmetric: A from targets (movies + series), B from anchors (movies in model)
    """

    def __init__(
        self,
        rated_movies: list[Movie],
        predicted_ratings: dict[str, float],  # tconst -> predicted rating
        exclude_pairs: Optional[set[tuple[str, str]]] = None,
        # Optional: for including series as targets
        target_items: Optional[list[Movie]] = None,  # Series + movies to calibrate
        target_ratings: Optional[dict[str, float]] = None,  # tconst -> user rating
    ):
        """
        Args:
            rated_movies: Movies the user has rated that are in model (anchors)
            predicted_ratings: Model's predicted rating for each anchor movie
            exclude_pairs: Set of (tconst_a, tconst_b) pairs to exclude
            target_items: Additional items (e.g., series) that can be targets (A side)
            target_ratings: User ratings for target_items (used as proxy for prediction)
        """
        self.rated_movies = rated_movies
        self.predicted_ratings = predicted_ratings
        self.exclude_pairs = exclude_pairs or set()
        self.target_items = target_items or []
        self.target_ratings = target_ratings or {}

        # Build lookup for anchors (movies in model)
        self.movie_by_tconst = {m.tconst: m for m in rated_movies}

        # Add target items to lookup
        for item in self.target_items:
            if item.tconst not in self.movie_by_tconst:
                self.movie_by_tconst[item.tconst] = item

        # Precompute all pairs with entropy
        self._compute_pair_entropies()

    def _compute_pair_entropies(self) -> None:
        """Compute entropy for all possible pairs."""
        self.pairs = []

        # Anchor tconsts (movies in model)
        anchor_tconsts = list(self.predicted_ratings.keys())

        # Target tconsts (series + movies being calibrated)
        target_tconsts = set(self.target_ratings.keys())

        # 1. Symmetric pairs: anchor vs anchor (both movies in model)
        for i, tc_a in enumerate(anchor_tconsts):
            for tc_b in anchor_tconsts[i + 1:]:
                # Skip excluded pairs
                if (tc_a, tc_b) in self.exclude_pairs or (tc_b, tc_a) in self.exclude_pairs:
                    continue

                # Skip if either movie not in rated set
                if tc_a not in self.movie_by_tconst or tc_b not in self.movie_by_tconst:
                    continue

                rating_a = self.predicted_ratings[tc_a]
                rating_b = self.predicted_ratings[tc_b]

                # Bradley-Terry probability
                prob_a_wins = sigmoid(rating_a - rating_b)
                entropy = binary_entropy(prob_a_wins)

                self.pairs.append((tc_a, tc_b, prob_a_wins, entropy, rating_a, rating_b))

        # 2. Asymmetric pairs: target (series) vs anchor (movie in model)
        for tc_target in target_tconsts:
            # Skip if target is also an anchor (already covered above)
            if tc_target in self.predicted_ratings:
                continue

            target_rating = self.target_ratings[tc_target]

            for tc_anchor in anchor_tconsts:
                # Skip excluded pairs
                if (tc_target, tc_anchor) in self.exclude_pairs or (tc_anchor, tc_target) in self.exclude_pairs:
                    continue

                if tc_anchor not in self.movie_by_tconst:
                    continue

                anchor_pred = self.predicted_ratings[tc_anchor]

                # Use user rating as proxy for target's "prediction"
                prob_target_wins = sigmoid(target_rating - anchor_pred)
                entropy = binary_entropy(prob_target_wins)

                self.pairs.append((tc_target, tc_anchor, prob_target_wins, entropy, target_rating, anchor_pred))

        # Sort by entropy descending (most uncertain first)
        self.pairs.sort(key=lambda x: -x[3])
        self._pair_index = 0

    def sample_pair(self) -> Optional[CandidatePair]:
        """Return the next highest-entropy pair."""
        if self._pair_index >= len(self.pairs):
            return None

        tc_a, tc_b, prob_a, entropy, rating_a, rating_b = self.pairs[self._pair_index]
        self._pair_index += 1

        return CandidatePair(
            movie_a=self.movie_by_tconst[tc_a],
            movie_b=self.movie_by_tconst[tc_b],
            prediction=ModelPrediction(
                prob_a_wins=prob_a,
                entropy=entropy,
                rating_a=rating_a,
                rating_b=rating_b,
            ),
        )

    def mark_used(self, tconst_a: str, tconst_b: str) -> None:
        """Mark a pair as used (add to exclude set)."""
        self.exclude_pairs.add((tconst_a, tconst_b))


class AdaptiveBinarySearchSampler(BaseSampler):
    """Adaptive binary search for rating a new movie.

    Selects anchors to narrow down the rating range efficiently.
    Prefers low-uncertainty anchors when selecting comparisons.
    """

    def __init__(
        self,
        target_movie: Movie,
        anchor_movies: list[Movie],
        anchor_ratings: dict[str, float],  # tconst -> rating (model pred or calibrated)
        model_prediction: Optional[float] = None,
        anchor_uncertainties: Optional[dict[str, float]] = None,  # tconst -> std
        uncertainty_weight: float = 0.5,  # how much to penalize uncertainty
    ):
        """
        Args:
            target_movie: The movie being rated
            anchor_movies: Rated movies to compare against
            anchor_ratings: Ratings for anchors (model predictions or calibrated)
            model_prediction: Model's prediction for target (if in MovieLens)
            anchor_uncertainties: Model uncertainty (std) for each anchor
            uncertainty_weight: Weight for preferring low-uncertainty anchors
        """
        self.target_movie = target_movie
        self.anchor_movies = anchor_movies
        self.anchor_ratings = anchor_ratings
        self.model_prediction = model_prediction
        self.anchor_uncertainties = anchor_uncertainties or {}
        self.uncertainty_weight = uncertainty_weight

        # Sort anchors by rating
        self.anchors_sorted = sorted(
            [(m, anchor_ratings[m.tconst]) for m in anchor_movies if m.tconst in anchor_ratings],
            key=lambda x: x[1],
        )

        # Initialize rating bounds
        if self.anchors_sorted:
            self.rating_low = self.anchors_sorted[0][1] - 0.5
            self.rating_high = self.anchors_sorted[-1][1] + 0.5
        else:
            self.rating_low = 1.0
            self.rating_high = 10.0

        # If we have model prediction, start search there
        if model_prediction is not None:
            self._current_estimate = model_prediction
        else:
            self._current_estimate = (self.rating_low + self.rating_high) / 2

        self._comparisons_done = 0
        self._used_anchors: set[str] = set()

    def sample_pair(self) -> Optional[CandidatePair]:
        """Select next anchor for binary search.

        Prefers anchors that are:
        1. Close to current estimate (informative for binary search)
        2. Within the valid rating range
        3. Low uncertainty (more reliable as anchors)

        Score = distance + uncertainty_weight * uncertainty
        Lower score is better.
        """
        best_anchor = None
        best_score = float("inf")

        for movie, rating in self.anchors_sorted:
            if movie.tconst in self._used_anchors:
                continue

            # Prefer anchors in the valid range
            if not (self.rating_low <= rating <= self.rating_high):
                continue

            distance = abs(rating - self._current_estimate)
            uncertainty = self.anchor_uncertainties.get(movie.tconst, 0.0)

            # Lower score = better (close + certain)
            score = distance + self.uncertainty_weight * uncertainty

            if score < best_score:
                best_score = score
                best_anchor = (movie, rating)

        if best_anchor is None:
            # No more valid anchors, try any unused anchor (prefer low uncertainty)
            best_score = float("inf")
            for movie, rating in self.anchors_sorted:
                if movie.tconst not in self._used_anchors:
                    uncertainty = self.anchor_uncertainties.get(movie.tconst, 0.0)
                    if uncertainty < best_score:
                        best_score = uncertainty
                        best_anchor = (movie, rating)

        if best_anchor is None:
            return None

        anchor_movie, anchor_rating = best_anchor

        # Predict probability (if we have model prediction)
        if self.model_prediction is not None:
            prob_target_wins = sigmoid(self.model_prediction - anchor_rating)
        else:
            prob_target_wins = 0.5  # No prediction available

        return CandidatePair(
            movie_a=self.target_movie,
            movie_b=anchor_movie,
            prediction=ModelPrediction(
                prob_a_wins=prob_target_wins,
                entropy=binary_entropy(prob_target_wins),
                rating_a=self.model_prediction,
                rating_b=anchor_rating,
            ),
        )

    def update(self, target_preferred: bool, anchor_tconst: str, anchor_rating: float) -> None:
        """Update bounds based on comparison result.

        Args:
            target_preferred: True if user preferred target movie
            anchor_tconst: tconst of the anchor movie
            anchor_rating: User's rating of the anchor
        """
        self._used_anchors.add(anchor_tconst)
        self._comparisons_done += 1

        if target_preferred:
            # Target > anchor, so target rating >= anchor rating
            self.rating_low = max(self.rating_low, anchor_rating)
        else:
            # Target < anchor, so target rating <= anchor rating
            self.rating_high = min(self.rating_high, anchor_rating)

        # Update estimate to midpoint
        self._current_estimate = (self.rating_low + self.rating_high) / 2

    def get_estimate(self) -> tuple[float, float, float]:
        """Get current rating estimate and confidence interval.

        Returns:
            (estimate, low, high)
        """
        return self._current_estimate, self.rating_low, self.rating_high

    def is_converged(self, threshold: float = 0.5) -> bool:
        """Check if search has converged (interval < threshold)."""
        return (self.rating_high - self.rating_low) < threshold

    @property
    def n_comparisons(self) -> int:
        return self._comparisons_done
