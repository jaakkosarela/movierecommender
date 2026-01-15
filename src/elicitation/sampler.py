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


class DiscrepancySampler(BaseSampler):
    """Sample pairs to correct model's confident mistakes.

    Strategy:
    1. Find items with high |pred - actual| (model is wrong)
    2. Compare against anchors where anchor_pred ≈ actual_discrepancy_item
    3. Model predicts discrepancy item wins, but user likely picks anchor
    4. Maximum information gain through surprise

    This targets "confidently wrong" predictions rather than uncertain ones.
    """

    def __init__(
        self,
        rated_movies: list[Movie],
        predicted_ratings: dict[str, float],  # tconst -> model prediction
        actual_ratings: dict[str, float],  # tconst -> user's actual rating
        exclude_pairs: Optional[set[tuple[str, str]]] = None,
        min_discrepancy: float = 1.0,  # minimum |pred - actual| to be a target
    ):
        """
        Args:
            rated_movies: Movies the user has rated that are in model
            predicted_ratings: Model's predicted rating for each movie
            actual_ratings: User's actual rating for each movie
            exclude_pairs: Set of (tconst_a, tconst_b) pairs to exclude
            min_discrepancy: Minimum discrepancy to consider an item as target
        """
        self.rated_movies = rated_movies
        self.predicted_ratings = predicted_ratings
        self.actual_ratings = actual_ratings
        self.exclude_pairs = exclude_pairs or set()
        self.min_discrepancy = min_discrepancy

        self.movie_by_tconst = {m.tconst: m for m in rated_movies}

        # Compute discrepancies and build pairs
        self._compute_discrepancy_pairs()

    def _compute_discrepancy_pairs(self) -> None:
        """Build pairs: high-discrepancy item vs anchor near actual rating."""
        self.pairs = []
        seen_pairs: set[tuple[str, str]] = set()  # Track pairs in either direction

        # Find items with high discrepancy
        discrepancies = []
        for tconst in self.predicted_ratings:
            if tconst not in self.actual_ratings:
                continue
            pred = self.predicted_ratings[tconst]
            actual = self.actual_ratings[tconst]
            disc = pred - actual  # positive = model overestimates
            abs_disc = abs(disc)

            if abs_disc >= self.min_discrepancy:
                discrepancies.append((tconst, pred, actual, disc, abs_disc))

        # Sort by absolute discrepancy descending
        discrepancies.sort(key=lambda x: -x[4])

        # For each high-discrepancy item, find good anchors
        for tc_target, pred_target, actual_target, disc, abs_disc in discrepancies:
            # Find anchors where anchor_pred is close to actual_target
            # This maximizes surprise when user picks based on actual ratings
            anchor_candidates = []

            for tc_anchor in self.predicted_ratings:
                if tc_anchor == tc_target:
                    continue
                if tc_anchor not in self.actual_ratings:
                    continue

                # Skip excluded pairs
                if (tc_target, tc_anchor) in self.exclude_pairs or (tc_anchor, tc_target) in self.exclude_pairs:
                    continue

                pred_anchor = self.predicted_ratings[tc_anchor]
                actual_anchor = self.actual_ratings[tc_anchor]

                # How close is anchor's prediction to target's actual?
                # We want anchor_pred ≈ actual_target
                anchor_distance = abs(pred_anchor - actual_target)

                # Also consider anchor's own calibration (prefer well-calibrated anchors)
                anchor_disc = abs(pred_anchor - actual_anchor)

                # Score: prefer anchors close to target's actual rating AND well-calibrated
                score = anchor_distance + 0.5 * anchor_disc

                # Model's belief about this comparison
                prob_target_wins = sigmoid(pred_target - pred_anchor)

                # Expected outcome based on actual ratings
                prob_target_wins_actual = sigmoid(actual_target - actual_anchor)

                # Information gain: how surprised will model be?
                # If model thinks target wins (p=0.8) but actual suggests anchor wins (p=0.3)
                # then we expect high information gain
                belief_divergence = abs(prob_target_wins - prob_target_wins_actual)

                anchor_candidates.append({
                    "tc_anchor": tc_anchor,
                    "pred_anchor": pred_anchor,
                    "actual_anchor": actual_anchor,
                    "score": score,
                    "prob_target_wins": prob_target_wins,
                    "prob_target_wins_actual": prob_target_wins_actual,
                    "belief_divergence": belief_divergence,
                })

            # Sort by belief divergence (maximize expected surprise)
            anchor_candidates.sort(key=lambda x: -x["belief_divergence"])

            # Take top anchors for this target
            for anchor in anchor_candidates[:5]:  # Up to 5 anchors per target
                tc_anchor = anchor["tc_anchor"]
                # Skip if we've already added this pair (in either direction)
                if (tc_target, tc_anchor) in seen_pairs or (tc_anchor, tc_target) in seen_pairs:
                    continue
                seen_pairs.add((tc_target, tc_anchor))
                self.pairs.append({
                    "tc_target": tc_target,
                    "tc_anchor": tc_anchor,
                    "pred_target": pred_target,
                    "pred_anchor": anchor["pred_anchor"],
                    "actual_target": actual_target,
                    "actual_anchor": anchor["actual_anchor"],
                    "discrepancy": disc,
                    "prob_target_wins": anchor["prob_target_wins"],
                    "belief_divergence": anchor["belief_divergence"],
                })

        # Sort all pairs by belief divergence (most informative first)
        self.pairs.sort(key=lambda x: -x["belief_divergence"])
        self._pair_index = 0

    def sample_pair(self) -> Optional[CandidatePair]:
        """Return the next highest-information pair."""
        if self._pair_index >= len(self.pairs):
            return None

        pair = self.pairs[self._pair_index]
        self._pair_index += 1

        return CandidatePair(
            movie_a=self.movie_by_tconst[pair["tc_target"]],
            movie_b=self.movie_by_tconst[pair["tc_anchor"]],
            prediction=ModelPrediction(
                prob_a_wins=pair["prob_target_wins"],
                entropy=binary_entropy(pair["prob_target_wins"]),
                rating_a=pair["pred_target"],
                rating_b=pair["pred_anchor"],
            ),
        )

    def mark_used(self, tconst_a: str, tconst_b: str) -> None:
        """Mark a pair as used (add to exclude set)."""
        self.exclude_pairs.add((tconst_a, tconst_b))

    def get_current_target_info(self) -> Optional[dict]:
        """Get info about the current target being calibrated."""
        if self._pair_index == 0 or self._pair_index > len(self.pairs):
            return None
        pair = self.pairs[self._pair_index - 1]
        return {
            "tconst": pair["tc_target"],
            "pred": pair["pred_target"],
            "actual": pair["actual_target"],
            "discrepancy": pair["discrepancy"],
        }


class FactorUncertaintySampler(BaseSampler):
    """Sample pairs that maximally reduce uncertainty in user factors θ.

    Strategy:
    1. For each pair (A, B), the comparison constrains θ along direction (β_A - β_B)
    2. Score = weighted variance of θ along that direction
    3. Weight by prior_scale² to downweight noisy later factors
    4. Optionally weight by entropy to prefer uncertain outcomes

    This targets pairs that probe dimensions where θ is uncertain.
    """

    def __init__(
        self,
        rated_movies: list[Movie],
        predicted_ratings: dict[str, float],  # tconst -> model prediction
        actual_ratings: dict[str, float],  # tconst -> user's actual rating
        item_mu: torch.Tensor,  # [n_items, K] - item factors β
        tconst_to_idx: dict[str, int],  # tconst -> item index
        user_log_std: torch.Tensor,  # [K] - user factor uncertainty
        prior_scales: torch.Tensor,  # [K] - prior scales for weighting
        exclude_pairs: Optional[set[tuple[str, str]]] = None,
        min_prob: float = 0.1,  # minimum P(A>B) to consider (avoid very lopsided pairs)
        entropy_weight: float = 1.0,  # weight for entropy in scoring (0=ignore, 1=multiply)
    ):
        """
        Args:
            rated_movies: Movies the user has rated that are in model
            predicted_ratings: Model's predicted rating for each movie
            actual_ratings: User's actual rating for each movie
            item_mu: Item factor means from trained model
            tconst_to_idx: Mapping from tconst to item index
            user_log_std: Log std of user factors (uncertainty)
            prior_scales: Prior scales per factor dimension
            exclude_pairs: Set of (tconst_a, tconst_b) pairs to exclude
            min_prob: Minimum P(A>B) - avoid pairs where outcome is obvious
        """
        self.rated_movies = rated_movies
        self.predicted_ratings = predicted_ratings
        self.actual_ratings = actual_ratings
        self.item_mu = item_mu
        self.tconst_to_idx = tconst_to_idx
        self.user_log_std = user_log_std
        self.prior_scales = prior_scales
        self.exclude_pairs = exclude_pairs or set()
        self.min_prob = min_prob
        self.entropy_weight = entropy_weight

        self.movie_by_tconst = {m.tconst: m for m in rated_movies}

        # Precompute pairs scored by factor uncertainty reduction
        self._compute_uncertainty_pairs()

    def _compute_uncertainty_pairs(self) -> None:
        """Compute information score for all pairs."""
        self.pairs = []
        seen_pairs: set[tuple[str, str]] = set()

        # User factor variance [K]
        theta_var = torch.exp(2 * self.user_log_std)

        # Prior weights - downweight later factors [K]
        prior_weights = self.prior_scales ** 2

        # Get list of tconsts that are in both rated set and model
        valid_tconsts = [
            tc for tc in self.predicted_ratings
            if tc in self.tconst_to_idx and tc in self.actual_ratings
        ]

        # Compute scores for all pairs
        for i, tc_a in enumerate(valid_tconsts):
            idx_a = self.tconst_to_idx[tc_a]
            pred_a = self.predicted_ratings[tc_a]
            actual_a = self.actual_ratings[tc_a]
            beta_a = self.item_mu[idx_a]  # [K]

            for tc_b in valid_tconsts[i + 1:]:
                # Skip excluded pairs
                if (tc_a, tc_b) in self.exclude_pairs or (tc_b, tc_a) in self.exclude_pairs:
                    continue
                if (tc_a, tc_b) in seen_pairs or (tc_b, tc_a) in seen_pairs:
                    continue

                idx_b = self.tconst_to_idx[tc_b]
                pred_b = self.predicted_ratings[tc_b]
                actual_b = self.actual_ratings[tc_b]
                beta_b = self.item_mu[idx_b]  # [K]

                # Direction of comparison
                direction = beta_a - beta_b  # [K]

                # Weighted variance along this direction
                # Higher = more uncertainty reduced by this comparison
                info_score = (theta_var * direction**2 * prior_weights).sum().item()

                # Model's belief about comparison
                prob_a_wins = sigmoid(pred_a - pred_b)

                # Skip very lopsided pairs (outcome too predictable)
                if prob_a_wins < self.min_prob or prob_a_wins > (1 - self.min_prob):
                    continue

                # Expected outcome based on actual ratings
                prob_a_wins_actual = sigmoid(actual_a - actual_b)

                # Entropy of the predicted outcome
                entropy = binary_entropy(prob_a_wins)

                # Combined score: info_score weighted by entropy
                # entropy_weight=0: pure info_score
                # entropy_weight=1: info_score * entropy
                if self.entropy_weight > 0:
                    combined_score = info_score * (entropy ** self.entropy_weight)
                else:
                    combined_score = info_score

                seen_pairs.add((tc_a, tc_b))
                self.pairs.append({
                    "tc_a": tc_a,
                    "tc_b": tc_b,
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "actual_a": actual_a,
                    "actual_b": actual_b,
                    "prob_a_wins": prob_a_wins,
                    "prob_a_wins_actual": prob_a_wins_actual,
                    "info_score": info_score,
                    "entropy": entropy,
                    "combined_score": combined_score,
                    "direction_norm": direction.norm().item(),
                })

        # Sort by combined_score descending (most informative first)
        self.pairs.sort(key=lambda x: -x["combined_score"])
        self._pair_index = 0

    def sample_pair(self) -> Optional[CandidatePair]:
        """Return the next highest-information pair."""
        if self._pair_index >= len(self.pairs):
            return None

        pair = self.pairs[self._pair_index]
        self._pair_index += 1

        return CandidatePair(
            movie_a=self.movie_by_tconst[pair["tc_a"]],
            movie_b=self.movie_by_tconst[pair["tc_b"]],
            prediction=ModelPrediction(
                prob_a_wins=pair["prob_a_wins"],
                entropy=binary_entropy(pair["prob_a_wins"]),
                rating_a=pair["pred_a"],
                rating_b=pair["pred_b"],
            ),
        )

    def mark_used(self, tconst_a: str, tconst_b: str) -> None:
        """Mark a pair as used (add to exclude set)."""
        self.exclude_pairs.add((tconst_a, tconst_b))

    def get_top_pairs_summary(self, n: int = 10) -> list[dict]:
        """Get summary of top N pairs for debugging."""
        return [
            {
                "tc_a": p["tc_a"],
                "tc_b": p["tc_b"],
                "info_score": p["info_score"],
                "prob_a_wins": p["prob_a_wins"],
                "direction_norm": p["direction_norm"],
            }
            for p in self.pairs[:n]
        ]
