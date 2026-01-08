"""Recommendation generation from IRT model."""

from .thompson import (
    ThompsonConfig,
    UserPredictions,
    Recommendation,
    compute_user_predictions,
    add_imdb_metadata,
    thompson_sample,
    generate_recommendations,
    print_recommendations,
)

__all__ = [
    "ThompsonConfig",
    "UserPredictions",
    "Recommendation",
    "compute_user_predictions",
    "add_imdb_metadata",
    "thompson_sample",
    "generate_recommendations",
    "print_recommendations",
]
