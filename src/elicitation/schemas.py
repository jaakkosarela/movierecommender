"""Data schemas for preference elicitation."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class ComparisonChoice(str, Enum):
    """User's choice in a pairwise comparison."""
    A = "a"
    B = "b"


@dataclass
class Movie:
    """Movie metadata for display and logging."""

    tconst: str
    title: str
    year: Optional[int] = None
    genres: list[str] = field(default_factory=list)

    # Optional: set if movie is in MovieLens
    movielens_id: Optional[int] = None

    # Optional: for rating mode target movie
    model_prediction: Optional[float] = None

    # Type: "movie", "tvMovie", "tvSeries", "tvMiniSeries"
    title_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tconst": self.tconst,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "in_movielens": self.movielens_id is not None,
            "model_prediction": self.model_prediction,
            "title_type": self.title_type,
        }

    def type_label(self) -> str:
        """Short label for display (e.g., 'series', 'mini')."""
        if self.title_type == "tvSeries":
            return "series"
        elif self.title_type == "tvMiniSeries":
            return "mini"
        elif self.title_type == "tvMovie":
            return "tv movie"
        return ""

    def display_str(self) -> str:
        """Format for CLI display."""
        year_str = f" ({self.year})" if self.year else ""
        genres_str = ", ".join(self.genres) if self.genres else ""
        return f"{self.title}{year_str}\n      {genres_str}"


@dataclass
class ModelPrediction:
    """Model's prediction for a comparison."""

    prob_a_wins: float
    entropy: float
    rating_a: Optional[float] = None
    rating_b: Optional[float] = None
    model_version: Optional[str] = None


@dataclass
class UserRatings:
    """User's actual ratings for compared movies (when known)."""

    rating_a: Optional[float] = None
    rating_b: Optional[float] = None


@dataclass
class SamplingInfo:
    """Metadata about how this pair was selected."""

    strategy: str  # "max_entropy", "adaptive_binary_search", "random"
    entropy: Optional[float] = None  # Binary entropy H(p) at selection time


@dataclass
class Comparison:
    """A single pairwise comparison."""

    movie_a: Movie
    movie_b: Movie
    choice: Optional[ComparisonChoice] = None

    # Metadata (ID assigned by logger)
    id: Optional[str] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    round_num: Optional[int] = None
    use_case: str = "calibrate"  # "calibrate" or "rate"

    # Model info
    model_prediction: Optional[ModelPrediction] = None

    # User's actual ratings (when known)
    user_ratings: Optional[UserRatings] = None

    # Sampling metadata
    sampling: Optional[SamplingInfo] = None

    # For rate mode: the target movie being rated and current estimate
    target_movie: Optional[Movie] = None
    rating_estimate: Optional["RatingEstimate"] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "session_id": self.session_id,
            "round": self.round_num,
            "use_case": self.use_case,
            "movie_a": self.movie_a.to_dict(),
            "movie_b": self.movie_b.to_dict(),
            "choice": self.choice.value if self.choice else None,
        }

        if self.model_prediction:
            d["model_prediction"] = {
                "version": self.model_prediction.model_version,
                "prob_a_wins": self.model_prediction.prob_a_wins,
                "rating_a": self.model_prediction.rating_a,
                "rating_b": self.model_prediction.rating_b,
            }

        if self.user_ratings:
            d["user_ratings"] = {
                "rating_a": self.user_ratings.rating_a,
                "rating_b": self.user_ratings.rating_b,
            }

        if self.sampling:
            d["sampling"] = {
                "strategy": self.sampling.strategy,
                "entropy": self.sampling.entropy,
            }

        if self.target_movie:
            d["target_movie"] = self.target_movie.to_dict()

        if self.rating_estimate:
            d["rating_estimate"] = {
                "rating": self.rating_estimate.rating,
                "confidence_low": self.rating_estimate.confidence_low,
                "confidence_high": self.rating_estimate.confidence_high,
            }

        return d


@dataclass
class RatingEstimate:
    """Estimated rating from binary search."""

    rating: float
    confidence_low: float
    confidence_high: float
    n_comparisons: int


@dataclass
class Session:
    """A preference elicitation session."""

    id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:8]}")
    use_case: str = "calibrate"  # "calibrate" or "rate"

    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    model_version: Optional[str] = None
    sampling_strategy: Optional[str] = None

    n_comparisons: int = 0
    comparisons: list[Comparison] = field(default_factory=list)

    # For rate mode
    target_movie: Optional[Movie] = None
    rating_estimate: Optional[RatingEstimate] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "session_id": self.id,
            "use_case": self.use_case,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "model_version": self.model_version,
            "sampling_strategy": self.sampling_strategy,
            "n_comparisons": self.n_comparisons,
        }

        if self.target_movie:
            d["target_movie"] = self.target_movie.to_dict()

        if self.rating_estimate:
            d["rating_estimate"] = {
                "rating": self.rating_estimate.rating,
                "confidence_low": self.rating_estimate.confidence_low,
                "confidence_high": self.rating_estimate.confidence_high,
                "n_comparisons": self.rating_estimate.n_comparisons,
            }

        return d
