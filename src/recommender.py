"""Movie recommendation engine using collaborative filtering."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Recommendation:
    """A single movie recommendation."""

    movie_id: int
    predicted_rating: float
    neighbor_count: int
    confidence: float  # Based on neighbor count and similarity


def predict_ratings(
    similar_users: list[tuple[int, float, int]],
    matrix: sparse.csr_matrix,
    user_rated_movies: set[int],
    movie_ids: np.ndarray,
    min_neighbors: int = 3,
    scale_to_10: bool = True,
) -> dict[int, tuple[float, int, float]]:
    """Predict ratings for unrated movies based on similar users.

    Args:
        similar_users: List of (user_idx, similarity, overlap) from find_similar_users
        matrix: Sparse ratings matrix (users x movies)
        user_rated_movies: Set of movieIds already rated by user
        movie_ids: Array mapping column index to movieId
        min_neighbors: Minimum neighbors required to make prediction
        scale_to_10: Scale predictions from MovieLens (0.5-5) to IMDb (1-10) scale

    Returns:
        Dict of {movieId: (predicted_rating, neighbor_count, avg_similarity)}
    """
    if not similar_users:
        return {}

    # Extract user indices and similarities
    neighbor_indices = [u[0] for u in similar_users]
    similarities = np.array([u[1] for u in similar_users], dtype=np.float32)

    # Get ratings from all neighbors
    neighbor_matrix = matrix[neighbor_indices, :].toarray()  # (n_neighbors, n_movies)

    predictions = {}

    for col_idx in range(len(movie_ids)):
        movie_id = movie_ids[col_idx]

        # Skip if user already rated this movie
        if movie_id in user_rated_movies:
            continue

        # Get neighbor ratings for this movie
        neighbor_ratings = neighbor_matrix[:, col_idx]

        # Find neighbors who rated this movie
        rated_mask = neighbor_ratings != 0
        n_rated = rated_mask.sum()

        if n_rated < min_neighbors:
            continue

        # Weighted average by similarity
        relevant_ratings = neighbor_ratings[rated_mask]
        relevant_sims = similarities[rated_mask]

        # Handle negative similarities
        weights = np.maximum(relevant_sims, 0)  # Only use positive similarities
        if weights.sum() == 0:
            continue

        predicted = np.average(relevant_ratings, weights=weights)
        avg_sim = weights.mean()

        # Scale from MovieLens (0.5-5.0) to IMDb-like (1-10) scale
        if scale_to_10:
            predicted = (predicted - 0.5) * (9.0 / 4.5) + 1.0  # Maps 0.5->1, 5.0->10

        predictions[int(movie_id)] = (float(predicted), int(n_rated), float(avg_sim))

    return predictions


def rank_recommendations(
    predictions: dict[int, tuple[float, int, float]],
    min_rating: float = 7.0,
    top_n: int = 50,
) -> list[Recommendation]:
    """Rank and filter predictions into recommendations.

    Args:
        predictions: Dict from predict_ratings
        min_rating: Minimum predicted rating to include
        top_n: Number of recommendations to return

    Returns:
        List of Recommendation objects, sorted by predicted rating
    """
    recommendations = []

    for movie_id, (pred_rating, n_neighbors, avg_sim) in predictions.items():
        if pred_rating < min_rating:
            continue

        # Confidence based on neighbor count (more = better)
        confidence = min(1.0, n_neighbors / 10) * avg_sim

        recommendations.append(Recommendation(
            movie_id=movie_id,
            predicted_rating=pred_rating,
            neighbor_count=n_neighbors,
            confidence=confidence,
        ))

    # Sort by predicted rating descending, then by confidence
    recommendations.sort(key=lambda r: (r.predicted_rating, r.confidence), reverse=True)

    return recommendations[:top_n]


def enrich_with_metadata(
    recommendations: list[Recommendation],
    movies_df: pd.DataFrame,
    movieid_to_tconst: dict,
    imdb_basics: pd.DataFrame = None,
    imdb_ratings: pd.DataFrame = None,
) -> pd.DataFrame:
    """Enrich recommendations with movie metadata.

    Args:
        recommendations: List of Recommendation objects
        movies_df: MovieLens movies DataFrame
        movieid_to_tconst: Mapping from movieId to IMDb tconst
        imdb_basics: Optional IMDb basics DataFrame
        imdb_ratings: Optional IMDb ratings DataFrame

    Returns:
        DataFrame with full recommendation details
    """
    if not recommendations:
        return pd.DataFrame()

    # Build base DataFrame
    data = []
    for rec in recommendations:
        data.append({
            "movieId": rec.movie_id,
            "predicted_rating": rec.predicted_rating,
            "neighbor_count": rec.neighbor_count,
            "confidence": rec.confidence,
        })

    df = pd.DataFrame(data)

    # Add MovieLens metadata
    df = df.merge(movies_df, on="movieId", how="left")

    # Add IMDb ID
    df["tconst"] = df["movieId"].map(movieid_to_tconst)

    # Add IMDb metadata if available
    if imdb_basics is not None and imdb_ratings is not None:
        imdb = imdb_basics.merge(imdb_ratings, on="tconst", how="inner")
        df = df.merge(
            imdb[["tconst", "startYear", "runtimeMinutes", "averageRating", "numVotes"]],
            on="tconst",
            how="left",
        )
        df = df.rename(columns={
            "averageRating": "imdb_rating",
            "numVotes": "imdb_votes",
        })

    return df


class Recommender:
    """High-level recommendation interface."""

    def __init__(self, data):
        """Initialize with RecommenderData object."""
        self.data = data

    def get_recommendations(
        self,
        user_ratings: dict[int, float],
        similar_users: list[tuple[int, float, int]],
        min_neighbors: int = 3,
        min_predicted_rating: float = 7.0,
        min_imdb_rating: float = None,
        min_imdb_votes: int = None,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """Generate recommendations for a user.

        Args:
            user_ratings: Dict of {movieId: rating}
            similar_users: Output from find_similar_users
            min_neighbors: Minimum neighbors to make prediction
            min_predicted_rating: Minimum predicted rating
            min_imdb_rating: Optional IMDb rating filter
            min_imdb_votes: Optional IMDb vote count filter
            top_n: Number of recommendations

        Returns:
            DataFrame of recommendations with metadata
        """
        # Predict ratings
        user_movie_ids = set(user_ratings.keys())
        predictions = predict_ratings(
            similar_users,
            self.data.matrix,
            user_movie_ids,
            self.data.movie_ids,
            min_neighbors=min_neighbors,
        )

        # Rank and filter
        recommendations = rank_recommendations(
            predictions,
            min_rating=min_predicted_rating,
            top_n=top_n * 2,  # Get extra to allow for filtering
        )

        # Enrich with metadata
        df = enrich_with_metadata(
            recommendations,
            self.data.movies,
            self.data.movieid_to_tconst,
            self.data.imdb_basics,
            self.data.imdb_ratings,
        )

        if df.empty:
            return df

        # Apply IMDb filters if available
        if min_imdb_rating is not None and "imdb_rating" in df.columns:
            df = df[df["imdb_rating"] >= min_imdb_rating]

        if min_imdb_votes is not None and "imdb_votes" in df.columns:
            df = df[df["imdb_votes"] >= min_imdb_votes]

        return df.head(top_n)

    def sample_recommendations(
        self,
        user_ratings: dict[int, float],
        similar_users: list[tuple[int, float, int]],
        n: int = 5,
        pool_size: int = 100,
        min_neighbors: int = 3,
        weighting: Literal["linear", "uniform", "squared", "softmax"] = "linear",
    ) -> list[dict]:
        """Sample n recommendations weighted by predicted rating.

        Args:
            user_ratings: Dict of {movieId: rating}
            similar_users: Output from find_similar_users
            n: Number of recommendations to sample
            pool_size: Size of candidate pool (top N predictions)
            min_neighbors: Minimum neighbors to make prediction
            weighting: Weighting scheme for sampling

        Returns:
            List of dicts with title, year, genres (no scores)
        """
        # Get predictions
        user_movie_ids = set(user_ratings.keys())
        predictions = predict_ratings(
            similar_users,
            self.data.matrix,
            user_movie_ids,
            self.data.movie_ids,
            min_neighbors=min_neighbors,
        )

        if not predictions:
            return []

        # Sort and take top pool_size
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1][0], reverse=True)
        sorted_preds = sorted_preds[:pool_size]

        movie_ids = np.array([p[0] for p in sorted_preds])
        scores = np.array([p[1][0] for p in sorted_preds])

        # Compute weights
        if weighting == "uniform":
            weights = np.ones_like(scores)
        elif weighting == "linear":
            weights = scores - scores.min() + 0.1
        elif weighting == "squared":
            weights = (scores - scores.min() + 0.1) ** 2
        elif weighting == "softmax":
            weights = np.exp((scores - scores.max()) / 1.0)
        else:
            weights = np.ones_like(scores)

        # Sample without replacement
        probs = weights / weights.sum()
        chosen_idx = np.random.choice(len(movie_ids), size=min(n, len(movie_ids)), replace=False, p=probs)
        chosen_movie_ids = movie_ids[chosen_idx]

        # Build results with metadata
        movies_df = self.data.movies.set_index("movieId")
        results = []

        for mid in chosen_movie_ids:
            mid = int(mid)
            info = {"movieId": mid}

            if mid in movies_df.index:
                row = movies_df.loc[mid]
                info["title"] = row["title"]
                info["genres"] = row["genres"]
            else:
                info["title"] = f"Movie {mid}"
                info["genres"] = ""

            results.append(info)

        return results
