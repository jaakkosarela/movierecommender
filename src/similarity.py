"""User similarity computation using Pearson correlation."""

import numpy as np
from scipy import sparse


def find_similar_users(
    user_ratings: dict[int, float],
    matrix: sparse.csr_matrix,
    movie_to_idx: dict[int, int],
    user_means: np.ndarray,
    user_stds: np.ndarray,
    min_overlap: int = 5,
    top_k: int = 50,
) -> list[tuple[int, float, int]]:
    """Find MovieLens users most similar to the target user.

    Args:
        user_ratings: Dict of {movieId: rating} for target user
        matrix: Sparse ratings matrix (users x movies)
        movie_to_idx: Mapping from movieId to matrix column index
        user_means: Pre-computed mean rating per user
        user_stds: Pre-computed std deviation per user
        min_overlap: Minimum number of common movies required
        top_k: Number of similar users to return

    Returns:
        List of (user_idx, similarity, overlap_count) tuples, sorted by similarity desc
    """
    # Convert user ratings to column indices
    user_movie_indices = []
    user_rating_values = []

    for movie_id, rating in user_ratings.items():
        if movie_id in movie_to_idx:
            user_movie_indices.append(movie_to_idx[movie_id])
            user_rating_values.append(rating)

    if len(user_movie_indices) < min_overlap:
        print(f"Warning: User has only {len(user_movie_indices)} mapped movies, need {min_overlap}")
        return []

    user_movie_indices = np.array(user_movie_indices)
    user_rating_values = np.array(user_rating_values, dtype=np.float32)

    # Compute target user's mean and std
    user_mean = user_rating_values.mean()
    user_std = user_rating_values.std()
    if user_std == 0:
        user_std = 1.0

    # Centered user ratings
    user_centered = user_rating_values - user_mean

    # Find similar users
    n_users = matrix.shape[0]
    similarities = []

    # Extract only the columns the user has rated (for efficiency)
    user_cols = matrix[:, user_movie_indices].toarray()  # (n_users, n_user_movies)

    for i in range(n_users):
        row = user_cols[i]
        # Find non-zero (rated) movies for this MovieLens user
        rated_mask = row != 0
        overlap = rated_mask.sum()

        if overlap < min_overlap:
            continue

        # Get overlapping ratings
        ml_ratings = row[rated_mask]
        target_ratings = user_centered[rated_mask]

        # Center MovieLens user ratings
        ml_mean = user_means[i]
        ml_std = user_stds[i]
        ml_centered = ml_ratings - ml_mean

        # Pearson correlation
        numerator = np.dot(target_ratings, ml_centered)
        denominator = user_std * ml_std * overlap

        if denominator > 0:
            pearson = numerator / denominator
            similarities.append((i, pearson, overlap))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def find_similar_users_vectorized(
    user_ratings: dict[int, float],
    matrix: sparse.csr_matrix,
    movie_to_idx: dict[int, int],
    user_means: np.ndarray = None,  # Not used, kept for API compatibility
    user_stds: np.ndarray = None,   # Not used, kept for API compatibility
    min_overlap: int = 5,
    top_k: int = 50,
) -> list[tuple[int, float, int]]:
    """Vectorized version - faster for large matrices.

    Computes Pearson correlation over overlapping movies only.
    """
    # Convert user ratings to column indices
    user_movie_indices = []
    user_rating_values = []

    for movie_id, rating in user_ratings.items():
        if movie_id in movie_to_idx:
            user_movie_indices.append(movie_to_idx[movie_id])
            user_rating_values.append(rating)

    n_user_movies = len(user_movie_indices)
    if n_user_movies < min_overlap:
        print(f"Warning: User has only {n_user_movies} mapped movies, need {min_overlap}")
        return []

    user_movie_indices = np.array(user_movie_indices)
    user_rating_values = np.array(user_rating_values, dtype=np.float32)

    # Extract submatrix for user's movies only
    submatrix = matrix[:, user_movie_indices].toarray()  # (n_users, n_user_movies)

    # Mask for non-zero ratings
    rated_mask = submatrix != 0

    # Count overlaps per user
    overlaps = rated_mask.sum(axis=1)

    # Filter users with sufficient overlap
    valid_users = overlaps >= min_overlap
    valid_indices = np.where(valid_users)[0]

    if len(valid_indices) == 0:
        print(f"Warning: No users found with {min_overlap}+ overlapping movies")
        return []

    # Compute similarities for valid users
    similarities = []

    for i in valid_indices:
        mask = rated_mask[i]

        # Get only overlapping ratings
        ml_ratings = submatrix[i, mask]
        target_ratings = user_rating_values[mask]

        # Compute means over overlapping items only
        ml_mean = ml_ratings.mean()
        target_mean = target_ratings.mean()

        # Center the ratings
        ml_centered = ml_ratings - ml_mean
        target_centered = target_ratings - target_mean

        # Pearson correlation: cov(x,y) / (std(x) * std(y))
        numerator = np.dot(target_centered, ml_centered)
        denom_target = np.sqrt(np.dot(target_centered, target_centered))
        denom_ml = np.sqrt(np.dot(ml_centered, ml_centered))
        denominator = denom_target * denom_ml

        if denominator > 0:
            pearson = numerator / denominator
            # Clip to [-1, 1] to handle floating point errors
            pearson = np.clip(pearson, -1.0, 1.0)
            similarities.append((i, float(pearson), int(overlaps[i])))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def get_user_ratings_dict(user_ratings_df, movie_id_col: str = "movieId") -> dict[int, float]:
    """Convert user ratings DataFrame to dict format.

    Args:
        user_ratings_df: DataFrame with movieId and rating columns
        movie_id_col: Name of movie ID column

    Returns:
        Dict of {movieId: rating}
    """
    return dict(zip(
        user_ratings_df[movie_id_col].astype(int),
        user_ratings_df["rating"].astype(float)
    ))
