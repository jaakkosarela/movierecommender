"""Data loading utilities for MovieLens and IMDb datasets."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse


# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data"


def load_movielens_ratings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load MovieLens 25M ratings.

    Returns DataFrame with columns: userId, movieId, rating, timestamp
    """
    data_dir = data_dir or DATA_DIR
    ratings_path = data_dir / "ml-25m" / "ratings.csv"

    return pd.read_csv(
        ratings_path,
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        usecols=["userId", "movieId", "rating"],  # Skip timestamp for memory
    )


def load_movielens_links(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load MovieLens to IMDb/TMDB mappings.

    Returns DataFrame with columns: movieId, imdbId, tmdbId
    """
    data_dir = data_dir or DATA_DIR
    links_path = data_dir / "ml-25m" / "links.csv"

    df = pd.read_csv(links_path)
    # Convert imdbId to tconst format (tt0114709 -> tt0114709)
    df["tconst"] = "tt" + df["imdbId"].astype(str).str.zfill(7)
    return df


def load_movielens_movies(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load MovieLens movie metadata.

    Returns DataFrame with columns: movieId, title, genres
    """
    data_dir = data_dir or DATA_DIR
    movies_path = data_dir / "ml-25m" / "movies.csv"

    return pd.read_csv(movies_path)


def create_id_mappings(links: pd.DataFrame) -> tuple[dict, dict]:
    """Create bidirectional mappings between MovieLens and IMDb IDs.

    Returns:
        (tconst_to_movieid, movieid_to_tconst) dictionaries
    """
    tconst_to_movieid = dict(zip(links["tconst"], links["movieId"]))
    movieid_to_tconst = dict(zip(links["movieId"], links["tconst"]))
    return tconst_to_movieid, movieid_to_tconst


def load_user_ratings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load user's personal IMDb ratings.

    Returns DataFrame with columns including: Const (tconst), Your Rating, Title, etc.
    """
    data_dir = data_dir or DATA_DIR
    user_path = data_dir / "user_ratings.csv"

    df = pd.read_csv(user_path)
    # Rename 'Const' to 'tconst' for consistency
    df = df.rename(columns={"Const": "tconst", "Your Rating": "rating"})
    return df


def map_user_ratings_to_movielens(
    user_ratings: pd.DataFrame,
    tconst_to_movieid: dict,
) -> pd.DataFrame:
    """Map user's IMDb ratings to MovieLens movie IDs.

    Returns DataFrame with movieId added, filtered to movies that exist in MovieLens.
    """
    df = user_ratings.copy()
    df["movieId"] = df["tconst"].map(tconst_to_movieid)

    # Report coverage
    total = len(df)
    mapped = df["movieId"].notna().sum()
    print(f"User ratings: {mapped}/{total} mapped to MovieLens ({100*mapped/total:.1f}%)")

    # Filter to mapped movies only
    return df[df["movieId"].notna()].copy()


def load_imdb_basics(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load IMDb title basics for metadata enrichment."""
    data_dir = data_dir or DATA_DIR
    basics_path = data_dir / "title.basics.tsv"

    return pd.read_csv(
        basics_path,
        sep="\t",
        usecols=["tconst", "primaryTitle", "startYear", "runtimeMinutes", "genres"],
        dtype={"startYear": str, "runtimeMinutes": str},
        na_values="\\N",
    )


def load_imdb_ratings(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load IMDb aggregate ratings."""
    data_dir = data_dir or DATA_DIR
    ratings_path = data_dir / "title.ratings.tsv"

    return pd.read_csv(
        ratings_path,
        sep="\t",
        dtype={"averageRating": np.float32, "numVotes": np.int32},
    )


def build_ratings_matrix(
    ratings: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build sparse user-movie ratings matrix.

    Returns:
        matrix: Sparse CSR matrix of shape (n_users, n_movies)
        user_ids: Array of user IDs (row index -> userId)
        movie_ids: Array of movie IDs (col index -> movieId)
    """
    # Create mappings from IDs to matrix indices
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    # Build sparse matrix
    row_indices = ratings["userId"].map(user_to_idx).values
    col_indices = ratings["movieId"].map(movie_to_idx).values
    values = ratings["rating"].values

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float32,
    )

    return matrix, user_ids, movie_ids


def compute_user_stats(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for each user (row) in ratings matrix.

    Returns:
        means: Array of user mean ratings
        stds: Array of user rating standard deviations
    """
    n_users = matrix.shape[0]
    means = np.zeros(n_users, dtype=np.float32)
    stds = np.zeros(n_users, dtype=np.float32)

    for i in range(n_users):
        row = matrix.getrow(i).data
        if len(row) > 0:
            means[i] = row.mean()
            stds[i] = row.std() if len(row) > 1 else 1.0

    # Avoid division by zero
    stds[stds == 0] = 1.0

    return means, stds


class RecommenderData:
    """Container for all data needed by the recommender."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

        # Lazy-loaded attributes
        self._ratings = None
        self._links = None
        self._movies = None
        self._user_ratings = None
        self._matrix = None
        self._user_ids = None
        self._movie_ids = None
        self._user_means = None
        self._user_stds = None
        self._tconst_to_movieid = None
        self._movieid_to_tconst = None
        self._movie_to_idx = None
        self._imdb_basics = None
        self._imdb_ratings = None

    def load_all(self, verbose: bool = True) -> "RecommenderData":
        """Load all datasets."""
        if verbose:
            print("Loading MovieLens ratings...")
        self._ratings = load_movielens_ratings(self.data_dir)

        if verbose:
            print("Loading MovieLens links...")
        self._links = load_movielens_links(self.data_dir)
        self._tconst_to_movieid, self._movieid_to_tconst = create_id_mappings(self._links)

        if verbose:
            print("Loading MovieLens movies...")
        self._movies = load_movielens_movies(self.data_dir)

        if verbose:
            print("Loading user ratings...")
        self._user_ratings = load_user_ratings(self.data_dir)
        self._user_ratings = map_user_ratings_to_movielens(
            self._user_ratings, self._tconst_to_movieid
        )

        if verbose:
            print("Building ratings matrix...")
        self._matrix, self._user_ids, self._movie_ids = build_ratings_matrix(self._ratings)
        self._movie_to_idx = {mid: idx for idx, mid in enumerate(self._movie_ids)}

        if verbose:
            print("Computing user statistics...")
        self._user_means, self._user_stds = compute_user_stats(self._matrix)

        if verbose:
            n_users, n_movies = self._matrix.shape
            n_ratings = self._matrix.nnz
            print(f"Loaded: {n_users:,} users, {n_movies:,} movies, {n_ratings:,} ratings")

        return self

    @property
    def ratings(self) -> pd.DataFrame:
        return self._ratings

    @property
    def matrix(self) -> sparse.csr_matrix:
        return self._matrix

    @property
    def user_ids(self) -> np.ndarray:
        return self._user_ids

    @property
    def movie_ids(self) -> np.ndarray:
        return self._movie_ids

    @property
    def movie_to_idx(self) -> dict:
        return self._movie_to_idx

    @property
    def user_means(self) -> np.ndarray:
        return self._user_means

    @property
    def user_stds(self) -> np.ndarray:
        return self._user_stds

    @property
    def user_ratings(self) -> pd.DataFrame:
        return self._user_ratings

    @property
    def tconst_to_movieid(self) -> dict:
        return self._tconst_to_movieid

    @property
    def movieid_to_tconst(self) -> dict:
        return self._movieid_to_tconst

    @property
    def movies(self) -> pd.DataFrame:
        return self._movies

    def load_imdb_metadata(self, verbose: bool = True) -> "RecommenderData":
        """Load IMDb metadata for enrichment (optional, call after load_all)."""
        if verbose:
            print("Loading IMDb basics...")
        self._imdb_basics = load_imdb_basics(self.data_dir)

        if verbose:
            print("Loading IMDb ratings...")
        self._imdb_ratings = load_imdb_ratings(self.data_dir)

        return self

    @property
    def imdb_basics(self) -> pd.DataFrame:
        return self._imdb_basics

    @property
    def imdb_ratings(self) -> pd.DataFrame:
        return self._imdb_ratings
