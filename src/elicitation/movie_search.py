"""Movie search with fuzzy matching."""

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .schemas import Movie


@dataclass
class SearchResult:
    """A search result with match score."""

    movie: Movie
    score: float  # 0-1, higher is better


class MovieSearcher:
    """Search for movies in IMDb with fuzzy matching."""

    def __init__(
        self,
        imdb_basics_path: str = "data/title.basics.tsv",
        movielens_links_path: str = "data/ml-25m/links.csv",
    ):
        self.imdb_basics_path = imdb_basics_path
        self.movielens_links_path = movielens_links_path

        self._imdb_df: Optional[pd.DataFrame] = None
        self._movielens_lookup: Optional[dict[str, int]] = None

    def _load_data(self) -> None:
        """Load IMDb and MovieLens data."""
        if self._imdb_df is not None:
            return

        print("Loading IMDb data for search...")

        # Load IMDb basics (movies and series)
        self._imdb_df = pd.read_csv(
            self.imdb_basics_path,
            sep="\t",
            usecols=["tconst", "primaryTitle", "startYear", "genres", "titleType"],
            dtype={"startYear": str},
            na_values="\\N",
        )

        # Filter to movies and TV series
        self._imdb_df = self._imdb_df[
            self._imdb_df["titleType"].isin(["movie", "tvMovie", "tvSeries", "tvMiniSeries"])
        ].copy()

        # Clean up year
        self._imdb_df["year"] = pd.to_numeric(
            self._imdb_df["startYear"], errors="coerce"
        ).astype("Int64")

        # Create lowercase title for matching
        self._imdb_df["title_lower"] = self._imdb_df["primaryTitle"].str.lower()

        # Load MovieLens links
        links_df = pd.read_csv(self.movielens_links_path)
        links_df["tconst"] = "tt" + links_df["imdbId"].astype(str).str.zfill(7)
        self._movielens_lookup = dict(zip(links_df["tconst"], links_df["movieId"]))

        print(f"Loaded {len(self._imdb_df):,} movies")

    def search(
        self,
        query: str,
        limit: int = 10,
        year: Optional[int] = None,
    ) -> list[SearchResult]:
        """Search for movies matching query.

        Args:
            query: Search query (title or partial title)
            limit: Maximum number of results
            year: Optional year filter

        Returns:
            List of SearchResult ordered by match score
        """
        self._load_data()

        query_lower = query.lower().strip()
        query_words = set(query_lower.split())

        results = []

        for _, row in self._imdb_df.iterrows():
            title = row["primaryTitle"]
            title_lower = row["title_lower"]
            movie_year = row["year"]

            # Skip if title is NaN
            if pd.isna(title_lower):
                continue

            # Year filter
            if year is not None and pd.notna(movie_year) and movie_year != year:
                continue

            # Compute match score
            score = self._match_score(query_lower, query_words, title_lower)

            if score > 0:
                genres = (
                    row["genres"].split(",") if pd.notna(row["genres"]) else []
                )

                movie = Movie(
                    tconst=row["tconst"],
                    title=title,
                    year=int(movie_year) if pd.notna(movie_year) else None,
                    genres=genres,
                    movielens_id=self._movielens_lookup.get(row["tconst"]),
                    title_type=row["titleType"],
                )

                results.append(SearchResult(movie=movie, score=score))

        # Sort by score descending, then by year descending (newer first)
        results.sort(key=lambda r: (-r.score, -(r.movie.year or 0)))

        return results[:limit]

    def _match_score(
        self, query_lower: str, query_words: set[str], title_lower: str
    ) -> float:
        """Compute match score between query and title.

        Returns 0-1 score, higher is better.
        """
        # Exact match
        if query_lower == title_lower:
            return 1.0

        # Starts with query
        if title_lower.startswith(query_lower):
            return 0.9

        # Contains query as substring
        if query_lower in title_lower:
            return 0.8

        # All query words appear in title
        title_words = set(title_lower.split())
        if query_words.issubset(title_words):
            return 0.7

        # Most query words appear
        matching_words = len(query_words & title_words)
        if matching_words > 0:
            return 0.5 * (matching_words / len(query_words))

        return 0.0

    def get_by_tconst(self, tconst: str) -> Optional[Movie]:
        """Get a movie by its IMDb tconst."""
        self._load_data()

        row = self._imdb_df[self._imdb_df["tconst"] == tconst]
        if row.empty:
            return None

        row = row.iloc[0]
        genres = row["genres"].split(",") if pd.notna(row["genres"]) else []

        return Movie(
            tconst=tconst,
            title=row["primaryTitle"],
            year=int(row["year"]) if pd.notna(row["year"]) else None,
            genres=genres,
            movielens_id=self._movielens_lookup.get(tconst),
            title_type=row["titleType"],
        )

    def is_in_movielens(self, tconst: str) -> bool:
        """Check if a movie is in MovieLens."""
        self._load_data()
        return tconst in self._movielens_lookup

    def get_movielens_id(self, tconst: str) -> Optional[int]:
        """Get MovieLens ID for a movie."""
        self._load_data()
        return self._movielens_lookup.get(tconst)
