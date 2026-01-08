"""JSONL logging for preference elicitation."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schemas import Comparison, Session


class ComparisonLogger:
    """Append-only JSONL logger for comparisons, sessions, and ratings."""

    def __init__(
        self,
        comparisons_path: str = "data/pairwise_comparisons.jsonl",
        sessions_path: str = "data/sessions.jsonl",
        ratings_path: str = "data/rating_events.jsonl",
    ):
        self.comparisons_path = Path(comparisons_path)
        self.sessions_path = Path(sessions_path)
        self.ratings_path = Path(ratings_path)

        # Ensure parent directories exist
        self.comparisons_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_path.parent.mkdir(parents=True, exist_ok=True)
        self.ratings_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize sequential ID counter
        self._next_id = self._get_max_comparison_id() + 1

    def _get_max_comparison_id(self) -> int:
        """Scan existing comparisons to find the highest ID number."""
        if not self.comparisons_path.exists():
            return 0

        max_id = 0
        pattern = re.compile(r"cmp_(\d+)")

        with open(self.comparisons_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        match = pattern.match(data.get("id", ""))
                        if match:
                            max_id = max(max_id, int(match.group(1)))
                    except json.JSONDecodeError:
                        continue

        return max_id

    def log_comparison(self, comparison: Comparison) -> str:
        """Append a comparison to the log.

        Assigns a sequential ID and returns it.
        """
        # Assign sequential ID
        comparison.id = f"cmp_{self._next_id:06d}"
        self._next_id += 1

        if comparison.timestamp is None:
            comparison.timestamp = datetime.utcnow()

        with open(self.comparisons_path, "a") as f:
            f.write(json.dumps(comparison.to_dict()) + "\n")

        return comparison.id

    def log_session(self, session: Session) -> None:
        """Append a session summary to the log."""
        with open(self.sessions_path, "a") as f:
            f.write(json.dumps(session.to_dict()) + "\n")

    def load_comparisons(self) -> list[dict]:
        """Load all comparisons from the log."""
        if not self.comparisons_path.exists():
            return []

        comparisons = []
        with open(self.comparisons_path) as f:
            for line in f:
                if line.strip():
                    comparisons.append(json.loads(line))
        return comparisons

    def load_sessions(self) -> list[dict]:
        """Load all sessions from the log."""
        if not self.sessions_path.exists():
            return []

        sessions = []
        with open(self.sessions_path) as f:
            for line in f:
                if line.strip():
                    sessions.append(json.loads(line))
        return sessions

    def get_session_comparisons(self, session_id: str) -> list[dict]:
        """Get all comparisons for a specific session."""
        return [c for c in self.load_comparisons() if c.get("session_id") == session_id]

    def count_comparisons(self) -> int:
        """Count total comparisons logged."""
        return len(self.load_comparisons())

    def count_sessions(self) -> int:
        """Count total sessions logged."""
        return len(self.load_sessions())

    def get_comparisons_after(self, watermark: int) -> list[dict]:
        """Get all comparisons with ID > watermark.

        Args:
            watermark: The last processed comparison number (e.g., 35 means
                       cmp_000035 was the last one used). Pass 0 to get all.

        Returns:
            List of comparison dicts with ID > watermark, in order.
        """
        pattern = re.compile(r"cmp_(\d+)")
        result = []

        for comparison in self.load_comparisons():
            match = pattern.match(comparison.get("id", ""))
            if match:
                cmp_num = int(match.group(1))
                if cmp_num > watermark:
                    result.append(comparison)

        # Sort by ID to ensure order
        result.sort(key=lambda c: int(pattern.match(c["id"]).group(1)))
        return result

    def get_max_comparison_number(self) -> int:
        """Get the highest comparison number in the log."""
        return self._get_max_comparison_id()

    def log_rating(
        self,
        tconst: str,
        title: str,
        year: Optional[int],
        rating: float,
        confidence_low: float,
        confidence_high: float,
        n_comparisons: int,
        session_id: str,
        source: str = "elicitation",
        model_uncertainty: Optional[float] = None,
    ) -> None:
        """Log a rating event.

        Args:
            model_uncertainty: Model's predicted std for this movie before rating.
                Used for information-weighted updates. If None (e.g., movie not
                in model), a default prior will be used during factor updates.
        """
        event = {
            "tconst": tconst,
            "title": title,
            "year": year,
            "rating": round(rating, 1),
            "confidence_low": round(confidence_low, 1),
            "confidence_high": round(confidence_high, 1),
            "n_comparisons": n_comparisons,
            "session_id": session_id,
            "source": source,
            "timestamp": datetime.utcnow().isoformat(),
            "model_uncertainty": round(model_uncertainty, 3) if model_uncertainty else None,
        }

        with open(self.ratings_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def load_ratings(self) -> list[dict]:
        """Load all rating events from the log."""
        if not self.ratings_path.exists():
            return []

        ratings = []
        with open(self.ratings_path) as f:
            for line in f:
                if line.strip():
                    ratings.append(json.loads(line))
        return ratings

    def get_current_ratings(self) -> dict[str, float]:
        """Get current rating for each movie (latest wins)."""
        ratings = {}
        for event in self.load_ratings():
            ratings[event["tconst"]] = event["rating"]
        return ratings

    def get_ratings_after(self, watermark_timestamp: Optional[str]) -> list[dict]:
        """Get all rating events after a timestamp.

        Args:
            watermark_timestamp: ISO format timestamp. Pass None to get all.

        Returns:
            List of rating event dicts after the watermark, in order.
        """
        ratings = self.load_ratings()

        if watermark_timestamp is None:
            return ratings

        result = [r for r in ratings if r["timestamp"] > watermark_timestamp]
        result.sort(key=lambda r: r["timestamp"])
        return result

    def get_max_rating_timestamp(self) -> Optional[str]:
        """Get the timestamp of the most recent rating event."""
        ratings = self.load_ratings()
        if not ratings:
            return None
        return max(r["timestamp"] for r in ratings)
