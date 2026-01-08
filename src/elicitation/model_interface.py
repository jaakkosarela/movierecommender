"""Interface to IRT model for preference elicitation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.irt_model import IRTModel, fit_new_user


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    version: str
    n_factors: int
    n_items: int
    global_mean: float


class ModelInterface:
    """Interface to trained IRT model for getting predictions.

    Handles:
    - Loading trained model
    - Mapping tconst → MovieLens ID → item index
    - Loading user factors from checkpoint (or fitting from ratings)
    - Getting predictions with uncertainty
    """

    def __init__(
        self,
        model_path: str = "models/irt_v1.pt",
        links_path: str = "data/ml-25m/links.csv",
        user_ratings_path: str = "data/user_ratings.csv",
        user_checkpoint_path: str = "models/user_theta.pt",
    ):
        self.model_path = model_path
        self.links_path = links_path
        self.user_ratings_path = user_ratings_path
        self.user_checkpoint_path = user_checkpoint_path

        self._model: Optional[IRTModel] = None
        self._tconst_to_item_idx: Optional[dict[str, int]] = None
        self._item_idx_to_tconst: Optional[dict[int, str]] = None
        self._user_ratings: Optional[dict[str, float]] = None  # tconst -> rating
        self._user_factors: Optional[tuple] = None  # (mu, log_std, bias_mu)
        self._model_info: Optional[ModelInfo] = None
        self._using_checkpoint: bool = False

    def _load_model(self) -> None:
        """Load trained model from disk."""
        if self._model is not None:
            return

        print(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

        # Get config and dimensions from checkpoint
        config = checkpoint["config"]
        model_state = checkpoint["model_state"]
        self._movie_ids = checkpoint["movie_ids"]  # MovieLens movieId in order
        user_ids = checkpoint["user_ids"]

        n_users = len(user_ids)
        n_items = len(self._movie_ids)

        from src.irt_model import IRTConfig

        irt_config = IRTConfig(**config)
        self._model = IRTModel(n_users, n_items, irt_config)
        self._model.load_state_dict(model_state)
        self._model.eval()

        # Build movieId -> item_idx mapping
        self._movieid_to_idx = {mid: idx for idx, mid in enumerate(self._movie_ids)}

        # Extract version from path
        version = Path(self.model_path).stem

        self._model_info = ModelInfo(
            version=version,
            n_factors=irt_config.n_factors,
            n_items=n_items,
            global_mean=float(self._model.global_mean.item()),
        )

        print(f"Loaded model: {version} ({n_items:,} items, K={irt_config.n_factors})")

    def _load_mappings(self) -> None:
        """Load MovieLens links to map tconst -> item index."""
        if self._tconst_to_item_idx is not None:
            return

        # Ensure model is loaded (we need _movieid_to_idx)
        self._load_model()

        print("Loading MovieLens links...")
        links_df = pd.read_csv(self.links_path)

        # Build tconst -> item_idx using model's movie mapping
        links_df["tconst"] = "tt" + links_df["imdbId"].astype(str).str.zfill(7)
        links_df["item_idx"] = links_df["movieId"].map(self._movieid_to_idx)

        # Only keep movies that are in the model
        links_df = links_df.dropna(subset=["item_idx"])
        links_df["item_idx"] = links_df["item_idx"].astype(int)

        self._tconst_to_item_idx = dict(zip(links_df["tconst"], links_df["item_idx"]))
        self._item_idx_to_tconst = {v: k for k, v in self._tconst_to_item_idx.items()}

        print(f"Loaded {len(self._tconst_to_item_idx):,} tconst -> item mappings")

    def _load_user_ratings(self) -> None:
        """Load user's ratings from CSV."""
        if self._user_ratings is not None:
            return

        print(f"Loading user ratings from {self.user_ratings_path}...")
        df = pd.read_csv(self.user_ratings_path)

        # Support multiple column naming conventions
        tconst_col = None
        rating_col = None

        for col in ["tconst", "imdb_id", "Const"]:
            if col in df.columns:
                tconst_col = col
                break

        for col in ["rating", "Your Rating"]:
            if col in df.columns:
                rating_col = col
                break

        if tconst_col is None:
            raise ValueError("User ratings CSV must have 'tconst', 'imdb_id', or 'Const' column")
        if rating_col is None:
            raise ValueError("User ratings CSV must have 'rating' or 'Your Rating' column")

        self._user_ratings = dict(zip(df[tconst_col], df[rating_col]))
        print(f"Loaded {len(self._user_ratings)} user ratings")

    def _load_user_checkpoint(self) -> bool:
        """Try to load user factors from checkpoint.

        Returns:
            True if checkpoint loaded, False otherwise.
        """
        checkpoint_path = Path(self.user_checkpoint_path)
        if not checkpoint_path.exists():
            return False

        try:
            print(f"Loading user checkpoint from {self.user_checkpoint_path}...")
            checkpoint = torch.load(
                self.user_checkpoint_path, map_location="cpu", weights_only=False
            )

            user_mu = checkpoint["theta_mu"]
            user_log_std = checkpoint["theta_log_std"]
            user_bias_mu = torch.tensor(checkpoint["bias_mu"])

            self._user_factors = (user_mu, user_log_std, user_bias_mu)
            self._using_checkpoint = True

            n_comp = checkpoint.get("n_comparisons_used", 0)
            n_ratings = checkpoint.get("n_ratings_used", 0)
            print(f"Loaded checkpoint: {n_ratings} ratings, {n_comp} comparisons")
            print(f"User bias: {user_bias_mu.item():.2f}")

            return True
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return False

    def _fit_user_factors(self) -> None:
        """Load user factors from checkpoint, or fit from ratings if no checkpoint."""
        if self._user_factors is not None:
            return

        self._load_model()
        self._load_mappings()
        self._load_user_ratings()

        # Try to load from checkpoint first
        if self._load_user_checkpoint():
            return

        # Fall back to fitting from ratings
        item_ratings = {}
        for tconst, rating in self._user_ratings.items():
            if tconst in self._tconst_to_item_idx:
                item_idx = self._tconst_to_item_idx[tconst]
                # Scale from 1-10 to MovieLens 0.5-5 scale
                scaled_rating = (rating - 1) / 9 * 4.5 + 0.5
                item_ratings[item_idx] = scaled_rating

        print(f"No checkpoint found. Fitting user factors from {len(item_ratings)} ratings...")
        user_mu, user_log_std, user_bias_mu = fit_new_user(
            self._model, item_ratings, n_iter=200, lr=0.1
        )
        self._user_factors = (user_mu, user_log_std, user_bias_mu)

        print(f"User bias: {user_bias_mu.item():.2f}")

    def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        self._load_model()
        return self._model_info

    def is_in_model(self, tconst: str) -> bool:
        """Check if a movie is in the model (has item factors)."""
        self._load_mappings()
        return tconst in self._tconst_to_item_idx

    def get_prediction(self, tconst: str) -> Optional[tuple[float, float]]:
        """Get prediction for a movie.

        Args:
            tconst: IMDb tconst

        Returns:
            (mean, std) prediction on 1-10 scale, or None if not in model
        """
        self._fit_user_factors()

        if tconst not in self._tconst_to_item_idx:
            return None

        item_idx = self._tconst_to_item_idx[tconst]
        item_indices = torch.tensor([item_idx])

        # Get user factors
        user_mu, user_log_std, user_bias_mu = self._user_factors

        with torch.no_grad():
            # Item factors from model
            item_mu = self._model.item_mu[item_indices]
            item_var = torch.exp(2 * self._model.item_log_std[item_indices])
            item_bias_mu = self._model.item_bias_mu[item_indices]

            # User factors
            user_var = torch.exp(2 * user_log_std)

            # Expected rating
            mean_rating = (user_mu * item_mu).sum(dim=1)
            mean_rating = mean_rating + user_bias_mu + item_bias_mu + self._model.global_mean

            # Variance
            var_rating = (user_var * item_var).sum(dim=1)
            var_rating = var_rating + (user_var * item_mu**2).sum(dim=1)
            var_rating = var_rating + (user_mu**2 * item_var).sum(dim=1)
            var_rating = var_rating + self._model.noise_std**2

            std_rating = torch.sqrt(var_rating)

            # Scale from MovieLens 0.5-5 to 1-10
            mean_scaled = (mean_rating.item() - 0.5) / 4.5 * 9 + 1
            std_scaled = std_rating.item() / 4.5 * 9

        return mean_scaled, std_scaled

    def get_predictions_for_rated_movies(self) -> dict[str, tuple[float, float]]:
        """Get predictions for all of user's rated movies.

        Returns:
            Dict mapping tconst -> (mean, std) predictions
        """
        self._fit_user_factors()

        predictions = {}
        for tconst in self._user_ratings:
            pred = self.get_prediction(tconst)
            if pred is not None:
                predictions[tconst] = pred

        return predictions

    def get_user_rating(self, tconst: str) -> Optional[float]:
        """Get user's actual rating for a movie."""
        self._load_user_ratings()
        return self._user_ratings.get(tconst)

    def get_rated_tconsts(self) -> list[str]:
        """Get list of tconsts the user has rated."""
        self._load_user_ratings()
        return list(self._user_ratings.keys())

    def get_rated_movies_in_model(self) -> list[str]:
        """Get tconsts of rated movies that are in the model."""
        self._load_mappings()
        self._load_user_ratings()

        return [tc for tc in self._user_ratings if tc in self._tconst_to_item_idx]
