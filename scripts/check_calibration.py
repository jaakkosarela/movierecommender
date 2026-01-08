#!/usr/bin/env python3
"""Check model calibration on user's known ratings."""

import torch
import numpy as np
from src.data_loader import RecommenderData
from src.irt_model import IRTModel, IRTConfig, fit_new_user

# Load data
print("Loading data...")
data = RecommenderData().load_all(verbose=False)

# Build mappings
movie_id_to_idx = {mid: idx for idx, mid in enumerate(data.movie_ids)}

# Load model
print("Loading model...")
checkpoint = torch.load("models/irt_v1.pt", weights_only=False)
config = IRTConfig(**checkpoint["config"])
model = IRTModel(data.matrix.shape[0], data.matrix.shape[1], config)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"\nModel config: K={config.n_factors}, global_mean={model.global_mean.item():.3f}")
print(f"Noise std: {model.noise_std.item():.3f}")

# Fit user factors
user_ratings_dict = {}
for _, row in data.user_ratings.iterrows():
    movie_id = int(row["movieId"])
    if movie_id in movie_id_to_idx:
        item_idx = movie_id_to_idx[movie_id]
        user_ratings_dict[item_idx] = float(row["rating"])

print(f"\nFitting user factors from {len(user_ratings_dict)} ratings...")
user_mu, user_log_std, user_bias = fit_new_user(model, user_ratings_dict, n_iter=200, lr=0.1)
print(f"User bias: {user_bias.item():.3f}")
print(f"User factor norm: {user_mu.norm().item():.3f}")

# Check predictions on known ratings
print("\n" + "="*80)
print("CALIBRATION CHECK: Predictions vs Actual Ratings")
print("="*80)

movies_df = data.movies.set_index("movieId")

results = []
with torch.no_grad():
    for item_idx, actual_rating in user_ratings_dict.items():
        movie_id = data.movie_ids[item_idx]

        # Predict
        item_mu = model.item_mu[item_idx]
        item_bias = model.item_bias_mu[item_idx]

        pred = (user_mu * item_mu).sum() + user_bias + item_bias + model.global_mean
        pred = pred.item()

        # Get title
        title = movies_df.loc[movie_id, "title"] if movie_id in movies_df.index else f"Movie {movie_id}"
        if len(title) > 35:
            title = title[:32] + "..."

        results.append((title, actual_rating, pred, pred - actual_rating))

# Sort by actual rating (show favorites first)
results.sort(key=lambda x: -x[1])

print(f"\n{'Title':<38} {'Actual':>7} {'Pred':>7} {'Error':>7}")
print("-"*60)
for title, actual, pred, error in results:
    print(f"{title:<38} {actual:>7.1f} {pred:>7.2f} {error:>+7.2f}")

print("-"*60)

# Summary stats
actuals = [r[1] for r in results]
preds = [r[2] for r in results]
errors = [r[3] for r in results]

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(np.array(errors)**2))
corr = np.corrcoef(actuals, preds)[0,1]
bias = np.mean(errors)

print(f"\nSummary (n={len(results)}):")
print(f"  MAE:  {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  Correlation: {corr:.3f}")
print(f"  Bias: {bias:+.2f} (positive = over-predicting)")
print(f"  Pred range: [{min(preds):.1f}, {max(preds):.1f}]")
print(f"  Actual range: [{min(actuals):.1f}, {max(actuals):.1f}]")
