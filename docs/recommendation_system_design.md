# Movie Recommendation System Design

## Overview

A collaborative filtering recommendation system that uses MovieLens 25M user ratings to find movies the user would likely enjoy, enriched with IMDb metadata.

## Data Sources

### MovieLens 25M
- **ratings.csv**: 25M ratings from 162K users (userId, movieId, rating, timestamp)
- **links.csv**: 62K movies mapped to IMDb IDs (movieId → imdbId)
- **movies.csv**: Movie titles and genres

### IMDb (already downloaded)
- **title.basics.tsv**: Movie metadata (year, runtime, genres)
- **title.ratings.tsv**: Aggregate ratings and vote counts
- **title.principals.tsv**: Cast and crew
- **name.basics.tsv**: Actor/director names

### User Data
- **user_ratings.csv**: 35 rated movies with IMDb IDs (tconst)

## Algorithm: User-Based Collaborative Filtering

### Step 1: Map User Ratings to MovieLens

```
User's IMDb ratings → links.csv → MovieLens movieIds
```

Convert user's 35 IMDb ratings to MovieLens movie IDs for comparison with other users.

### Step 2: Find Similar Users

For each MovieLens user, calculate similarity to our user based on overlapping movie ratings.

**Similarity Metric: Pearson Correlation**

```
similarity(u, v) = Σ(r_u - μ_u)(r_v - μ_v) / (σ_u * σ_v * n)
```

Where:
- r_u, r_v = ratings from user u and v
- μ = mean rating for each user
- σ = standard deviation
- n = number of overlapping movies

**Why Pearson over Cosine:**
- Accounts for rating scale differences (some users rate harsh, others generous)
- User rates 0.9 points higher than IMDb average → Pearson handles this

**Minimum Overlap Threshold:** 5 movies
- Users with fewer overlapping ratings are excluded (insufficient signal)

### Step 3: Select Neighborhood

Select top-K most similar users as the "neighborhood."

**K = 50** (tunable parameter)

Rationale:
- Too small (K=10): Noisy, overfits to few users
- Too large (K=500): Dilutes signal with weakly similar users
- K=50 is a common starting point

### Step 4: Predict Ratings

For each candidate movie m not yet rated by user:

```
predicted_rating(m) = μ_user + Σ sim(user, n) * (r_n,m - μ_n) / Σ |sim(user, n)|
```

Where:
- μ_user = user's mean rating
- sim(user, n) = similarity to neighbor n
- r_n,m = neighbor n's rating for movie m
- μ_n = neighbor n's mean rating

This weighted average adjusts for each neighbor's rating tendencies.

### Step 5: Filter and Rank

**Filters:**
1. Minimum neighbor votes: At least 3 neighbors must have rated the movie
2. IMDb rating threshold: ≥ 7.0 (user prefers quality films)
3. IMDb vote count: ≥ 10,000 (avoid obscure films)
4. Exclude already-rated movies

**Ranking:**
Primary: Predicted rating (descending)
Secondary: Number of neighbor votes (confidence)

### Step 6: Enrich with IMDb Metadata

Join recommendations with IMDb data to provide:
- Director and top cast
- Genres
- IMDb rating and vote count
- Year and runtime

## Output Format

```
Rank | Title | Predicted | IMDb | Year | Genres | Director
-----+-------+-----------+------+------+--------+---------
1    | ...   | 9.2       | 8.1  | 2019 | ...    | ...
```

## Architecture

```
src/
├── __init__.py
├── data_loader.py      # Load and join MovieLens + IMDb data
├── similarity.py       # User similarity calculations
├── recommender.py      # Core recommendation engine
└── cli.py              # Command-line interface

scripts/
├── build_user_index.py # Precompute user similarity matrix
└── recommend.py        # Generate recommendations for user
```

## Performance Considerations

### Challenge: 25M ratings × 162K users

**Solution: Sparse Matrix + Precomputation**

1. Store ratings as sparse matrix (scipy.sparse)
2. Precompute user means and standard deviations
3. Only compute similarity for users with ≥5 overlapping movies
4. Cache similarity scores for repeated queries

**Expected Performance:**
- Initial similarity computation: ~2-5 minutes
- Recommendation generation: <5 seconds

## Future Enhancements (Out of Scope for V1)

1. **Item-based CF**: Precompute movie-movie similarity (more stable)
2. **Hybrid**: Combine CF predictions with content-based signals
3. **Temporal weighting**: Recent ratings weighted higher
4. **Genre boosting**: Boost predictions for preferred genres

---

## V2: Item Response Theory (IRT) Latent Factor Model

A more principled probabilistic approach using latent factors for both users and items.

### Motivation

The Pearson-based approach (V1) has limitations:
- Only considers pairwise user similarity, not global structure
- No uncertainty quantification on predictions
- Doesn't learn interpretable latent dimensions (e.g., "prefers arthouse", "likes action")

IRT-style latent factor models address these by learning a shared latent space where user preferences and movie characteristics interact.

### Model Formulation

**Generative model:**

```
For each user u:
    θ_u ~ N(0, Σ_user)           # User latent factors (K dimensions)

For each movie m:
    β_m ~ N(0, Σ_item)           # Movie latent factors (K dimensions)
    b_m ~ N(0, σ_b²)             # Movie bias (popularity)

For each rating r_{u,m}:
    μ_{u,m} = θ_u · β_m + b_m + b_u
    r_{u,m} ~ N(μ_{u,m}, σ²)     # Or ordinal/truncated for discrete ratings
```

Where:
- θ_u ∈ ℝ^K = user u's latent preferences
- β_m ∈ ℝ^K = movie m's latent characteristics
- b_u, b_m = user and movie biases (account for "harsh raters" and "popular movies")
- K = latent dimensionality (typically 10-50)

### Inference: Variational Approximation

**Why variational inference:**
- 25M ratings makes full MCMC prohibitively slow
- VI provides point estimates with uncertainty
- Scales well with stochastic mini-batch updates

**Variational family:**

```
q(θ, β, b) = ∏_u q(θ_u) · ∏_m q(β_m) · q(b_m)

q(θ_u) = N(μ_θ^u, diag(σ_θ^u)²)
q(β_m) = N(μ_β^m, diag(σ_β^m)²)
```

Mean-field assumption (independence) enables efficient coordinate ascent updates alternating between user and item factors.

**Optimization:**
- Stochastic VI with mini-batches of ratings
- Adam optimizer for variational parameters
- ELBO (Evidence Lower Bound) as objective

### Handling Rotational Invariance

**The problem:** For any orthogonal matrix Q:
```
θ_u · β_m = (θ_u Q) · (β_m Q)
```

The likelihood is invariant to rotations in latent space, causing:
- Non-identifiability of latent dimensions
- Mode-hopping during inference
- Inconsistent solutions across runs

**Solution: Non-symmetric regularization**

Break rotational symmetry by using different prior variances per dimension:

```
θ_u[k] ~ N(0, σ_k²)    where σ_1 > σ_2 > ... > σ_K
β_m[k] ~ N(0, τ_k²)    where τ_1 > τ_2 > ... > τ_K
```

For example, with K=20:
```python
sigma_k = [1.0, 0.9, 0.8, ..., 0.1]  # Decreasing variance
```

**Why this works:**
- Dimensions become ordered by "importance" (variance explained)
- First dimensions capture dominant patterns, later dimensions capture nuance
- Variational posterior inherits this structure, converging to consistent modes
- Interpretability: Dimension 1 might be "mainstream vs. arthouse", Dimension 2 "action vs. drama", etc.

**Alternative approaches considered:**
- Lower-triangular constraint on β (like factor analysis) — more complex to implement
- Post-hoc Procrustes rotation — doesn't help during training
- Anchoring specific users/items — arbitrary choices affect results

### Implementation Considerations

**Libraries:**
- PyTorch with custom VI (most flexible)
- Pyro / NumPyro (probabilistic programming, built-in VI)
- TensorFlow Probability

**Computational strategy:**
1. Initialize with SVD on rating matrix (warm start)
2. Stochastic VI with mini-batches (~10K ratings per batch)
3. Early stopping based on held-out ELBO
4. Learn K via cross-validation or automatic relevance determination (ARD)

**Cold-start for new user:**
- With 35 ratings, estimate θ_user via MAP given fixed β_m
- Uncertainty quantification: posterior variance indicates confidence

### Prediction

For a new movie m not rated by user:

```
E[r_{u,m}] = μ_θ^u · μ_β^m + b_m + b_u

Var[r_{u,m}] = σ² + ∑_k (σ_θ^u[k])² (μ_β^m[k])² + (μ_θ^u[k])² (σ_β^m[k])²
```

The variance provides confidence intervals—recommend movies with high expected rating AND low uncertainty.

### V2 Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| V2.1 | Implement basic IRT model with fixed K | Done |
| V2.2 | Add variational inference with stochastic updates | Done |
| V2.3 | Non-symmetric priors for identifiability | Done |
| V2.4 | Uncertainty-aware recommendations | Done |
| V2.5 | Model diagnostics and validation | In Progress |
| V2.6 | Compare V1 vs V2 on held-out ratings | Planned |

### V2 Model Diagnostics

After initial training (20 epochs, K=20), recommendations showed red flags:
- All predictions > 10 (outside valid 1-10 range)
- Dominated by obscure films (<5K votes)
- Some low IMDb scores (4.1, 4.8) predicted as 10+
- High uncertainty (±1.3 to ±2.0) across all predictions

**Diagnostic checks to implement:**

#### 1. Training Convergence
```python
# Plot ELBO over epochs
plt.plot(trainer.history['elbo'])
plt.xlabel('Epoch')
plt.ylabel('ELBO')
# Should show convergence, not plateau or oscillation
```

#### 2. Prediction Distribution
```python
# Histogram of predictions for all candidate movies
# Should center around population mean (~3.5), not 10+
plt.hist(all_predictions, bins=50)
```

#### 3. Calibration on Known Ratings
**Critical check**: For movies the user has rated, compare prediction vs actual.

```python
# For each movie in user_ratings.csv:
#   - Get model's prediction
#   - Compare to actual rating
#   - Compute MAE, RMSE, correlation

# Example output:
# Movie                      | Actual | Predicted | Error
# The Big Lebowski           | 10     | ???       | ???
# Before Sunset              | 10     | ???       | ???
# The Bourne Identity        | 10     | ???       | ???
# (user's favorites - model should predict high)
```

If the model predicts 10+ for movies the user rated 5, or 5 for movies rated 10, something is wrong.

#### 4. Vote Count vs Prediction
```python
# Scatter plot: x = log(numVotes), y = predicted rating
# Check if rare movies systematically get extreme predictions
```

Hypothesis: Movies with few ratings have high-variance β factors. When sampled or point-estimated, these can produce outlier predictions.

#### 5. Latent Factor Inspection
```python
# Check item factor magnitudes
item_norms = torch.norm(model.item_mu, dim=1)
# Are some items extreme? Correlate with vote count.

# Check user factor magnitude
user_norm = torch.norm(user_mu)
# Is it reasonable?
```

#### 6. Rating Scale Check
MovieLens uses 0.5-5.0 scale, user ratings are 1-10.
- Verify data_loader handles this correctly
- Check global_mean learned by model (should be ~3.5 for MovieLens)
- User bias should account for scale difference

**Potential fixes if diagnostics reveal issues:**

| Problem | Fix |
|---------|-----|
| Predictions unbounded | Clip to [1,10] or use sigmoid output |
| Rare items get extreme scores | Filter by min vote count, or add popularity prior |
| Poor calibration on known ratings | Check rating scale handling, retrain with adjusted priors |
| ELBO not converged | More epochs, adjust learning rate |
| Item factors exploding | Stronger regularization (tighter priors) |

## Success Metrics

### V1 (Pearson-based)
1. **Coverage**: % of catalog with predictions
2. **Confidence**: Number of neighbors per prediction
3. **MAE on held-out**: 1.18 (from leave-one-out evaluation)

### V2 (IRT model)
1. **ELBO convergence**: Training loss should plateau
2. **Calibration**: Predictions for known ratings should correlate with actuals
3. **Prediction range**: Should be within [1, 10], centered reasonably
4. **Uncertainty calibration**: High-confidence predictions should be more accurate

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
duckdb>=1.0.0
torch>=2.0.0
tqdm>=4.65.0
```

## Next Steps

1. [x] Implement data_loader.py - join MovieLens + IMDb
2. [x] Implement similarity.py - Pearson correlation (V1)
3. [x] Implement recommender.py - prediction engine (V1)
4. [x] Implement irt_model.py - IRT with VI (V2)
5. [x] Build CLI for generating recommendations
6. [ ] **Run model diagnostics on trained model**
7. [ ] Fix any issues revealed by diagnostics
8. [ ] Compare V1 vs V2 predictions
9. [ ] Implement preference elicitation system (V3)
