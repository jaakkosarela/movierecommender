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

| Phase | Description |
|-------|-------------|
| V2.1 | Implement basic IRT model with fixed K |
| V2.2 | Add variational inference with stochastic updates |
| V2.3 | Non-symmetric priors for identifiability |
| V2.4 | Uncertainty-aware recommendations |
| V2.5 | Compare V1 vs V2 on held-out ratings |

## Success Metrics

1. **Coverage**: % of IMDb catalog with predictions
2. **Confidence**: Average number of neighbors per prediction
3. **Diversity**: Genre distribution of recommendations
4. **User validation**: Do recommendations match user preferences?

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
duckdb>=1.0.0
```

## Next Steps

1. [ ] Implement data_loader.py - join MovieLens + IMDb
2. [ ] Implement similarity.py - Pearson correlation
3. [ ] Implement recommender.py - prediction engine
4. [ ] Build CLI for generating recommendations
5. [ ] Test with user's 35 ratings
6. [ ] Evaluate and tune K parameter
