# Session Continuity Summary

Reverse chronological order (newest first).

---

## 2025-12-28 (Session 8): P7 Complete - Update Factors from Comparisons

### Current Status
P1-P7 complete. Full preference elicitation workflow operational. Only P8 (analysis tools) remains.

### Completed Work

1. **Information-Weighted Update Design**:
   - Analyzed information content: 1 rating ≈ 2.5 bits, 1 max-entropy comparison ≈ 1 bit
   - Comparisons weighted by stored entropy / 2.5 (normalized to rating scale)
   - Elicitation ratings weighted by model_uncertainty (capped at 2.0)
   - IMDb ratings weighted uniformly (no uncertainty info available)

2. **User Theta Checkpoint** (`models/user_theta.pt`):
   ```python
   {
       "theta_mu": tensor[20],        # User latent factors
       "theta_log_std": tensor[20],   # Uncertainty in θ
       "bias_mu": float,              # User bias
       "comparisons_watermark": 27,   # Last comparison ID used
       "ratings_watermark": "...",    # Last rating timestamp
       "n_comparisons_used": 20,
       "n_ratings_used": 71,
   }
   ```

3. **Update Script** (`scripts/update_factors.py`):
   - Loads main model (β fixed) + user checkpoint (if exists)
   - Gets new comparisons via `get_comparisons_after(watermark)`
   - Gets new ratings via `get_ratings_after(timestamp)`
   - Combined likelihood: L_imdb + L_elicitation + L_comparisons (weighted)
   - Bradley-Terry for comparisons: P(A>B) = sigmoid(pred_A - pred_B)
   - Saves updated checkpoint with new watermarks

4. **ModelInterface Updates**:
   - Now loads from user checkpoint if available
   - Falls back to `fit_new_user()` if no checkpoint
   - Added `user_checkpoint_path` parameter

5. **Logger Updates**:
   - Added `model_uncertainty` field to rating events
   - Added `get_ratings_after(timestamp)` for incremental updates
   - Added `get_max_rating_timestamp()`

6. **CLI Updates**:
   - Rate mode now passes `model_uncertainty` when saving rating

### Full Workflow

```bash
# 1. Run calibration (logs comparisons)
PYTHONPATH=. python scripts/elicit_preferences.py calibrate --n-rounds 10

# 2. Update θ from logged data
PYTHONPATH=. python scripts/update_factors.py

# 3. Next session uses updated θ
PYTHONPATH=. python scripts/elicit_preferences.py calibrate --n-rounds 10
```

### Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| P1 | Data schema and logging utilities | **Done** |
| P2 | IMDb search with fuzzy matching | **Done** |
| P3 | MovieLens lookup (model interface) | **Done** |
| P4 | Sampling strategies | **Done** |
| P5 | CLI for calibrate mode | **Done** |
| P6 | CLI for rate mode | **Done** |
| P7 | Factor update from comparisons | **Done** |
| P8 | Analysis tools | Pending |

### Key Files Modified
- `src/elicitation/logger.py` - model_uncertainty, ratings watermark methods
- `src/elicitation/model_interface.py` - loads from user checkpoint
- `scripts/elicit_preferences.py` - passes model_uncertainty
- `scripts/update_factors.py` - **new**, updates θ from comparisons + ratings

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Entropy-weighted comparisons | High-entropy (uncertain) pairs more informative |
| Uncertainty-weighted ratings | Consistent with comparison weighting |
| Separate user checkpoint | Main model β stays fixed, only θ updates |
| Watermarks in checkpoint | Append-only logs, incremental updates |

### Next Steps (P8)
- Analysis tools for logged comparisons
- Visualize θ drift over time
- Compare predictions before/after calibration

---

## 2025-12-18 (Session 7): Preference Elicitation CLI Complete (P1-P6)

### Current Status
Preference elicitation system fully implemented and ready to use. P1-P6 complete. P7 (θ update from comparisons) and P8 (analysis tools) remain.

### Completed Work

1. **Refined Data Schema** (`schemas.py`):
   - Added `UserRatings` - user's actual ratings for compared movies
   - Added `SamplingInfo` - strategy name + entropy at selection time
   - Added `model_version` to `ModelPrediction`
   - Comparison IDs now sequential (`cmp_000001`, `cmp_000002`, ...)

2. **Model Interface** (`model_interface.py`):
   - Loads trained IRT model from checkpoint
   - Maps tconst → MovieLens ID → model item index
   - Fits user θ from ratings via `fit_new_user()`
   - `get_prediction(tconst)` returns (mean, std) on 1-10 scale
   - `get_predictions_for_rated_movies()` for calibration

3. **Enhanced Logger** (`logger.py`):
   - Sequential comparison IDs (scans file on init to continue sequence)
   - `get_comparisons_after(watermark)` for θ update script
   - `log_rating()` for rating events
   - `get_current_ratings()` returns latest rating per movie

4. **CLI** (`scripts/elicit_preferences.py`):
   - **Calibrate mode**: `PYTHONPATH=. python scripts/elicit_preferences.py calibrate --n-rounds 20`
     - Max entropy sampling (pairs where P(A>B) ≈ 0.5)
     - Excludes previously shown pairs (no repeats across sessions)
     - Model confidence shown only AFTER user chooses (no bias)
   - **Rate mode**: `PYTHONPATH=. python scripts/elicit_preferences.py rate "Oppenheimer"`
     - Fuzzy IMDb search, user selects from results
     - Checks if already rated, asks to re-rate
     - Excludes target from anchors (no self-comparison)
     - ~7 comparisons via adaptive binary search
     - Saves to `rating_events.jsonl`

5. **Data Files**:
   ```
   data/
   ├── pairwise_comparisons.jsonl   # All comparisons (calibrate + rate)
   ├── sessions.jsonl               # Session metadata
   └── rating_events.jsonl          # Final ratings (append-only history)
   ```

6. **Bug Fixes**:
   - Fixed NaN title handling in movie search
   - Fixed model checkpoint loading (different format than expected)
   - Fixed user ratings CSV column names (Const vs tconst)

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sequential comparison IDs | Enables watermark-based θ updates |
| Watermark in checkpoint (not in comparison records) | Keeps comparison log immutable/append-only |
| Exclude previous pairs in calibration | No repeats, fresh pairs each session |
| Separate `rating_events.jsonl` | Full rating history, latest wins |
| Don't show model confidence before choice | Avoid biasing user |

### Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| P1 | Data schema and logging utilities | **Done** |
| P2 | IMDb search with fuzzy matching | **Done** |
| P3 | MovieLens lookup (model interface) | **Done** |
| P4 | Sampling strategies | **Done** |
| P5 | CLI for calibrate mode | **Done** |
| P6 | CLI for rate mode | **Done** |
| P7 | Factor update from comparisons | Pending |
| P8 | Analysis tools | Pending |

### Usage

```bash
# Calibrate: refine θ by comparing rated movies
PYTHONPATH=. python scripts/elicit_preferences.py calibrate --n-rounds 20

# Rate: rate a new movie via ~7 comparisons
PYTHONPATH=. python scripts/elicit_preferences.py rate "Oppenheimer"
```

### Next Steps (P7)

1. Create `scripts/update_factors.py`:
   - Load model checkpoint with `comparisons_watermark`
   - Get new comparisons via `logger.get_comparisons_after(watermark)`
   - Update θ using Bradley-Terry likelihood
   - Save new checkpoint with updated watermark

2. Bradley-Terry likelihood:
   ```
   P(A > B | θ) = sigmoid(predicted_A - predicted_B)
   LL = Σ [y_i * log(P) + (1-y_i) * log(1-P)]
   ```

### Key Files Modified
- `src/elicitation/schemas.py` - added UserRatings, SamplingInfo, sequential IDs
- `src/elicitation/logger.py` - sequential IDs, rating events, watermark support
- `src/elicitation/model_interface.py` - new file, model loading + predictions
- `src/elicitation/movie_search.py` - fixed NaN handling
- `scripts/elicit_preferences.py` - new CLI for both modes

---

## 2025-12-18 (Session 6): Preference Elicitation Implementation (P1-P4)

### Current Status
Started implementing the preference elicitation system. Completed P1, P2, P4. P3, P5, P6 remain.

### Completed Work

1. **Clarified Use Cases** - 2 user-facing modes:
   - **Calibrate**: System asks about movies where `|predicted - actual| > threshold`
   - **Rate**: User enters movie title, system handles MovieLens lookup internally

2. **Created `src/elicitation/` package**:
   ```
   src/elicitation/
   ├── __init__.py          # Package exports
   ├── schemas.py           # Data classes
   ├── logger.py            # JSONL logging
   ├── sampler.py           # Sampling strategies
   └── movie_search.py      # IMDb fuzzy search
   ```

3. **Schemas** (`schemas.py`):
   - `Movie` - tconst, title, year, genres, movielens_id, model_prediction
   - `Comparison` - movie_a, movie_b, choice, metadata, model_prediction
   - `Session` - session tracking with use_case ("calibrate" or "rate")
   - `RatingEstimate` - rating + confidence_low/high + n_comparisons
   - `ComparisonChoice` - enum for "a" or "b"

4. **Logger** (`logger.py`):
   - `ComparisonLogger` class
   - Appends to `data/pairwise_comparisons.jsonl` and `data/sessions.jsonl`
   - Methods: `log_comparison()`, `log_session()`, `load_comparisons()`, `load_sessions()`

5. **Samplers** (`sampler.py`):
   - `MaxEntropySampler` - for calibration, picks pairs where P(A>B) ≈ 0.5
   - `AdaptiveBinarySearchSampler` - for rating, binary search through anchors
   - Uses Bradley-Terry model: P(A>B) = sigmoid(rating_A - rating_B)

6. **Movie Search** (`movie_search.py`):
   - `MovieSearcher` class - fuzzy search on IMDb title.basics.tsv
   - Scoring: exact match (1.0) > starts with (0.9) > contains (0.8) > word overlap
   - Returns `SearchResult` with Movie and score
   - Checks MovieLens via links.csv, sets `movielens_id` if found

7. **Updated design docs**:
   - Simplified to 2 user-facing modes (calibrate, rate)
   - Added selection bias section explaining why rare movies get inflated predictions
   - Documented Bayesian approach (vote-count-dependent prior) as future consideration

### Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| P1 | Data schema and logging utilities | **Done** |
| P2 | IMDb search with fuzzy matching | **Done** |
| P3 | MovieLens lookup (movie → model predictions) | Pending |
| P4 | Sampling strategies | **Done** |
| P5 | CLI for calibrate mode | Pending |
| P6 | CLI for rate mode | Pending |
| P7 | Factor update from comparisons | Pending |
| P8 | Analysis tools | Pending |

### Next Steps
1. P3: Create model interface to get predictions for movies
2. P5: Build CLI for calibrate mode
3. P6: Build CLI for rate mode
4. Test end-to-end flow

### Key Files
- `src/elicitation/` - new package (schemas, logger, sampler, movie_search)
- `docs/preference_elicitation_design.md` - updated with 2-mode structure
- `docs/recommendation_system_design.md` - added selection bias section

---

## 2025-12-17 (Session 5): Model Diagnostics & Thompson Sampling Design

### Current Status
V2 IRT model validated - calibration is good. Designed Thompson sampling with uncertainty shrinkage for recommendations.

### Completed Work

1. **Model Calibration Check** - `scripts/check_calibration.py`
   - Model predictions on known ratings: MAE 0.58, RMSE 0.76, correlation 0.55
   - Much better than V1 (MAE 1.18)
   - Predictions in valid range [7.5, 9.4], not >10 as feared
   - User bias correctly learned: +4.92

2. **Diagnosed Recommendation Issue**
   - Raw mean ranking dominated by obscure films (<5K votes)
   - Root cause: **item uncertainty scales with vote count**
     - <1K votes: uncertainty 0.52
     - 100K+ votes: uncertainty 0.045 (10x lower)
   - 73.8% of movies have <5K votes (extremely skewed distribution)
   - Vote distribution is log-normal (median 1,558, mean 18,221)

3. **Thompson Sampling Design Exploration**
   - Pure Thompson: high-uncertainty movies dominate (worse than mean)
   - Thompson + vote filter: good but excludes interesting obscure films
   - Aggregated Thompson (1000 samples): more stable but still noisy
   - P(rating ≥ 9): clean but no exploration

4. **Final Approach: Squared Log-Vote Shrinkage + Soft IMDb Floor**
   ```python
   # Shrink uncertainty for low-vote movies
   shrink = (log(votes) / log(50000))^2  # squared for steeper curve
   effective_std = std * shrink

   # Thompson sample
   score = mean + effective_std * N(0,1)

   # Penalize wild divergence from IMDb
   penalty = max(0, pred_mean - imdb_rating - 3) * 0.5
   final_score = score - penalty
   ```

5. **New Recommendation Script** - `scripts/recommend_irt.py`
   - Thompson sampling with configurable parameters
   - CLI flags: `--reference-votes`, `--imdb-tolerance`, `--imdb-penalty-weight`
   - Produces balanced recommendations: ~0 <1K, ~19 1K-10K, ~10 10K-50K, ~1 >50K

### Key Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Shrinkage curve | Squared | Linear too gentle (0.64 at 1K), squared gives 0.41 |
| Reference votes | 50,000 | Full uncertainty trusted above this |
| IMDb tolerance | 3.0 | User can like a movie 3 pts above IMDb consensus |
| IMDb penalty weight | 0.5 | Halve excess divergence, don't eliminate |

### Shrinkage Examples
| Votes | Shrink Factor |
|-------|---------------|
| 100 | 0.18 |
| 1,000 | 0.41 |
| 5,000 | 0.62 |
| 10,000 | 0.72 |
| 50,000 | 1.00 |

### Usage
```bash
# Default Thompson sampling
PYTHONPATH=. python scripts/recommend_irt.py

# With custom parameters
PYTHONPATH=. python scripts/recommend_irt.py \
  --top-n 30 \
  --reference-votes 50000 \
  --imdb-tolerance 3.0 \
  --seed 42
```

### Files Created/Modified
- `src/recommendation/` - new package for recommendation logic
  - `__init__.py` - exports ThompsonConfig, generate_recommendations, etc.
  - `thompson.py` - Thompson sampling with shrinkage and IMDb floor
- `scripts/recommend_irt.py` - refactored to use recommendation package
- `scripts/train_irt.py` - refactored to use recommendation package
- `scripts/check_calibration.py` - created (model diagnostics)
- `docs/compacting/compacting_summary.md` - updated

### Next Steps
1. Test recommendations in practice (watch some movies, provide feedback)
2. Consider training longer (current model: 20 epochs) if calibration degrades
3. Implement preference elicitation system (V3)

---

## 2025-12-17 (Session 4): IRT Model & Preference Elicitation Design

### Current Status
Built V2 recommendation system using Item Response Theory with variational inference. Designed binary preference elicitation system (not yet implemented).

### Completed Work

1. **IRT Model Implementation** - `src/irt_model.py`
   - Latent factor model: user factors θ, item factors β, biases
   - Variational inference with stochastic mini-batch training
   - **Non-symmetric priors** (σ_k decreasing from 1.0 → 0.1) to break rotational invariance
   - SVD initialization for warm start
   - `fit_new_user()` for cold-start (fit θ given ratings, fixed β)
   - Prediction with uncertainty (mean ± std)

2. **Training Script** - `scripts/train_irt.py`
   - Loads MovieLens 25M, trains IRT model
   - Fits new user factors from user_ratings.csv
   - Generates recommendations with uncertainty estimates
   - Model saving/loading support

3. **Updated Design Doc** - `docs/recommendation_system_design.md`
   - Added comprehensive V2 section on IRT approach
   - Covers model formulation, VI, rotational invariance solution

4. **Preference Elicitation Design** - `docs/preference_elicitation_design.md`
   - Two use cases: (A) calibrate existing ratings, (B) rate new movies
   - Binary comparisons: "Do you prefer A or B?"
   - Sampling strategies: max_entropy, adaptive for binary search
   - Handles movies not in MovieLens (IMDb-only) via comparison-based rating
   - Data schema: JSONL logs for all comparisons, versioned models
   - CLI interface designed (text-based, shows title/year/genres/plot)

### Key Design Decisions (Preference Elicitation)
- Only use rated movies as anchors (no "haven't seen" issue)
- Don't show existing ratings (avoid bias)
- Force choice (no "equal" option)
- 15-20 comparisons per calibration session, ~7 for new movie rating
- Log everything for experimentation with update strategies

### Test Run Results
- 3 epochs, 10 factors: ELBO improved -35M → -32M
- User bias learned: +4.98 (user rates high relative to population)
- Predictions > 10 possible (Gaussian likelihood unbounded) - clip at inference

### Issue Identified (20 epochs, 20 factors)
Full training run produced suspicious recommendations:
- All predictions > 10 (outside valid 1-10 range)
- Dominated by obscure films (<5K votes) user has never heard of
- Some with poor IMDb scores (4.1, 4.8) predicted as 10+
- High uncertainty (±1.3 to ±2.0) across all predictions

**Added diagnostics section to design doc** with 6 checks:
1. Training convergence (ELBO plot)
2. Prediction distribution
3. **Calibration on known ratings** (critical - check user's favorites)
4. Vote count vs prediction
5. Latent factor inspection
6. Rating scale check (MovieLens 0.5-5 vs user 1-10)

### Next Steps (Updated)
1. ~~Train IRT model with full parameters~~ Done, but needs diagnostics
2. **Run model diagnostics, especially calibration on known ratings**
3. Fix issues revealed by diagnostics
4. Then proceed to preference elicitation

### Training Command (Reference)
```bash
PYTHONPATH=. python scripts/train_irt.py \
  --n-factors 20 \
  --n-epochs 20 \
  --batch-size 20000 \
  --lr 0.005 \
  --save-model models/irt_v1_20epochs.pt \
  --top-n 30
```

### Next Steps
1. Train IRT model with full parameters (20 epochs)
2. Implement preference elicitation system (7 phases in design doc)
3. Start with P1: data schema and logging utilities

### Files Created/Modified
- `src/irt_model.py` - created (IRT model + VI trainer)
- `scripts/train_irt.py` - created (training script)
- `requirements.txt` - added torch, tqdm
- `docs/recommendation_system_design.md` - added V2 IRT section
- `docs/preference_elicitation_design.md` - created (full design)
- `models/` - directory created for model checkpoints

### Dependencies Added
- torch>=2.0.0
- tqdm>=4.65.0

---

## 2025-12-16 (Session 3): Sampling & Evaluation

### Current Status
Recommendation system complete with CLI. Evaluated with leave-one-out cross-validation.

### Completed Work
1. **Updated user ratings** - new IMDb export with 82 ratings (70 map to MovieLens)
   - 12 unmapped: 7 TV series + 5 movies from 2021+ (MovieLens is from 2019)

2. **Increased min_overlap to 10** - eliminated spurious perfect correlations
   - Similarity range now 0.82-0.93 (no more 1.0s)
   - Top 200 users: 3 high (≥0.9), 197 moderate (0.7-0.9)

3. **Sampling-based recommendations** - `sample_recommendations()` in recommender.py
   - Top 100 predictions as candidate pool (scores 8.6-9.8)
   - Linear weighting (13x ratio between best/worst)
   - Returns random 5 movies weighted by predicted rating
   - No scores shown to user (avoids anchoring)

4. **CLI script** - `scripts/recommend.py`
   - `python scripts/recommend.py` → 5 recommendations
   - Options: `-n 10`, `--pool 100`, `--min-overlap 10`, `--weighting linear`

5. **Leave-one-out evaluation**
   - MAE: 1.18, RMSE: 1.44, Correlation: 0.16
   - Bias: -0.94 (system under-predicts by ~1 point)
   - User rates ~1 point higher than neighbors on same movies
   - Selection bias: user rated movies they loved, not random sample

6. **Future ideas documented** - `docs/future_ideas.md`
   - Pairwise comparison system for rating coherence
   - Bias adjustment, alternative similarity metrics
   - Feedback loop, genre filtering

### Usage
```bash
python scripts/recommend.py        # 5 recommendations
python scripts/recommend.py -n 10  # 10 recommendations
python scripts/recommend.py -v     # verbose loading
```

### Key Findings
- User's ratings: mean 8.7, std 0.9, 91% are 8-10
- Pool IMDb mean: 7.9 (user rates ~0.8 above IMDb average)
- Low correlation (0.16) suggests we predict "7-8 range" for everything, not precision ranking
- System is fine for "find something good" use case

### Files Modified
- `src/recommender.py` - added `sample_recommendations()`
- `scripts/recommend.py` - created CLI
- `data/user_ratings.csv` - updated with new export
- `docs/future_ideas.md` - created
- `docs/compacting/compacting_summary.md` - updated

---

## 2024-12-16 (Session 2): Implementation Complete

### Current Status
Core recommendation system implemented and working.

### Completed Work
1. **src/data_loader.py** - Loads MovieLens + IMDb, builds sparse matrix
   - `RecommenderData` class with lazy loading
   - Matrix: 162,541 users × 59,047 movies × 25M ratings

2. **src/similarity.py** - Pearson correlation
   - `find_similar_users_vectorized()` - finds top-K similar MovieLens users
   - Computes correlation over overlapping movies only
   - Fixed bug: was computing correlation >1.0, now properly in [-1,1]

3. **src/recommender.py** - Prediction engine
   - `predict_ratings()` - weighted average from neighbors
   - `rank_recommendations()` - filter and sort
   - `enrich_with_metadata()` - add IMDb info
   - `Recommender` class - high-level interface
   - Fixed: scale MovieLens 0.5-5.0 → IMDb 1-10 scale

4. **Tested full pipeline** - generates quality recommendations

### Dependencies Installed
- duckdb, pandas, numpy, scipy

---

## 2024-12-16 (Session 1): Project Setup & Design

### Completed Work
1. Downloaded IMDb datasets (title.basics, title.ratings, title.principals, name.basics, title.akas)
2. Downloaded MovieLens 25M (25M ratings, 162K users, 62K movies)
3. Analyzed best actors/actresses/directors/writers
4. Created algorithm design document
5. Set up project structure (src/, scripts/, data/, docs/, tests/)

### Key Findings
- User profile: 35 movies, avg 8.7 rating (+0.9 vs IMDb), prefers Thriller/Action/Drama
- Directors > Actors > Actresses in average ratings
- Top: James Stewart (actor 8.05), Joan Allen (actress 7.54), Christopher Nolan (director 8.17)

### Design Decisions
- User-based collaborative filtering with Pearson correlation
- K=50 neighbors, min 5 overlapping movies
- IMDb enrichment for metadata
