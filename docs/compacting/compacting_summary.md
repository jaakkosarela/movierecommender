# Session Continuity Summary

Reverse chronological order (newest first).

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
