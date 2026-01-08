# Movie Recommender

A personalized movie recommendation system using collaborative filtering with Item Response Theory (IRT). Learns your taste from ratings and pairwise comparisons to recommend movies you'll love.

## Features

- **IRT-based recommendations**: Latent factor model trained on MovieLens 25M (25 million ratings from 162K users)
- **Thompson sampling**: Balances exploration vs exploitation with uncertainty-aware recommendations
- **Preference elicitation**: Refine your profile through pairwise comparisons ("Do you prefer A or B?")
- **Genre filtering**: Filter recommendations by genre
- **TV series support**: Rate and calibrate TV series via comparisons against movie anchors

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Get recommendations
PYTHONPATH=. python scripts/recommend_irt.py

# Filter by genre
PYTHONPATH=. python scripts/recommend_irt.py --genre Thriller --genre Crime

# List available genres
PYTHONPATH=. python scripts/recommend_irt.py --list-genres
```

## Data Setup

You'll need:

1. **MovieLens 25M** dataset in `data/ml-25m/`
   - Download from https://grouplens.org/datasets/movielens/25m/
   - Required files: `ratings.csv`, `links.csv`, `movies.csv`

2. **IMDb datasets** in `data/`
   - Download from https://datasets.imdbws.com/
   - Required: `title.basics.tsv`, `title.ratings.tsv`
   - Optional: `title.principals.tsv`, `name.basics.tsv`

3. **Your ratings** in `data/user_ratings.csv`
   - Export from IMDb (Your Ratings → Export)
   - Or create CSV with columns: `Const` (tconst), `Your Rating`

## Commands

### Get Recommendations

```bash
# Basic recommendations (uses your trained model + user checkpoint)
PYTHONPATH=. python scripts/recommend_irt.py

# More options
PYTHONPATH=. python scripts/recommend_irt.py \
  --top-n 30 \
  --genre Thriller \
  --seed 42 \
  --show-details
```

### Calibrate Your Profile

Improve model accuracy by answering "which do you prefer?" questions:

```bash
PYTHONPATH=. python scripts/elicit_preferences.py calibrate --n-rounds 20
```

- Picks pairs where model is most uncertain (max entropy)
- Supports movies and TV series
- Press `s` to skip pairs you can't compare

### Rate a New Movie/Series

Rate something you just watched via ~7 comparisons:

```bash
PYTHONPATH=. python scripts/elicit_preferences.py rate "No Way Out" --year 1987
```

- Searches IMDb, you select the match
- Compares against your rated movies
- Estimates rating with confidence interval

### Update Your Model

After calibrating or rating, update your user factors:

```bash
PYTHONPATH=. python scripts/update_factors.py
```

- Incorporates new comparisons and ratings
- Uses information-weighted likelihood (uncertain observations count more)
- Saves to `models/user_theta.pt`

## How It Works

### Model Architecture

1. **Item Response Theory (IRT)**: Each user and movie has a latent factor vector (K=20 dimensions). Predicted rating = dot product + biases.

2. **Variational Inference**: Trained on MovieLens 25M with stochastic mini-batches. Non-symmetric priors break rotational invariance.

3. **User Fitting**: Your factors (θ) are optimized while movie factors (β) stay fixed. Updated incrementally from ratings + comparisons.

### Recommendation Strategy

**Thompson Sampling with shrinkage**:
- Sample from posterior: `score = mean + std * N(0,1)`
- Shrink uncertainty for low-vote movies (avoids obscure film domination)
- Soft penalty for predictions that diverge too far from IMDb consensus

### Preference Elicitation

**Calibrate mode**: Max-entropy sampling picks pairs where P(A>B) ≈ 0.5 (model is uncertain).

**Rate mode**: Adaptive binary search narrows the rating interval with each comparison. Converges when interval < 0.5.

## Project Structure

```
├── scripts/
│   ├── recommend_irt.py      # Generate recommendations
│   ├── elicit_preferences.py # Calibrate/rate CLI
│   ├── update_factors.py     # Update user θ from comparisons
│   ├── train_irt.py          # Train IRT model (one-time)
│   └── check_calibration.py  # Model diagnostics
│
├── src/
│   ├── irt_model.py          # IRT model + variational inference
│   ├── data_loader.py        # Load MovieLens + IMDb
│   ├── elicitation/          # Preference elicitation package
│   │   ├── schemas.py        # Data classes (Movie, Comparison, etc.)
│   │   ├── sampler.py        # MaxEntropy, AdaptiveBinarySearch
│   │   ├── logger.py         # JSONL logging
│   │   ├── movie_search.py   # IMDb fuzzy search
│   │   └── model_interface.py# Model predictions
│   └── recommendation/       # Recommendation generation
│       └── thompson.py       # Thompson sampling + shrinkage
│
├── models/
│   ├── irt_v1.pt             # Trained model (β fixed)
│   └── user_theta.pt         # Your latent factors (θ)
│
├── data/
│   ├── ml-25m/               # MovieLens dataset
│   ├── *.tsv                 # IMDb datasets
│   ├── user_ratings.csv      # Your IMDb export
│   ├── pairwise_comparisons.jsonl  # Logged comparisons
│   ├── rating_events.jsonl   # Logged ratings
│   └── sessions.jsonl        # Session metadata
│
└── docs/
    ├── recommendation_system_design.md
    ├── preference_elicitation_design.md
    └── compacting/           # Session continuity notes
```

## Workflow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Your IMDb       │────▶│ Train/fit θ      │────▶│ Recommendations │
│ ratings         │     │ (update_factors) │     │ (recommend_irt) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               ▲
                               │
┌─────────────────┐     ┌──────┴───────────┐
│ Calibrate       │────▶│ pairwise_        │
│ (elicit_prefs)  │     │ comparisons.jsonl│
└─────────────────┘     └──────────────────┘
                               ▲
┌─────────────────┐            │
│ Rate new movie  │────────────┘
│ (elicit_prefs)  │
└─────────────────┘
```

## Requirements

- Python 3.12+
- ~2GB RAM for MovieLens 25M
- ~500MB disk for IMDb datasets

## License

Personal project. MovieLens and IMDb data have their own terms of use.
