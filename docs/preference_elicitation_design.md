# Binary Preference Elicitation System Design

## Overview

A system for gathering binary preference data ("Do you prefer A or B?") with two user-facing modes:

1. **Calibrate** - System asks about movies where model predictions differ from user ratings
2. **Rate** - User rates a movie they watched (system handles MovieLens lookup internally)

Pairwise comparisons are cognitively easier than absolute ratings and provide richer signal about relative preferences.

## Core Principles

1. **Log everything** - All comparisons saved with timestamps and model version
2. **Only rated movies as anchors** - Reference set = movies user has watched (avoids "haven't seen")
3. **Versioned models** - Can replay comparisons against different model versions
4. **Randomized anchors** - Avoid systematic bias from fixed reference sets
5. **No bias induction** - Don't show user's existing ratings during comparison
6. **Force choice** - No "equal" option; forces discrimination
7. **Start simple** - Text CLI first, enrich with metadata

## Use Cases

### Mode 1: Calibrate

**Goal**: Refine user's latent factors θ by asking about movies where model predictions are surprising.

**When to use**: Model predicts rating that differs significantly from user's actual rating.

**Flow**:
1. Load user's rated movies and current θ estimate
2. Identify "surprising" movies: `|predicted - actual| > threshold`
3. Sample pairs involving surprising movies (max entropy selection)
4. User answers 15-20 comparisons per session
5. Log all comparisons
6. (Later) Update θ using logged comparisons

**Information gain**: The most informative pairs are where P(A>B) ≈ 0.5, meaning the model is maximally uncertain. Pairs involving surprising movies help determine if:
- User's original rating was noisy, or
- Model's θ estimate is wrong for this preference dimension

### Mode 2: Rate

**Goal**: User has watched a movie and wants to rate it. Use ~7 pairwise comparisons to estimate rating.

**Flow**:
```
User enters: "Oppenheimer"
    ↓
Search IMDb (fuzzy match if needed)
    ↓
User confirms: "Oppenheimer (2023) tt15398776"
    ↓
System checks MovieLens internally
    ├─ Found → use model prediction as starting point for binary search
    └─ Not found → pure anchor-based binary search
    ↓
Binary comparison session (~7 questions)
    ↓
Rating saved to user_ratings.csv
```

**Key insight**: The user doesn't need to know whether the movie is in MovieLens. The system adapts internally:

| Movie in MovieLens? | Strategy |
|---------------------|----------|
| Yes | Model predicts rating, binary search starts near prediction |
| No | Pure binary search through anchors spanning rating range |

**Anchor selection**: Select rated movies spanning user's rating range (e.g., movies rated 5, 7, 8, 9, 10). Binary search narrows down position, then interpolate rating from neighboring anchors.

## Data Schema

### Pairwise Comparisons Log

Append-only JSONL file: `data/pairwise_comparisons.jsonl`

```json
{
    "id": "cmp_20241217_143201_a1b2c3",
    "timestamp": "2024-12-17T14:32:01Z",
    "model_version": "irt_v1_20epochs",
    "use_case": "calibration",
    "session_id": "sess_abc123",
    "round": 5,
    "movie_a": {
        "tconst": "tt0088763",
        "title": "Back to the Future",
        "year": 1985,
        "genres": ["Adventure", "Comedy", "Sci-Fi"]
    },
    "movie_b": {
        "tconst": "tt0111161",
        "title": "The Shawshank Redemption",
        "year": 1994,
        "genres": ["Drama"]
    },
    "choice": "a",
    "model_prediction": {
        "prob_a_wins": 0.52,
        "entropy": 0.693,
        "user_theta_version": "v1"
    },
    "sampling_strategy": "max_entropy"
}
```

For rating sessions, additional fields:
```json
{
    "use_case": "rate",
    "target_movie": {
        "tconst": "tt15398776",
        "title": "Oppenheimer",
        "year": 2023,
        "in_movielens": false,
        "model_prediction": null
    },
    "estimated_rating": 8.5,
    "confidence_interval": [8.0, 9.0]
}
```

### Session Log

Track session-level metadata: `data/sessions.jsonl`

```json
{
    "session_id": "sess_abc123",
    "started_at": "2024-12-17T14:30:00Z",
    "ended_at": "2024-12-17T14:45:00Z",
    "use_case": "calibration",
    "model_version": "irt_v1_20epochs",
    "n_comparisons": 18,
    "sampling_strategy": "max_entropy",
    "notes": "First calibration session"
}
```

## Sampling Strategies

### For Calibration (max information gain)

**Max Entropy**: Select pair (A, B) where model prediction P(A>B) is closest to 0.5.

```
H[P(A>B)] = -p log(p) - (1-p) log(1-p)
```

Maximized when p = 0.5 (entropy = 0.693 nats).

**Uncertainty-Weighted**: Also consider uncertainty in θ. Prefer pairs where:
1. P(A>B) ≈ 0.5
2. β_A - β_B has large projection onto uncertain dimensions of θ

**Diverse Latent**: Ensure pairs span different directions in latent space. Avoid repeatedly comparing movies that are similar in latent representation.

**Random Baseline**: Uniform random pairs. Use occasionally to avoid systematic gaps.

### For New Movie Rating (binary search)

**Adaptive Anchoring**:
1. Start with anchor near median of user's ratings
2. If user prefers new movie: next anchor from higher-rated movies
3. If user prefers anchor: next anchor from lower-rated movies
4. Repeat until confidence interval < 1 star

**Randomized Anchors**: Don't always pick the same anchor at each level. Sample from movies within the target rating range to avoid overfitting to specific comparisons.

## CLI Interface

### Calibration Mode

```
$ python scripts/elicit_preferences.py calibrate --n-rounds 20

╔═══════════════════════════════════════════════════════════════╗
║                   PREFERENCE CALIBRATION                       ║
║  Model: irt_v1_20epochs | Strategy: max_entropy | Session: 1   ║
╚═══════════════════════════════════════════════════════════════╝

Round 1 of 20
─────────────────────────────────────────────────────────────────
  [A] The Big Lebowski (1998)
      Comedy, Crime
      A laid-back slacker gets embroiled in a kidnapping scheme
      after being mistaken for a millionaire with the same name.

  [B] Fargo (1996)
      Crime, Drama, Thriller
      A car salesman hires two criminals to kidnap his wife,
      but the plan goes terribly wrong.
─────────────────────────────────────────────────────────────────

Which do you prefer? [a/b]: a

✓ Logged (model was 52% confident in A)

Round 2 of 20
...

═══════════════════════════════════════════════════════════════
Session complete! 20 comparisons logged.
Run `python scripts/update_factors.py` to update your θ.
═══════════════════════════════════════════════════════════════
```

### Rate Mode

```
$ python scripts/elicit_preferences.py rate "Oppenheimer"

Searching IMDb for "Oppenheimer"...

  [1] Oppenheimer (2023) - Biography, Drama, History
  [2] Oppenheimer (1980) - Biography, Drama, History
  [3] The Trials of J. Robert Oppenheimer (2008) - Documentary

Select movie [1-3]: 1

Found: Oppenheimer (2023) [tt15398776]
       Biography, Drama, History

╔═══════════════════════════════════════════════════════════════╗
║                    RATE: Oppenheimer (2023)                    ║
║  ~7 questions to estimate your rating                          ║
╚═══════════════════════════════════════════════════════════════╝

Round 1 of ~7
─────────────────────────────────────────────────────────────────
  Do you prefer...

  [A] Oppenheimer (2023)
      Biography, Drama, History

  [B] The Dark Knight (2008)
      Action, Crime, Drama
─────────────────────────────────────────────────────────────────

Which do you prefer? [a/b]: a

Round 2 of ~7
...

═══════════════════════════════════════════════════════════════
Rating estimate: 9.0 ± 0.5

Would you like to save this rating? [y/n]: y
✓ Saved to user_ratings.csv
✓ Comparison session logged (7 comparisons)
═══════════════════════════════════════════════════════════════
```

Note: The system internally checks if the movie is in MovieLens. If yes, it uses the model prediction to start the binary search near the expected rating. If no, it does a full binary search through anchors. The user doesn't see this distinction.

## Movie Metadata

### Data Sources

1. **MovieLens movies.csv**: title, genres
2. **IMDb title.basics.tsv**: title, year, runtime, genres
3. **IMDb title.ratings.tsv**: average rating, vote count
4. **OMDb API** (optional): plot summaries, posters

### Metadata Display

During comparison, show:
- Title (year)
- Genres
- Brief plot description (if available)
- Runtime (optional)

Do NOT show:
- User's existing rating (would bias choice)
- IMDb rating (would bias choice)
- Model's predicted preference

## Model Versioning

```
models/
├── irt_v1_3epochs.pt
├── irt_v1_20epochs.pt
├── irt_v2_30factors.pt
└── manifest.json
```

**manifest.json**:
```json
{
    "models": [
        {
            "filename": "irt_v1_20epochs.pt",
            "created_at": "2024-12-17T12:00:00Z",
            "config": {
                "n_factors": 20,
                "n_epochs": 20,
                "prior_scale_start": 1.0,
                "prior_scale_end": 0.1
            },
            "metrics": {
                "final_elbo": -28500000,
                "n_ratings": 25000095
            },
            "notes": "First production model"
        }
    ]
}
```

## Architecture

```
src/
├── irt_model.py              # Existing IRT model
├── preference_elicitation.py # Core elicitation logic
│   ├── SamplingStrategy      # Protocol for pair selection
│   ├── MaxEntropySampler     # Default strategy
│   ├── AdaptiveRatingSampler # For new movie rating
│   ├── preference_prob()     # P(A>B) calculation
│   └── update_factors()      # Update θ from comparisons
├── comparison_logger.py      # JSONL logging utilities
└── movie_metadata.py         # Fetch/cache movie info

scripts/
├── elicit_preferences.py     # Main CLI
├── update_factors.py         # Batch update θ from logged comparisons
└── analyze_comparisons.py    # Explore logged data

data/
├── pairwise_comparisons.jsonl  # All comparisons (append-only)
├── sessions.jsonl              # Session metadata
└── movie_cache.json            # Cached metadata (plot, etc.)
```

## Update Strategies (Future)

Once we have logged comparisons, multiple update strategies:

1. **MAP Update**: Point estimate of θ that maximizes likelihood of observed comparisons
2. **VI Update**: Full posterior over θ incorporating comparison likelihood
3. **Online Update**: Update θ after each comparison (for real-time feedback)
4. **Batch Update**: Accumulate comparisons, update periodically

The logged data supports all these - we can experiment with different approaches.

## Bradley-Terry Likelihood

For comparisons, use Bradley-Terry model:

```
P(A > B | θ) = σ(s_A - s_B)

where:
    s_A = θ · β_A + b_A + b_user + μ_global  (predicted rating for A)
    s_B = θ · β_B + b_B + b_user + μ_global  (predicted rating for B)
    σ(x) = 1 / (1 + exp(-x))                  (sigmoid)
```

Log-likelihood for a set of comparisons:
```
LL = Σ [y_i log P(A_i > B_i) + (1-y_i) log P(B_i > A_i)]

where y_i = 1 if user chose A, 0 if chose B
```

## Open Questions (Deferred)

1. **Transitivity violations**: What if user says A > B, B > C, but C > A?
   - Log it, don't enforce consistency
   - Can analyze later for noise estimation

2. **Temporal drift**: Preferences change over time
   - Timestamp everything
   - Could weight recent comparisons higher

3. **Confidence calibration**: Is the model's P(A>B) well-calibrated?
   - After gathering data, can plot calibration curves

4. **Active learning**: Beyond max entropy, could use expected model change
   - More complex, defer to v2

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| P1 | Data schema and logging utilities | **Done** |
| P2 | IMDb search with fuzzy matching | **Done** |
| P3 | MovieLens lookup (movie → model predictions) | Planned |
| P4 | Sampling strategies (max_entropy, adaptive binary search) | **Done** |
| P5 | CLI for calibrate mode | Planned |
| P6 | CLI for rate mode (unified flow) | Planned |
| P7 | Factor update from comparisons | Planned |
| P8 | Analysis tools for logged data | Planned |

## Implementation Details (P1, P2, P4)

### Package Structure

```
src/elicitation/
├── __init__.py          # Package exports
├── schemas.py           # Movie, Comparison, Session, RatingEstimate
├── logger.py            # ComparisonLogger (JSONL append-only)
├── sampler.py           # MaxEntropySampler, AdaptiveBinarySearchSampler
└── movie_search.py      # MovieSearcher (IMDb fuzzy search)
```

### Key Classes

**Schemas:**
- `Movie(tconst, title, year, genres, movielens_id, model_prediction)`
- `Comparison(movie_a, movie_b, choice, session_id, model_prediction, ...)`
- `Session(id, use_case, started_at, ended_at, comparisons, rating_estimate)`
- `RatingEstimate(rating, confidence_low, confidence_high, n_comparisons)`

**Logger:**
- `ComparisonLogger.log_comparison(comparison)` → appends to `data/pairwise_comparisons.jsonl`
- `ComparisonLogger.log_session(session)` → appends to `data/sessions.jsonl`

**Samplers:**
- `MaxEntropySampler(rated_movies, predicted_ratings)` → for calibration
- `AdaptiveBinarySearchSampler(target_movie, anchor_movies, anchor_ratings)` → for rating

**Search:**
- `MovieSearcher.search(query, limit=10)` → fuzzy search IMDb
- `MovieSearcher.get_by_tconst(tconst)` → lookup by ID
- Returns `SearchResult(movie, score)` with MovieLens ID if available

### Remaining Work

- **P3**: Model interface to get predictions for user's rated movies and new movies
- **P5**: CLI loop for calibrate mode (load model, sample pairs, prompt user)
- **P6**: CLI loop for rate mode (search movie, binary search, save rating)

---

*Implementation in progress.*
