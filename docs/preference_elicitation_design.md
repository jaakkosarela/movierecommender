# Binary Preference Elicitation System Design

## Overview

A system for gathering binary preference data ("Do you prefer A or B?") to:
1. **Calibrate** existing ratings by refining user latent factors
2. **Rate new movies** by placing them in the user's preference ordering

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

### Use Case A: Calibration

**Goal**: Refine user's latent factors θ using pairwise comparisons between already-rated movies.

**Flow**:
1. Load user's rated movies and current θ estimate
2. Sample informative pairs (movies where model is uncertain about preference)
3. User answers 15-20 comparisons per session
4. Log all comparisons
5. (Later) Update θ using logged comparisons

**Information gain**: The most informative pairs are where P(A>B) ≈ 0.5, meaning the model is maximally uncertain.

### Use Case B: Rating a New Movie

**Goal**: User has seen a new movie and wants to rate it. Use ~7 pairwise comparisons to estimate rating.

**Flow**:
1. User specifies movie (by title search or IMDb ID)
2. System checks if movie exists (in MovieLens or IMDb)
3. Sample anchor movies from user's rated set
4. Binary search through preference space
5. Convert final position to rating estimate

**Key insight**: We don't need IRT item factors for the new movie. We can place it in the user's preference ordering using comparisons alone, then interpolate a rating.

### Handling Movies Not in MovieLens

If a movie is in IMDb but not MovieLens:
- We lack trained item factors β_m
- But we CAN still rate it via pairwise comparisons

**Approach**:
1. Select anchor movies spanning user's rating range (e.g., movies rated 4, 6, 8, 10)
2. Binary search: "Do you prefer [new movie] or [anchor]?"
3. Narrow down position in preference ordering
4. Interpolate rating from neighboring anchors

This works because we're using the user's own rated movies as a calibrated scale.

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

For new movie rating sessions, additional fields:
```json
{
    "use_case": "new_movie",
    "target_movie": {
        "tconst": "tt15398776",
        "title": "Oppenheimer",
        "year": 2023,
        "in_movielens": false
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

### New Movie Rating Mode

```
$ python scripts/elicit_preferences.py rate "Oppenheimer"

Searching for "Oppenheimer"...
Found: Oppenheimer (2023) [tt15398776]
       Biography, Drama, History
       In MovieLens: No (will use comparison-based rating)

╔═══════════════════════════════════════════════════════════════╗
║                    RATE: Oppenheimer (2023)                    ║
║  ~7 questions to estimate your rating                          ║
╚═══════════════════════════════════════════════════════════════╝

Round 1 of ~7
─────────────────────────────────────────────────────────────────
  Do you prefer...

  [A] Oppenheimer (2023)
      Biography, Drama, History
      The story of the atomic bomb's development and its aftermath.

  [B] The Dark Knight (2008)
      Action, Crime, Drama
      Batman faces the Joker, a criminal mastermind who seeks
      to plunge Gotham into anarchy.
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
| P1 | Data schema and logging utilities | Planned |
| P2 | Movie metadata fetching/caching | Planned |
| P3 | Sampling strategies (max_entropy, adaptive) | Planned |
| P4 | CLI for calibration mode | Planned |
| P5 | CLI for new movie rating mode | Planned |
| P6 | Factor update from comparisons | Planned |
| P7 | Analysis tools for logged data | Planned |

---

*Design complete. Ready for implementation on user approval.*
