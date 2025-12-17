# CLAUDE.md - Movie Recommendation System

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A **collaborative filtering movie recommendation system** that uses MovieLens 25M user ratings to find movies the user would enjoy, enriched with IMDb metadata.

**Core approach:**
- MovieLens provides user-user similarity (collaborative filtering)
- IMDb provides rich metadata (directors, actors, genres, ratings)
- User's personal ratings anchor the recommendations

## Project Structure

```
src/
├── __init__.py
├── data_loader.py      # Load and join MovieLens + IMDb data
├── similarity.py       # User similarity calculations (Pearson) [V1]
├── recommender.py      # Core recommendation engine [V1]
└── irt_model.py        # IRT latent factor model with VI [V2]

scripts/
├── analyze_*.py        # Ad-hoc analysis scripts (actors, directors, etc.)
├── recommend.py        # Generate recommendations (V1 Pearson-based)
└── train_irt.py        # Train IRT model and generate recommendations (V2)

data/
├── ml-25m/             # MovieLens 25M dataset
│   ├── ratings.csv     # 25M ratings (userId, movieId, rating)
│   ├── links.csv       # MovieLens → IMDb mapping
│   └── movies.csv      # Titles and genres
├── *.tsv               # IMDb datasets (basics, ratings, principals, names)
├── user_ratings.csv    # User's personal IMDb ratings
├── pairwise_comparisons.jsonl  # Binary preference data (future)
└── sessions.jsonl      # Elicitation session logs (future)

models/
└── *.pt                # Trained IRT model checkpoints

docs/
├── recommendation_system_design.md   # Algorithm design (V1 + V2)
├── preference_elicitation_design.md  # Binary preference system design
├── future_ideas.md                   # Ideas for improvements
└── compacting/
    └── compacting_summary.md         # Session continuity notes
```

## Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run analysis scripts
python scripts/analyze_actors.py

# Run tests
pytest tests/
pytest tests/test_similarity.py -v

# Format and lint
black src/ scripts/
ruff check src/ scripts/
```

## Data Sources

| Source | Size | Key Files |
|--------|------|-----------|
| **MovieLens 25M** | 25M ratings, 162K users | ratings.csv, links.csv |
| **IMDb** | 12M titles, 15M people | title.basics.tsv, title.principals.tsv |
| **User ratings** | 82 movies (70 in MovieLens) | data/user_ratings.csv |

## Environment

Python 3.12 (`.venv/pyvenv.cfg`)

```bash
# Create/activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- pandas, numpy, scipy - Data manipulation
- duckdb - Fast SQL queries on large files
- torch - IRT model and variational inference
- tqdm - Progress bars

## Modes of Operation

Data science work requires flexibility - data frequently challenges assumptions.

### Design Mode

Use when: exploring problem space, validating hypotheses, data is challenging assumptions.

- **Brainstorm** multiple approaches before committing
- Prioritize learning over code quality
- EDA scripts are disposable (prefix with `eda_*`)
- Use inline Python for quick exploration
- Summarize findings and implications for design
- Output: Updated understanding, design decisions — not production code

**Transition out**: "Design is stable. Ready to implement?"

### Implementation Mode

Use when: design is validated, building production components.

- Follow phased implementation with review points
- Production-quality code with tests
- Stop after each phase for review
- If implementation reveals design issues → flag and return to Design Mode

## Coding Style

- Format with `black` (line length 88), lint with `ruff`
- **Naming**: snake_case files/functions, PascalCase classes, UPPER_CASE constants
- Type hints and docstrings (`"""Summary."""`) for public functions
- Keep imports sorted via `ruff --select I`

## Testing Guidelines

- Mirror package paths in `tests/` (e.g., `tests/test_similarity.py`)
- Name tests after behavior (`test_pearson_handles_rating_bias`)
- Mark slow tests with `@pytest.mark.slow`

## Start of Session

At the start of a new conversation:
1. Check `docs/compacting/compacting_summary.md` for context from previous sessions
2. Check `git status` to see uncommitted work
3. Ask user what they'd like to focus on

## Session Continuity (Compacting)

When context is running low or before user compacts:
1. Write/update entry in `docs/compacting/compacting_summary.md`
2. Use reverse chronological order (newest first)
3. Include:
   - Current status
   - Completed work
   - Design decisions made
   - Open issues
   - Next steps
   - Key files modified
4. Reference design docs in `docs/` for deeper context

After compacting, check `docs/compacting/compacting_summary.md` for where we left off.

## Key Design Decisions

### V1: Pearson-based Collaborative Filtering
1. **User-based collaborative filtering** - Find similar MovieLens users, predict ratings
2. **Pearson correlation** - Handles user rating bias (user rates +0.9 vs IMDb average)
3. **K=50 neighborhood** - Balance between signal and noise
4. **IMDb enrichment** - Add director, cast, metadata to recommendations

### V2: IRT Latent Factor Model
1. **Item Response Theory** - Latent factors for users (θ) and items (β)
2. **Variational inference** - Scales to 25M ratings with stochastic mini-batches
3. **Non-symmetric priors** - Decreasing variance per dimension breaks rotational invariance
4. **SVD initialization** - Warm start for faster convergence
5. **Uncertainty quantification** - Predictions include confidence intervals

### Preference Elicitation (Designed, Not Implemented)
1. **Binary comparisons** - "Do you prefer A or B?" easier than absolute ratings
2. **Max entropy sampling** - Select pairs where model is most uncertain
3. **Versioned logging** - All comparisons saved for experimentation
4. **IMDb-only movies supported** - Via comparison-based rating against anchors

## Implementation Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **V1** | Pearson-based collaborative filtering | Done |
| **V2** | IRT model with variational inference | Done |
| **V2.1** | Train production model (20 epochs) | Next |
| **V3** | Preference elicitation system | Designed |
| **V3.1** | Data schema and logging utilities | Planned |
| **V3.2** | Sampling strategies (max_entropy, adaptive) | Planned |
| **V3.3** | CLI for calibration mode | Planned |
| **V3.4** | CLI for new movie rating mode | Planned |
| **V3.5** | Factor update from comparisons | Planned |
