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
├── similarity.py       # User similarity calculations (Pearson)
├── recommender.py      # Core recommendation engine
└── cli.py              # Command-line interface

scripts/
├── analyze_*.py        # Ad-hoc analysis scripts (actors, directors, etc.)
├── build_user_index.py # Precompute user similarity matrix
└── recommend.py        # Generate recommendations for user

data/
├── ml-25m/             # MovieLens 25M dataset
│   ├── ratings.csv     # 25M ratings (userId, movieId, rating)
│   ├── links.csv       # MovieLens → IMDb mapping
│   └── movies.csv      # Titles and genres
├── *.tsv               # IMDb datasets (basics, ratings, principals, names)
├── *.tsv.gz            # IMDb compressed originals
└── user_ratings.csv    # User's personal IMDb ratings

results/
└── recommendations/    # Generated recommendation outputs

docs/
├── recommendation_system_design.md  # Algorithm design document
└── compacting/
    └── compacting_summary.md        # Session continuity notes
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
| **User ratings** | ~35 movies | data/user_ratings.csv |

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
- (future) scikit-learn - ML utilities

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

1. **User-based collaborative filtering** - Find similar MovieLens users, predict ratings
2. **Pearson correlation** - Handles user rating bias (user rates +0.9 vs IMDb average)
3. **K=50 neighborhood** - Balance between signal and noise
4. **IMDb enrichment** - Add director, cast, metadata to recommendations
5. **Quality threshold** - Filter to IMDb ≥7.0 (user prefers quality films)

## Implementation Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **A** | Data exploration and actor/director analysis | Done |
| **B** | Algorithm design document | Done |
| **C** | Implement data loader (MovieLens + IMDb join) | Next |
| **D** | Implement similarity computation | Planned |
| **E** | Implement recommendation engine | Planned |
| **F** | Build CLI and test with user ratings | Planned |
