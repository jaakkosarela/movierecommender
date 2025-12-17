# Future Ideas

Ideas for improving the recommendation system, collected during development.

---

## Rating Data Quality

### Pairwise Comparison System
Build a system that improves rating coherence by:
- Polling two movies randomly and asking "which did you like better?"
- Collecting pairwise preference data
- Using importance sampling to focus on comparisons where the system is uncertain
- Could help calibrate ratings and surface inconsistencies

### Adaptive Pairwise Rating for New Movies
When rating a new movie X, instead of gut-feel 1-10, use adaptive pairwise comparisons.

**The idea:**
- You watched new movie X, want to place it in your preference space
- System asks: "Did you like X more or less than Y?"
- Choice of anchor Y matters - some comparisons are more informative

**What makes a good comparison anchor Y:**
1. **Binary search (1D):** If prior for X is "probably 7-9", compare to Y rated 8. Answer narrows range.
2. **Multidimensional:** Pick Y that discriminates along uncertain dimensions.
3. **IRT formulation:** Choose Y that minimizes expected posterior variance on X's position.

**Example adaptive sequence:**
```
System: "Did you like [New Movie] more or less than Fargo?"
You: "More"
System: "More or less than Pulp Fiction?"
You: "Less"
System: "More or less than The Departed?"
You: "About the same"
→ System places X near The Departed in your preference space
```

**Benefits:**
- 3-4 comparisons can pin down a rating more reliably than one gut-feel number
- Comparisons are easier cognitively than absolute ratings
- Naturally calibrates against your existing ratings
- With IRT model, can compute optimal anchor selection mathematically

**Implementation needs:**
- Bradley-Terry or Thurstone model for pairwise preferences
- Posterior inference on item position given comparison outcomes
- Greedy or optimal selection of next comparison anchor

### Recalibrate Existing Ratings via Pairwise Comparisons
Use the same pairwise system to refine and check consistency of existing ratings.

**Use cases:**
1. **Surface inconsistencies:** "You rated A = 8 and B = 8, but you say A > B. Adjust?"
2. **Refine ties:** When two movies have same rating, comparison establishes ordering
3. **Detect drift:** Compare old ratings to recent ones, see if preferences shifted

**Importance sampling for comparisons:**
- Focus on pairs where model predicts high uncertainty
- Prioritize comparisons that would most change the latent space fit
- Skip pairs where ratings already imply clear ordering (9 vs 6)

**Feedback loop:**
- After N comparisons, suggest rating adjustments
- "Based on your comparisons, A should be 9 not 8. Update?"
- User confirms or rejects, system learns their calibration

### Expand Rating Coverage
- Rate mediocre movies (6-7 range) not just favorites
- Rate movies you disliked
- This would help find users whose "meh" and "bad" ratings also match yours

### Suggest Which Movies to Rate
Help user prioritize which watched movies to rate for maximum system improvement.

**Use case 1: "I just watched movie X - should I rate it?"**
- Is it in MovieLens? (if not, no value for collaborative filtering)
- Did similar users rate it with high variance? (discriminating item = valuable)
- Does it fill a gap in genre coverage?

**Use case 2: "Which of my unrated movies should I prioritize?"**
- Adaptive item selection (from IRT literature)
- Prioritize movies that are:
  - In MovieLens (connects to the population)
  - Rated by many similar users (strengthens similarity estimates)
  - High variance among similar users (discriminates between them)
  - In underrepresented genres for the user

**Implementation sketch:**
```python
def suggest_movies_to_rate(user_watched_unrated, similar_users, matrix):
    """Rank unrated movies by information value."""
    # For each candidate movie:
    # - Count how many similar users rated it
    # - Compute variance of their ratings
    # - Score = count * variance (or similar)
    # Return sorted by score descending
```

With IRT/MF approach, this becomes principled: pick items that maximally reduce posterior uncertainty on u_i.

---

## Prediction Accuracy

### Bias Adjustment
LOO analysis shows predictions are ~1 point below actual ratings. Could:
- Add constant offset (+1) to predictions
- Learn user-specific bias term
- Current: user rates ~0.9 above their neighbors on average

### Alternative Similarity Metrics
- Cosine similarity instead of Pearson
- Weighted Pearson (weight by overlap count)
- Item-based collaborative filtering as complement to user-based

### Low Correlation Issue
Current correlation between predicted and actual is only 0.16. Possible improvements:
- More neighbors (K > 50)
- Higher min_overlap threshold
- Hybrid approach combining content-based features

---

## Features

### Feedback Loop
- After watching a recommendation, user rates it
- System tracks prediction accuracy over time
- Could re-weight similar users based on prediction success

### Genre Filtering
- Allow filtering recommendations by genre
- "Give me 5 thriller recommendations"

### Exclude Recently Watched
- Track what user has already seen from recommendations
- Avoid re-recommending

---

## Data Sources

### Newer MovieLens Data
- MovieLens 25M is from 2019
- Check if newer dataset available for recent movies

### Combine with Content-Based
- Use IMDb metadata (director, actors, genre) as features
- Hybrid recommender: collaborative + content-based

---

## Evaluation

### Holdout Validation
- When user adds more ratings, hold out some for proper validation
- Track prediction accuracy over time

### A/B Testing
- Compare weighting schemes (linear vs softmax) on actual user satisfaction

---

## Advanced: IRT / Matrix Factorization with Ordinal Likelihood

A more principled approach inspired by Item Response Theory (IRT) and probabilistic matrix factorization.

### Core Model
Ordinal (graded response) likelihood:
```
P(R_ij ≤ k) = logistic((τ_k - μ_i - u_i^T v_j) / σ_i)
```

Where:
- `τ_k` = rating thresholds (shared across users)
- `u_i` = user's position in latent preference space (k dimensions)
- `v_j` = item's direction/discrimination in latent space
- `μ_i` = user-specific baseline shift (low vs high rater)
- `σ_i` = user-specific scale usage (extreme vs middle-hugging)

### Why This Helps
1. **Separates preference from expression** - your +1 bias becomes μ_i, not noise
2. **Ordinal likelihood** - respects that 8 < 9 < 10 without assuming equal intervals
3. **Multidimensional preferences** - captures that you might love thrillers AND comedies
4. **Uncertainty quantification** - posterior on u_i tells you how confident we are

### Implementation Path
1. **Pre-calibrate items on MovieLens population**
   - Matrix factorization to get v_j embeddings (k ≈ 5-20 dimensions)
   - Can use SVD, probabilistic MF, or Bayesian approach
   - 162K users provides strong identification

2. **Fix item parameters, estimate single user**
   - With v_j fixed, estimate u_i, μ_i, σ_i from 70 ratings
   - This is tractable - just posterior inference on ~25 parameters
   - Stan/PyMC could handle this

3. **Adaptive item selection (optional)**
   - Like computerized adaptive testing
   - Ask about movies that maximally reduce uncertainty in u_i
   - Focus on high-discrimination items spread across latent space

### Requirements
- ~10-15 ratings minimum to separate μ_i from u_i
- More ratings needed for higher dimensionality (5-10× per added dimension)
- We have 70 ratings, so k ≤ 5-7 dimensions seems reasonable

### References
- IRT / Graded Response Model literature
- Probabilistic Matrix Factorization (Salakhutdinov & Mnih)
- Bayesian Personalized Ranking
