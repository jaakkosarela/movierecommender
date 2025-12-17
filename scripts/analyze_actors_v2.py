#!/usr/bin/env python3
"""
Refined analysis to find the best actors/actresses of all time.
Multiple ranking systems to account for different criteria.
"""

import duckdb
import pandas as pd

con = duckdb.connect()

print("Loading IMDb data and calculating actor statistics...")
print("This takes ~1-2 minutes...\n")

# First, build a comprehensive actor stats table
base_query = """
CREATE OR REPLACE TABLE actor_stats AS
WITH actor_movies AS (
    SELECT
        p.nconst,
        p.category,
        p.ordering,  -- 1 = lead role, higher = supporting
        b.tconst,
        b.primaryTitle,
        b.startYear,
        r.averageRating,
        r.numVotes
    FROM read_csv_auto('data/title.principals.tsv', delim='\t', header=true, quote='') p
    JOIN read_csv_auto('data/title.basics.tsv', delim='\t', header=true, quote='') b
        ON p.tconst = b.tconst
    JOIN read_csv_auto('data/title.ratings.tsv', delim='\t', header=true, quote='') r
        ON p.tconst = r.tconst
    WHERE p.category IN ('actor', 'actress')
      AND b.titleType = 'movie'
      AND b.isAdult = '0'
      AND r.numVotes >= 1000
)
SELECT
    am.nconst,
    am.category,
    n.primaryName,
    n.birthYear,
    n.deathYear,
    COUNT(*) as num_movies,
    COUNT(*) FILTER (WHERE am.ordering <= 2) as lead_roles,
    ROUND(AVG(am.averageRating), 2) as avg_rating,
    ROUND(STDDEV(am.averageRating), 2) as rating_stddev,
    SUM(am.numVotes) as total_votes,
    ROUND(SUM(am.averageRating * am.numVotes) / SUM(am.numVotes), 2) as weighted_avg,
    COUNT(*) FILTER (WHERE am.averageRating >= 7.0) as movies_7plus,
    COUNT(*) FILTER (WHERE am.averageRating >= 8.0) as movies_8plus,
    ROUND(100.0 * COUNT(*) FILTER (WHERE am.averageRating >= 7.0) / COUNT(*), 1) as pct_good,
    MIN(am.startYear) as career_start,
    MAX(am.startYear) as career_end
FROM actor_movies am
JOIN read_csv_auto('data/name.basics.tsv', delim='\t', header=true, quote='') n
    ON am.nconst = n.nconst
GROUP BY am.nconst, am.category, n.primaryName, n.birthYear, n.deathYear
"""

con.execute(base_query)

# Calculate global mean for Bayesian rating
global_mean = con.execute("SELECT AVG(avg_rating) FROM actor_stats WHERE num_movies >= 10").fetchone()[0]
print(f"Global average rating across all actors (10+ movies): {global_mean:.2f}\n")

# Add Bayesian weighted rating
# Formula: (n * avg + m * C) / (n + m) where m = minimum movies threshold
con.execute(f"""
    ALTER TABLE actor_stats ADD COLUMN bayesian_rating DOUBLE;
    UPDATE actor_stats SET bayesian_rating =
        ROUND((num_movies * avg_rating + 15 * {global_mean}) / (num_movies + 15), 2);
""")

def print_ranking(title, query, columns):
    print("=" * 90)
    print(title)
    print("=" * 90)
    df = con.execute(query).fetchdf()
    for i, row in df.iterrows():
        status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
        birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
        print(f"{i+1:3}. {row['primaryName']:<35} {birth}{status}")
        details = " | ".join([f"{col}: {row[col]}" for col in columns])
        print(f"     {details}")
    print()
    return df

# RANKING 1: Best by Bayesian Rating (minimum 20 movies)
r1 = print_ranking(
    "🏆 RANKING 1: BEST OVERALL (Bayesian Rating, min 20 movies)",
    """SELECT * FROM actor_stats
       WHERE num_movies >= 20
       ORDER BY bayesian_rating DESC LIMIT 30""",
    ['bayesian_rating', 'avg_rating', 'num_movies', 'movies_8plus']
)

# RANKING 2: Best Lead Actors (based on lead roles)
r2 = print_ranking(
    "🎬 RANKING 2: BEST LEAD ACTORS (min 15 lead roles)",
    """SELECT * FROM actor_stats
       WHERE lead_roles >= 15
       ORDER BY bayesian_rating DESC LIMIT 30""",
    ['bayesian_rating', 'lead_roles', 'num_movies', 'pct_good']
)

# RANKING 3: Most Acclaimed (most 8+ rated movies)
r3 = print_ranking(
    "⭐ RANKING 3: MOST ACCLAIMED (most 8.0+ movies, min 5)",
    """SELECT * FROM actor_stats
       WHERE movies_8plus >= 5
       ORDER BY movies_8plus DESC, avg_rating DESC LIMIT 30""",
    ['movies_8plus', 'num_movies', 'avg_rating', 'pct_good']
)

# RANKING 4: Most Consistent (low std dev, good average)
r4 = print_ranking(
    "📊 RANKING 4: MOST CONSISTENT (low variance, avg 7+, min 20 movies)",
    """SELECT * FROM actor_stats
       WHERE num_movies >= 20 AND avg_rating >= 7.0
       ORDER BY rating_stddev ASC LIMIT 30""",
    ['rating_stddev', 'avg_rating', 'num_movies', 'pct_good']
)

# RANKING 5: Box Office Presence (total votes as proxy)
r5 = print_ranking(
    "💰 RANKING 5: BIGGEST STARS (by total audience, avg 7+)",
    """SELECT * FROM actor_stats
       WHERE avg_rating >= 7.0 AND num_movies >= 20
       ORDER BY total_votes DESC LIMIT 30""",
    ['total_votes', 'avg_rating', 'num_movies']
)

# RANKING 6: Golden Age (actors active before 1970)
r6 = print_ranking(
    "🎞️ RANKING 6: GOLDEN AGE LEGENDS (career ended before 1980)",
    """SELECT * FROM actor_stats
       WHERE career_end <= '1980' AND num_movies >= 15
       ORDER BY bayesian_rating DESC LIMIT 30""",
    ['bayesian_rating', 'num_movies', 'career_start', 'career_end']
)

# COMPOSITE SCORE - combine multiple metrics
print("=" * 90)
print("🏅 COMPOSITE RANKING: THE GREATEST OF ALL TIME")
print("   (Combining: Bayesian rating, # of great movies, consistency, lead roles)")
print("=" * 90)

composite_query = """
WITH ranked AS (
    SELECT *,
        PERCENT_RANK() OVER (ORDER BY bayesian_rating) as pct_bayesian,
        PERCENT_RANK() OVER (ORDER BY movies_8plus) as pct_great,
        PERCENT_RANK() OVER (ORDER BY lead_roles) as pct_leads,
        PERCENT_RANK() OVER (ORDER BY pct_good) as pct_consistent,
        PERCENT_RANK() OVER (ORDER BY total_votes) as pct_popularity
    FROM actor_stats
    WHERE num_movies >= 15
)
SELECT *,
    ROUND(0.30 * pct_bayesian +
          0.25 * pct_great +
          0.20 * pct_leads +
          0.15 * pct_consistent +
          0.10 * pct_popularity, 4) as composite_score
FROM ranked
ORDER BY composite_score DESC
LIMIT 50
"""

composite = con.execute(composite_query).fetchdf()
for i, row in composite.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"{i+1:3}. {row['primaryName']:<35} {birth}{status}")
    print(f"     Score: {row['composite_score']:.3f} | "
          f"Bayesian: {row['bayesian_rating']} | "
          f"8+ movies: {row['movies_8plus']} | "
          f"Lead roles: {row['lead_roles']} | "
          f"Total movies: {row['num_movies']}")

# Save all data
full_data = con.execute("SELECT * FROM actor_stats ORDER BY bayesian_rating DESC").fetchdf()
full_data.to_csv('data/actor_full_stats.csv', index=False)
composite.to_csv('data/actor_composite_ranking.csv', index=False)

print(f"\n✅ Full statistics saved to data/actor_full_stats.csv")
print(f"✅ Composite ranking saved to data/actor_composite_ranking.csv")
