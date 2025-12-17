#!/usr/bin/env python3
"""Analyze IMDb data to find the best actors/actresses of all time."""

import duckdb

# Connect to DuckDB (in-memory)
con = duckdb.connect()

print("Loading and analyzing IMDb data...")
print("This may take a minute due to data size (~6GB)...\n")

# Build the analysis query
# - Filter to movies only (not TV, shorts, etc.)
# - Filter to actors/actresses only
# - Join with ratings
# - Require minimum vote threshold for statistical significance
query = """
WITH actor_movies AS (
    SELECT
        p.nconst,
        p.category,
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
      AND r.numVotes >= 1000  -- Only consider movies with significant votes
),
actor_stats AS (
    SELECT
        am.nconst,
        am.category,
        n.primaryName,
        n.birthYear,
        n.deathYear,
        COUNT(*) as num_movies,
        ROUND(AVG(am.averageRating), 2) as avg_rating,
        ROUND(STDDEV(am.averageRating), 2) as rating_stddev,
        SUM(am.numVotes) as total_votes,
        ROUND(SUM(am.averageRating * am.numVotes) / SUM(am.numVotes), 2) as weighted_avg_rating,
        COUNT(*) FILTER (WHERE am.averageRating >= 7.0) as good_movies,
        COUNT(*) FILTER (WHERE am.averageRating >= 8.0) as great_movies,
        MIN(am.startYear) as first_movie_year,
        MAX(am.startYear) as last_movie_year
    FROM actor_movies am
    JOIN read_csv_auto('data/name.basics.tsv', delim='\t', header=true, quote='') n
        ON am.nconst = n.nconst
    GROUP BY am.nconst, am.category, n.primaryName, n.birthYear, n.deathYear
)
SELECT * FROM actor_stats
WHERE num_movies >= 10  -- Minimum 10 movies for ranking
ORDER BY weighted_avg_rating DESC, num_movies DESC
LIMIT 100
"""

print("Running query (joining ~100M+ rows)...")
result = con.execute(query).fetchdf()

print("\n" + "="*80)
print("TOP 100 ACTORS/ACTRESSES BY WEIGHTED AVERAGE RATING")
print("(Minimum 10 movies with 1000+ votes each)")
print("="*80 + "\n")

# Display results
for i, row in result.head(50).iterrows():
    status = "" if row['deathYear'] == '\\N' else f" (†{row['deathYear']})"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Weighted Avg: {row['weighted_avg_rating']:.2f} | "
          f"Movies: {row['num_movies']:3} | "
          f"Great (8+): {row['great_movies']:2} | "
          f"Total Votes: {row['total_votes']:,}")
    print()

# Save full results to CSV
result.to_csv('data/actor_rankings.csv', index=False)
print(f"\nFull results saved to data/actor_rankings.csv")

# Also show some interesting stats
print("\n" + "="*80)
print("ADDITIONAL ANALYSIS")
print("="*80)

# Best by number of great movies
print("\n📊 Most movies rated 8.0+:")
by_great = result.nlargest(10, 'great_movies')[['primaryName', 'great_movies', 'num_movies', 'weighted_avg_rating']]
for i, row in by_great.iterrows():
    print(f"   {row['primaryName']:<30} {row['great_movies']:2} great movies (of {row['num_movies']})")

# Most consistent (low stddev with good average)
print("\n📊 Most consistent (low rating variance, avg 7+):")
consistent = result[result['avg_rating'] >= 7.0].nsmallest(10, 'rating_stddev')[['primaryName', 'rating_stddev', 'avg_rating', 'num_movies']]
for i, row in consistent.iterrows():
    print(f"   {row['primaryName']:<30} stddev: {row['rating_stddev']:.2f}, avg: {row['avg_rating']:.2f} ({row['num_movies']} movies)")
