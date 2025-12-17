#!/usr/bin/env python3
"""Find best actors from Hollywood and European cinema."""

import duckdb
import os

os.chdir('/Users/jaakkosarela/imdb')
con = duckdb.connect()

print("Analyzing Hollywood & European cinema actors...")
print("This may take a few minutes...\n")

# Western regions: US, UK, Germany, France, Italy, Spain, etc.
WESTERN_REGIONS = "('US', 'GB', 'DE', 'FR', 'IT', 'ES', 'CA', 'AU', 'NL', 'BE', 'AT', 'CH', 'SE', 'DK', 'NO', 'FI', 'IE', 'PT', 'PL', 'CZ', 'HU', 'RO', 'GR')"

query = f"""
WITH western_movies AS (
    -- Get movies that have releases in Western regions
    SELECT DISTINCT titleId as tconst
    FROM read_csv_auto('data/title.akas.tsv', delim='\t', header=true, quote='')
    WHERE region IN {WESTERN_REGIONS}
),
actor_movies AS (
    SELECT
        p.nconst,
        p.category,
        p.ordering,
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
    JOIN western_movies w
        ON p.tconst = w.tconst
    WHERE p.category IN ('actor', 'actress')
      AND b.titleType = 'movie'
      AND b.isAdult = '0'
      AND r.numVotes >= 1000  -- Still need some votes for validity
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
        ROUND(MIN(am.averageRating), 1) as min_rating,
        ROUND(MAX(am.averageRating), 1) as max_rating,
        SUM(am.numVotes) as total_votes,
        STRING_AGG(am.primaryTitle || ' (' || am.startYear || ', ' || am.averageRating || ')', '; ' ORDER BY am.averageRating DESC) as films
    FROM actor_movies am
    JOIN read_csv_auto('data/name.basics.tsv', delim='\t', header=true, quote='') n
        ON am.nconst = n.nconst
    GROUP BY am.nconst, am.category, n.primaryName, n.birthYear, n.deathYear
)
SELECT * FROM actor_stats
ORDER BY avg_rating DESC, num_movies DESC
"""

print("Running query...")
result = con.execute(query).fetchdf()

# Save full results
result.to_csv('data/western_actor_rankings.csv', index=False)

print("=" * 100)
print("BEST AVERAGE RATING - HOLLYWOOD & EUROPEAN CINEMA")
print("(Movies with 1000+ votes, released in Western regions)")
print("=" * 100)

# Top 50 by pure average (any number of films)
print("\n🎬 TOP 50 BY AVERAGE RATING (any number of qualifying films):\n")
for i, row in result.head(50).iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    films_list = row['films'][:150] + "..." if len(row['films']) > 150 else row['films']

    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']} | Range: {row['min_rating']}-{row['max_rating']}")
    print(f"     Films: {films_list}")
    print()

# Also show those with exactly 1 film (the "one hit wonder" effect)
print("\n" + "=" * 100)
print("🎯 ONE-FILM WONDERS (appeared in exactly 1 qualifying Western film)")
print("=" * 100 + "\n")

one_film = result[result['num_movies'] == 1].head(30)
for i, row in one_film.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"    {row['primaryName']:<35} {birth}{status}")
    print(f"    Rating: {row['avg_rating']} | {row['films']}")
    print()

# More interesting: actors with 2-5 films, all highly rated
print("\n" + "=" * 100)
print("⭐ HIDDEN GEMS (2-5 films, average 8.0+)")
print("=" * 100 + "\n")

hidden_gems = result[(result['num_movies'] >= 2) & (result['num_movies'] <= 5) & (result['avg_rating'] >= 8.0)].head(30)
for i, row in hidden_gems.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    films_list = row['films'][:200] + "..." if len(row['films']) > 200 else row['films']

    print(f"    {row['primaryName']:<35} {birth}{status}")
    print(f"    Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']}")
    print(f"    {films_list}")
    print()

print(f"\n✅ Full results saved to data/western_actor_rankings.csv ({len(result)} actors)")
