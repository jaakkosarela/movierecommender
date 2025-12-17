#!/usr/bin/env python3
"""Find best actors from Hollywood and European cinema - stricter filtering."""

import duckdb
import os

os.chdir('/Users/jaakkosarela/imdb')
con = duckdb.connect()

print("Analyzing Hollywood & European cinema actors (strict filter)...")
print("This may take a few minutes...\n")

# Strategy: Use original title language from akas + higher vote threshold
# Films with English/European original titles AND high vote counts are likely Western
query = """
WITH western_originals AS (
    -- Movies where the original title is in a Western language region
    SELECT DISTINCT titleId as tconst
    FROM read_csv_auto('data/title.akas.tsv', delim='\t', header=true, quote='')
    WHERE isOriginalTitle = '1'
      AND (
        region IN ('US', 'GB', 'XWW', '\\N')  -- XWW = worldwide, NULL often = English
        OR language IN ('en', 'de', 'fr', 'it', 'es', 'pt', 'nl', 'sv', 'da', 'no', 'pl', 'cs', 'hu', 'ro', 'el')
      )
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
    JOIN western_originals w
        ON p.tconst = w.tconst
    WHERE p.category IN ('actor', 'actress')
      AND b.titleType = 'movie'
      AND b.isAdult = '0'
      AND r.numVotes >= 5000  -- Higher threshold to filter out niche films
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

result.to_csv('data/western_actor_rankings_v2.csv', index=False)

print("=" * 100)
print("BEST AVERAGE RATING - HOLLYWOOD & EUROPEAN CINEMA (STRICT)")
print("(Movies with 5000+ votes, original language English/European)")
print("=" * 100)

# Top 50 by pure average
print("\n🎬 TOP 50 BY AVERAGE RATING:\n")
for i, row in result.head(50).iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    films_list = row['films'][:120] + "..." if len(row['films']) > 120 else row['films']

    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']} | Votes: {row['total_votes']:,.0f}")
    print(f"     Films: {films_list}")
    print()

# One-film wonders with recognizable films
print("\n" + "=" * 100)
print("🎯 ONE-FILM WONDERS (1 qualifying film, 5000+ votes)")
print("=" * 100 + "\n")

one_film = result[result['num_movies'] == 1].head(40)
for i, row in one_film.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"    {row['primaryName']:<35} {birth}{status}")
    print(f"    {row['avg_rating']} | {row['total_votes']:,.0f} votes | {row['films']}")
    print()

# Hidden gems: 2-5 films, all great
print("\n" + "=" * 100)
print("⭐ HIDDEN GEMS (2-5 films, average 8.0+)")
print("=" * 100 + "\n")

hidden_gems = result[(result['num_movies'] >= 2) & (result['num_movies'] <= 5) & (result['avg_rating'] >= 8.0)].head(30)
for i, row in hidden_gems.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    films_list = row['films']

    print(f"    {row['primaryName']:<35} {birth}{status}")
    print(f"    Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']} | Votes: {row['total_votes']:,.0f}")
    print(f"    {films_list}")
    print()

# Best with 10+ films
print("\n" + "=" * 100)
print("🏆 BEST CAREER ACTORS (10+ films, sorted by average)")
print("=" * 100 + "\n")

career = result[result['num_movies'] >= 10].head(30)
for i, row in career.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""

    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']} | Range: {row['min_rating']}-{row['max_rating']} | Votes: {row['total_votes']:,.0f}")
    print()

print(f"\n✅ Saved to data/western_actor_rankings_v2.csv ({len(result)} actors)")
