#!/usr/bin/env python3
"""Find best actors - Hollywood/European only using high vote threshold."""

import duckdb
import os

os.chdir('/Users/jaakkosarela/imdb')
con = duckdb.connect()

print("Analyzing actors from major Western films...")
print("Using 50,000+ votes filter (captures Hollywood/major European releases)...\n")

query = """
WITH actor_movies AS (
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
    WHERE p.category IN ('actor', 'actress')
      AND b.titleType = 'movie'
      AND b.isAdult = '0'
      AND r.numVotes >= 50000  -- High threshold = mostly Hollywood
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

result.to_csv('data/hollywood_actor_rankings.csv', index=False)

print("=" * 100)
print("BEST ACTORS - MAJOR FILMS (50,000+ votes)")
print("=" * 100)

# Top by pure average
print("\n🎬 TOP 50 BY AVERAGE RATING:\n")
for i, row in result.head(50).iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    films_list = row['films'][:100] + "..." if len(row['films']) > 100 else row['films']

    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']} | Votes: {row['total_votes']:,.0f}")
    print(f"     Films: {films_list}")
    print()

# One-film wonders
print("\n" + "=" * 100)
print("🎯 ONE-FILM WONDERS (best single performance in a major film)")
print("=" * 100 + "\n")

one_film = result[result['num_movies'] == 1].head(30)
for i, row in one_film.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"    {row['primaryName']:<35} {birth}{status}")
    print(f"    {row['avg_rating']} | {row['total_votes']:,.0f} votes | {row['films']}")
    print()

# Hidden gems: 2-5 films
print("\n" + "=" * 100)
print("⭐ HIDDEN GEMS (2-5 major films, average 8.0+)")
print("=" * 100 + "\n")

hidden = result[(result['num_movies'] >= 2) & (result['num_movies'] <= 5) & (result['avg_rating'] >= 8.0)].head(25)
for i, row in hidden.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"    {row['primaryName']:<35} {birth}{status}")
    print(f"    Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']}")
    print(f"    {row['films'][:200]}")
    print()

# Best careers (10+ major films)
print("\n" + "=" * 100)
print("🏆 BEST CAREER ACTORS (10+ major films, by average)")
print("=" * 100 + "\n")

career = result[result['num_movies'] >= 10].head(40)
for i, row in career.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']} | Range: {row['min_rating']}-{row['max_rating']}")
    print()

# Best with 20+ major films
print("\n" + "=" * 100)
print("🌟 ELITE TIER (20+ major films, by average)")
print("=" * 100 + "\n")

elite = result[result['num_movies'] >= 20].head(30)
for i, row in elite.iterrows():
    status = "" if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ""
    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {row['num_movies']} | Range: {row['min_rating']}-{row['max_rating']} | Votes: {row['total_votes']:,.0f}")
    print()

print(f"\n✅ Saved to data/hollywood_actor_rankings.csv ({len(result)} actors)")
