#!/usr/bin/env python3
import pandas as pd
import os

os.chdir('/Users/jaakkosarela/imdb')
df = pd.read_csv('data/hollywood_actor_rankings.csv')

actresses = df[df['category'] == 'actress'].copy()
actresses = actresses.reset_index(drop=True)

print('=' * 90)
print('BEST ACTRESSES - MAJOR FILMS (50,000+ votes)')
print('=' * 90)

print('\n🎬 TOP 30 BY AVERAGE RATING:\n')
for i, row in actresses.head(30).iterrows():
    status = '' if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ''
    films = str(row['films'])[:100] + '...' if len(str(row['films'])) > 100 else row['films']
    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {int(row['num_movies'])} | Votes: {row['total_votes']:,.0f}")
    print(f"     {films}")
    print()

print('\n' + '=' * 90)
print('🏆 BEST CAREER ACTRESSES (10+ major films)')
print('=' * 90 + '\n')

career = actresses[actresses['num_movies'] >= 10].reset_index(drop=True)
for i, row in career.head(20).iterrows():
    status = '' if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ''
    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {int(row['num_movies'])} | Range: {row['min_rating']}-{row['max_rating']}")
    print()

print('\n' + '=' * 90)
print('🌟 ELITE TIER ACTRESSES (20+ major films)')
print('=' * 90 + '\n')

elite = actresses[actresses['num_movies'] >= 20].reset_index(drop=True)
for i, row in elite.head(15).iterrows():
    status = '' if row['deathYear'] == '\\N' else f" †{row['deathYear']}"
    birth = f"b.{row['birthYear']}" if row['birthYear'] != '\\N' else ''
    print(f"{i+1:3}. {row['primaryName']:<30} {birth}{status}")
    print(f"     Avg: {row['avg_rating']:.2f} | Movies: {int(row['num_movies'])} | Votes: {row['total_votes']:,.0f}")
    print()

print(f"\nTotal actresses in dataset: {len(actresses)}")
print(f"Actresses with 10+ films: {len(career)}")
print(f"Actresses with 20+ films: {len(elite)}")
