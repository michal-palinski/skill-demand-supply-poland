#!/usr/bin/env python3
"""
Analyze skill categories from matched skills.
Group by skill type and count job offers.
"""

import sqlite3
import pandas as pd
from collections import Counter

# Load ESCO skills
esco = pd.read_parquet('df_skills_en_emb.parquet')
print(f"Total ESCO skills: {len(esco)}")

# Connect to embeddings DB
conn = sqlite3.connect('req_resp_embeddings.db')

# Get all matched skills with their types
query = """
SELECT DISTINCT 
    skill_label,
    skill_type,
    job_id
FROM skill_matches
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"\nTotal skill matches: {len(df)}")
print(f"Unique skills matched: {df['skill_label'].nunique()}")
print(f"Unique job offers with skills: {df['job_id'].nunique()}")

# Count by skill type
type_counts = df.groupby('skill_type')['job_id'].nunique().reset_index()
type_counts.columns = ['Skill Type', 'Job Offers']
type_counts = type_counts.sort_values('Job Offers', ascending=False)

print("\n" + "="*80)
print("SKILL CATEGORIES BY TYPE")
print("="*80)
print(type_counts.to_string(index=False))

# Get top skills per type
print("\n" + "="*80)
print("TOP 10 MOST POPULAR SKILLS BY TYPE")
print("="*80)

for skill_type in df['skill_type'].dropna().unique():
    type_df = df[df['skill_type'] == skill_type]
    top_skills = type_df.groupby('skill_label')['job_id'].nunique().sort_values(ascending=False).head(10)
    
    print(f"\n{skill_type.upper()}:")
    for skill, count in top_skills.items():
        print(f"  {skill}: {count} offers")

# Calculate percentages
total_offers = df['job_id'].nunique()
type_counts['Percentage'] = (type_counts['Job Offers'] / total_offers * 100).round(2)

print("\n" + "="*80)
print(f"SUMMARY (Total unique offers with skills: {total_offers})")
print("="*80)
print(type_counts.to_string(index=False))

# Save for chart
type_counts.to_csv('skill_categories_chart.csv', index=False)
print(f"\nSaved to: skill_categories_chart.csv")
