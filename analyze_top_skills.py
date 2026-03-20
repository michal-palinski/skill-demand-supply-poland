#!/usr/bin/env python3
"""
Analyze top skills from matched skills for chart.
"""

import sqlite3
import pandas as pd

# Connect to embeddings DB
conn = sqlite3.connect('req_resp_embeddings.db')

# Get all matched skills
query = """
SELECT 
    skill_label,
    skill_type,
    job_id
FROM skill_matches
WHERE rank = 1
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"Total top-1 skill matches: {len(df)}")
print(f"Unique job offers: {df['job_id'].nunique()}")

# Count offers per skill
skill_counts = df.groupby(['skill_label', 'skill_type'])['job_id'].nunique().reset_index()
skill_counts.columns = ['Skill', 'Skill Type', 'Job Offers']
skill_counts = skill_counts.sort_values('Job Offers', ascending=False)

# Get top 12 skills
top_skills = skill_counts.head(12).copy()

# Calculate percentage
total_offers = df['job_id'].nunique()
top_skills['Percentage'] = (top_skills['Job Offers'] / total_offers * 100).round(2)

print("\n" + "="*100)
print("TOP 12 MOST DEMANDED SKILLS")
print("="*100)
print(top_skills.to_string(index=False))

# Shorten skill names for chart (keep first 50 chars)
top_skills['Skill Short'] = top_skills['Skill'].str[:60]

# Save for Datawrapper
chart_data = top_skills[['Skill Short', 'Skill Type', 'Job Offers', 'Percentage']].copy()
chart_data.columns = ['Skill', 'Type', 'Job Offers', 'Percentage']

chart_data.to_csv('top_skills_chart.csv', index=False)
print(f"\n{chr(10)}Saved to: top_skills_chart.csv")
print(f"Total sample offers: {total_offers}")
