#!/usr/bin/env python3
"""
Analyze KZIS categories based on first digit of code.
Generate data for Datawrapper chart with most popular occupation as example.
"""

import sqlite3
import pandas as pd

# Category names based on ISCO-08 standard (first digit)
CATEGORY_NAMES = {
    "0": "Armed Forces",
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians and Associate Professionals",
    "4": "Clerical Support Workers",
    "5": "Service and Sales Workers",
    "6": "Skilled Agricultural, Forestry and Fishery Workers",
    "7": "Craft and Related Trades Workers",
    "8": "Plant and Machine Operators, and Assemblers",
    "9": "Elementary Occupations"
}

# Connect to KZIS DB to get codes
conn_kzis = sqlite3.connect("kzis_occupations_descriptions.db")
kzis_df = pd.read_sql("SELECT kod, nazwa FROM opisy_zawodow", conn_kzis)
conn_kzis.close()

# Extract category (first digit)
kzis_df['category'] = kzis_df['kod'].astype(str).str[0]

# Connect to jobs DB
conn_jobs = sqlite3.connect("jobs_database.db")
c = conn_jobs.cursor()

# Get all top-1 KZIS matches
c.execute("""
    SELECT kzis_occupation_name, COUNT(*) as job_count
    FROM job_kzis_matches
    WHERE rank = 1
    GROUP BY kzis_occupation_name
""")
matches = c.fetchall()
conn_jobs.close()

# Create DataFrame
matches_df = pd.DataFrame(matches, columns=['nazwa', 'job_count'])

# Merge with KZIS to get codes and categories
merged = matches_df.merge(kzis_df, on='nazwa', how='left')

# Count by category
category_stats = merged.groupby('category').agg({
    'job_count': 'sum'
}).reset_index()
category_stats.columns = ['category', 'job_offers']

# Calculate percentage
total_offers = category_stats['job_offers'].sum()
category_stats['percentage'] = (category_stats['job_offers'] / total_offers * 100).round(2)

# Add category names
category_stats['category_name'] = category_stats['category'].map(CATEGORY_NAMES)

# Get the most popular occupation in each category
examples = []
for cat in category_stats['category']:
    cat_occupations = merged[merged['category'] == cat].nlargest(1, 'job_count')
    if len(cat_occupations) > 0:
        example = cat_occupations.iloc[0]['nazwa']
        example_count = cat_occupations.iloc[0]['job_count']
        examples.append(f"{example} ({example_count:,})")
    else:
        examples.append("")

category_stats['example'] = examples

# Sort by category
category_stats = category_stats.sort_values('category')

# Display results
print("KZIS Categories Analysis")
print("=" * 120)
print(f"Total job offers matched: {total_offers:,}")
print()
for _, row in category_stats.iterrows():
    print(f"Category {row['category']}: {row['category_name']}")
    print(f"  Offers: {row['job_offers']:,} ({row['percentage']}%)")
    print(f"  Top occupation: {row['example']}")
    print()
print("=" * 120)

# Filter out category 0 for chart
category_stats_filtered = category_stats[category_stats['category'] != '0'].copy()

# Save for Datawrapper
datawrapper_data = category_stats_filtered[['category', 'category_name', 'job_offers', 'percentage', 'example']].copy()
datawrapper_data.columns = ['Category', 'Category Name', 'Job Offers', 'Percentage', 'Example Occupation']

# Save CSV
csv_path = "kzis_categories_chart.csv"
datawrapper_data.to_csv(csv_path, index=False)
print(f"\nCSV saved to: {csv_path}")
print("\nDatawrapper CSV preview:")
print(datawrapper_data.to_string(index=False))
