#!/usr/bin/env python3
"""Update Datawrapper chart - make it vertical and sorted by frequency."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"
CHART_ID = "U2ao5"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Load data
df = pd.read_csv("kzis_categories_chart.csv")

# Sort by job offers (descending)
df = df.sort_values('Job Offers', ascending=False)

# Remove count from example
df['Example'] = df['Example Occupation'].str.replace(r'\s*\(\d{1,3}(?:,\d{3})*\)', '', regex=True)

print("Data to upload (sorted by frequency):")
print(df[['Category', 'Example', 'Job Offers', 'Percentage']].to_string(index=False))
print()

# Create labels as "KZIS-X (e.g. Example)"
csv_data = "Category,Job Offers,Percentage\n"
for _, row in df.iterrows():
    label = f"KZIS-{row['Category']} (e.g. {row['Example']})"
    csv_data += f'"{label}",{row["Job Offers"]},{row["Percentage"]}\n'

print("Uploading data...")
response = requests.put(
    f"{BASE_URL}/charts/{CHART_ID}/data",
    headers=headers,
    data=csv_data.encode('utf-8')
)
print(f"Upload status: {response.status_code}")

# Update chart to column chart (vertical) with proper settings
chart_props = {
    "type": "column-chart",
    "metadata": {
        "describe": {
            "intro": "Distribution of 186,030 job offers across KZIS occupation categories"
        },
        "visualize": {
            "base-color": "#4d94b8",
            "custom-colors": {},
            "labeling": "top",
            "show-grid": True,
            "sort": False,  # Don't sort in chart - we already sorted the data
            "rotate-labels": False
        }
    }
}

print("Updating chart type and properties...")
response = requests.patch(
    f"{BASE_URL}/charts/{CHART_ID}",
    headers=headers,
    json=chart_props
)
print(f"Properties status: {response.status_code}")

# Republish
print("Publishing...")
response = requests.post(f"{BASE_URL}/charts/{CHART_ID}/publish", headers=headers)
print(f"Publish status: {response.status_code}")

version = response.json().get('version', 4)
print(f"\nChart updated: https://datawrapper.dwcdn.net/{CHART_ID}/{version}/")
print(f"Total offers: {df['Job Offers'].sum():,}")
print("\nCategories sorted by frequency (highest first):")
for _, row in df.iterrows():
    print(f"  KZIS-{row['Category']}: {row['Job Offers']:,} offers ({row['Percentage']}%)")
