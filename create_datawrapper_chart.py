#!/usr/bin/env python3
"""
Create Datawrapper chart for KZIS categories.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"

# Load data
df = pd.read_csv("kzis_categories_chart.csv")

# Prepare data for chart
chart_data = df[['Category', 'Category Name', 'Job Offers', 'Percentage', 'Example Occupation']].copy()

# Format for display
chart_data['Label'] = chart_data.apply(
    lambda r: f"{r['Category']}. {r['Category Name']}\n(e.g., {r['Example Occupation']})", 
    axis=1
)

# Create chart via API
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Create chart
create_payload = {
    "title": "Job Offers by KZIS Occupation Category",
    "type": "column-chart",
    "metadata": {
        "describe": {
            "intro": "Distribution of 186,030 job offers across 10 KZIS occupation categories based on matched standardized occupations"
        }
    }
}

print("Creating chart...")
response = requests.post(f"{BASE_URL}/charts", headers=headers, json=create_payload)
chart_id = response.json()["id"]
print(f"Chart ID: {chart_id}")

# 2. Upload data
csv_data = f"Category,Job Offers,Percentage\n"
for _, row in chart_data.iterrows():
    label = f"{row['Category']}. {row['Category Name']}"
    csv_data += f"{label},{row['Job Offers']},{row['Percentage']}\n"

print("Uploading data...")
requests.put(
    f"{BASE_URL}/charts/{chart_id}/data",
    headers={"Authorization": f"Bearer {API_KEY}"},
    data=csv_data.encode('utf-8')
)

# 3. Update chart properties
print("Configuring chart...")
config = {
    "metadata": {
        "visualize": {
            "basemap": "",
            "custom-colors": {},
            "highlighted-series": [],
            "highlighted-values": [],
            "labeling": "right",
            "show-color-legend": False,
            "sort-bars": False,
            "value-labels": "on",
            "x-grid": "off",
            "y-grid": "on"
        },
        "describe": {
            "intro": "Distribution of 186,030 job offers across 10 KZIS occupation categories",
            "source-name": "Pracuj.pl job ads 2025, matched to KZIS classification"
        },
        "axes": {
            "values": "Job Offers"
        }
    }
}

requests.patch(
    f"{BASE_URL}/charts/{chart_id}",
    headers=headers,
    json=config
)

# 4. Publish
print("Publishing chart...")
requests.post(
    f"{BASE_URL}/charts/{chart_id}/publish",
    headers=headers
)

# Get public URL
response = requests.get(f"{BASE_URL}/charts/{chart_id}", headers=headers)
public_url = response.json().get("publicUrl", "")

print(f"\n{'='*80}")
print(f"Chart created successfully!")
print(f"Chart ID: {chart_id}")
print(f"Public URL: {public_url}")
print(f"{'='*80}")

# Display summary
print("\nCategory Summary:")
for _, row in chart_data.iterrows():
    print(f"{row['Category']}. {row['Category Name']:50s} {row['Job Offers']:>7,} offers ({row['Percentage']:>5.1f}%)")
    print(f"   Example: {row['Example Occupation']}")

# Save chart ID
with open("datawrapper_chart_id.txt", "w") as f:
    f.write(f"Chart ID: {chart_id}\n")
    f.write(f"Public URL: {public_url}\n")

print(f"\nChart ID saved to datawrapper_chart_id.txt")
