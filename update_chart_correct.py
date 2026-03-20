#!/usr/bin/env python3
"""Update Datawrapper chart with correct data and popular examples."""

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

print("Data to upload:")
print(df.to_string(index=False))
print()

# Create chart data - simplify example to just occupation name
df_chart = df.copy()
# Remove the count from example (keep just name)
df_chart['Example'] = df_chart['Example Occupation'].str.replace(r'\s*\(\d{1,3}(?:,\d{3})*\)', '', regex=True)

# Upload data
csv_data = "Category,Job Offers,Percentage,Example\n"
for _, row in df_chart.iterrows():
    label = f"{row['Category']}. {row['Category Name']}"
    csv_data += f'"{label}",{row["Job Offers"]},{row["Percentage"]},"{row["Example"]}"\n'

print("Updating chart...")
response = requests.put(
    f"{BASE_URL}/charts/{CHART_ID}/data",
    headers=headers,
    data=csv_data.encode('utf-8')
)
print(f"Upload status: {response.status_code}")

# Update chart properties to show example
chart_props = {
    "metadata": {
        "describe": {
            "intro": "Distribution of 186,030 job offers across 10 KZIS occupation categories"
        },
        "visualize": {
            "base-color": "#4d94b8",
            "custom-colors": {},
            "labeling": "right",
            "show-grid": True
        }
    }
}

print("Updating properties...")
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

print(f"\nChart updated: https://datawrapper.dwcdn.net/{CHART_ID}/3/")
print(f"Total offers: {df['Job Offers'].sum():,}")
