#!/usr/bin/env python3
"""Update Datawrapper chart - short English category names, horizontal bars, sorted."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"
CHART_ID = "U2ao5"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Shortened English translations of KZIS categories
SHORT_NAMES = {
    1: "Officials & Managers",
    2: "Specialists",
    3: "Technicians",
    4: "Office Workers",
    5: "Service & Sales",
    6: "Agriculture & Forestry",
    7: "Industrial & Craft",
    8: "Machine Operators",
    9: "Elementary Workers",
}

# Load data
df = pd.read_csv("kzis_categories_chart.csv")

# Sort by job offers (descending)
df = df.sort_values('Job Offers', ascending=False)

# Build CSV with short labels
csv_data = "Category,Job Offers,Percentage\n"
for _, row in df.iterrows():
    cat = int(row['Category'])
    label = f"KZIS-{cat}: {SHORT_NAMES[cat]}"
    csv_data += f'"{label}",{row["Job Offers"]},{row["Percentage"]}\n'

print("Data:")
print(csv_data)

print("Uploading data...")
r = requests.put(f"{BASE_URL}/charts/{CHART_ID}/data", headers=headers, data=csv_data.encode('utf-8'))
print(f"Upload: {r.status_code}")

# Change to horizontal bar chart for readability
chart_props = {
    "type": "d3-bars",
    "metadata": {
        "describe": {
            "intro": "Distribution of 186,030 job offers across KZIS occupation categories"
        },
        "visualize": {
            "base-color": "#4d94b8",
            "sort-bars": False,
            "labeling": "right"
        }
    }
}

print("Updating chart type to horizontal bars...")
r = requests.patch(f"{BASE_URL}/charts/{CHART_ID}", headers=headers, json=chart_props)
print(f"Properties: {r.status_code}")

print("Publishing...")
r = requests.post(f"{BASE_URL}/charts/{CHART_ID}/publish", headers=headers)
print(f"Publish: {r.status_code}")

version = r.json().get('version', 5)
print(f"\nChart updated: https://datawrapper.dwcdn.net/{CHART_ID}/{version}/")
