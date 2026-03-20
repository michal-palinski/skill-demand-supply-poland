#!/usr/bin/env python3
"""Update Datawrapper chart - fix tooltip with job offers count and example."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"
CHART_ID = "U2ao5"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Shortened English translations
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
df = df.sort_values('Job Offers', ascending=False)

# Remove count from example (keep just name)
df['Example Clean'] = df['Example Occupation'].str.replace(r'\s*\(\d{1,3}(?:,\d{3})*\)', '', regex=True)

# Build CSV with all columns for tooltip
csv_data = "Category,Percentage,Job Offers,Example\n"
for _, row in df.iterrows():
    cat = int(row['Category'])
    label = f"KZIS-{cat}: {SHORT_NAMES[cat]}"
    example = row['Example Clean']
    csv_data += f'"{label}",{row["Percentage"]},{row["Job Offers"]},"{example}"\n'

print("Data:")
print(csv_data)

print("\nUploading data...")
r = requests.put(f"{BASE_URL}/charts/{CHART_ID}/data", headers=headers, data=csv_data.encode('utf-8'))
print(f"Upload: {r.status_code}")

# Update chart with custom tooltip
chart_props = {
    "type": "d3-bars",
    "metadata": {
        "describe": {
            "intro": "Distribution of 186,030 job offers across KZIS occupation categories"
        },
        "visualize": {
            "base-color": "#4d94b8",
            "sort-bars": False,
            "labeling": "right",
            "labels": {
                "show": True
            },
            "tooltip": {
                "enabled": True,
                "body": "{{ Job Offers }} offers ({{ Percentage }}%)<br><b>e.g.</b> {{ Example }}"
            }
        },
        "data": {
            "column-format": {
                "Percentage": {
                    "type": "auto",
                    "number-append": "%",
                    "number-format": "n1"
                },
                "Job Offers": {
                    "type": "auto",
                    "number-format": "n0"
                }
            }
        },
        "axes": {
            "values": "Percentage"
        }
    }
}

print("Updating chart properties with custom tooltip...")
r = requests.patch(f"{BASE_URL}/charts/{CHART_ID}", headers=headers, json=chart_props)
print(f"Properties: {r.status_code}")
if r.status_code != 200:
    print(f"Error: {r.text}")

print("Publishing...")
r = requests.post(f"{BASE_URL}/charts/{CHART_ID}/publish", headers=headers)
print(f"Publish: {r.status_code}")

version = r.json().get('version', 7)
print(f"\nChart updated: https://datawrapper.dwcdn.net/{CHART_ID}/{version}/")
print("\nTooltip will show:")
print("- Job offers count (formatted with comma)")
print("- Percentage")
print("- Most popular job example in category")
