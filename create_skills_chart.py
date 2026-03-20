#!/usr/bin/env python3
"""Create Datawrapper chart for top skills."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Load data
df = pd.read_csv('top_skills_chart.csv')

print("Top skills data:")
print(df.to_string(index=False))

# Create new chart
chart_config = {
    "title": "Top Skills in Job Offers",
    "type": "d3-bars",
    "metadata": {
        "describe": {
            "intro": "Most demanded skills in 4,000 sample job offers"
        },
        "visualize": {
            "base-color": "#7a5a1a",
            "sort-bars": False,
            "labeling": "right",
            "labels": {
                "show": True
            },
            "tooltip": {
                "enabled": True,
                "body": "{{ Job Offers }} offers ({{ Percentage }}%)<br><b>Type:</b> {{ Type }}"
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

print("\nCreating chart...")
r = requests.post(f"{BASE_URL}/charts", headers=headers, json=chart_config)
chart = r.json()
chart_id = chart.get('id')
print(f"Chart created with ID: {chart_id}")

# Upload data
csv_data = "Skill,Percentage,Job Offers,Type\n"
for _, row in df.iterrows():
    csv_data += f'"{row["Skill"]}",{row["Percentage"]},{row["Job Offers"]},"{row["Type"]}"\n'

print("Uploading data...")
r = requests.put(f"{BASE_URL}/charts/{chart_id}/data", headers=headers, data=csv_data.encode('utf-8'))
print(f"Upload: {r.status_code}")

# Publish
print("Publishing...")
r = requests.post(f"{BASE_URL}/charts/{chart_id}/publish", headers=headers)
print(f"Publish: {r.status_code}")

print(f"\nChart URL: https://datawrapper.dwcdn.net/{chart_id}/1/")
print(f"Chart ID: {chart_id}")

# Save chart ID
with open('.chart_skills_id.txt', 'w') as f:
    f.write(chart_id)

print("\nChart ID saved to .chart_skills_id.txt")
