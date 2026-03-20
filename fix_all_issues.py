#!/usr/bin/env python3
"""Fix tooltip, shorten labels with ellipsis, update skills chart."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Load and update skills data with shortened labels
df = pd.read_csv('top_skills_chart.csv')

# Shorten long labels (max 45 chars with ellipsis)
def shorten_label(text, max_len=45):
    if len(text) > max_len:
        return text[:max_len-3] + "..."
    return text

df['Skill Short'] = df['Skill'].apply(lambda x: shorten_label(x, 45))
df['Skill Full'] = df['Skill']  # Keep full for tooltip

print("Updated skills with ellipsis:")
print(df[['Skill Short', 'Job Offers', 'Percentage']].to_string(index=False))

# Upload new data to skills chart
csv_data = "Skill,Percentage,Job Offers,Type,Skill Full\n"
for _, row in df.iterrows():
    csv_data += f'"{row["Skill Short"]}",{row["Percentage"]},{row["Job Offers"]},"{row["Type"]}","{row["Skill Full"]}"\n'

print("\nUploading data to FMGJD...")
r = requests.put(f"{BASE_URL}/charts/FMGJD/data", headers=headers, data=csv_data.encode('utf-8'))
print(f"Upload: {r.status_code}")

# Fix tooltip - use simpler format without HTML which works better in iframes
skills_props = {
    "metadata": {
        "visualize": {
            "tooltip": {
                "enabled": True
            },
            "labeling": "right"
        }
    }
}

print("Updating skills chart properties...")
r = requests.patch(f"{BASE_URL}/charts/FMGJD", headers=headers, json=skills_props)
print(f"Update: {r.status_code}")

# Publish
print("Publishing...")
r = requests.post(f"{BASE_URL}/charts/FMGJD/publish", headers=headers)
version = r.json().get('version', 5)
print(f"Published: version {version}")

# Also fix KZIS tooltip
kzis_props = {
    "metadata": {
        "visualize": {
            "tooltip": {
                "enabled": True
            }
        }
    }
}

print("\nUpdating KZIS chart...")
r = requests.patch(f"{BASE_URL}/charts/U2ao5", headers=headers, json=kzis_props)
print(f"Update: {r.status_code}")

r = requests.post(f"{BASE_URL}/charts/U2ao5/publish", headers=headers)
version_kzis = r.json().get('version', 9)
print(f"Published: version {version_kzis}")

print(f"\nSkills chart: https://datawrapper.dwcdn.net/FMGJD/{version}/")
print(f"KZIS chart: https://datawrapper.dwcdn.net/U2ao5/{version_kzis}/")
