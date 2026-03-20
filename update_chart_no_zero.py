#!/usr/bin/env python3
"""Update Datawrapper chart - exclude category 0."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"
CHART_ID = "U2ao5"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Load and filter data (exclude category 0)
df = pd.read_csv("kzis_categories_chart.csv")
df_filtered = df[df['Category'] != '0'].copy()

# Upload new data
csv_data = "Category,Job Offers,Percentage\n"
for _, row in df_filtered.iterrows():
    label = f"{row['Category']}. {row['Category Name']}"
    csv_data += f"{label},{row['Job Offers']},{row['Percentage']}\n"

print(f"Updating chart {CHART_ID}...")
requests.put(
    f"{BASE_URL}/charts/{CHART_ID}/data",
    headers=headers,
    data=csv_data.encode('utf-8')
)

# Republish
print("Publishing...")
requests.post(f"{BASE_URL}/charts/{CHART_ID}/publish", headers=headers)

print(f"\nChart updated: https://datawrapper.dwcdn.net/{CHART_ID}/2/")
print(f"\nShowing {len(df_filtered)} categories (excluded category 0: Armed Forces)")
