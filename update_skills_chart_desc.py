#!/usr/bin/env python3
"""Update skills chart description - remove sample size text."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Update Skills chart description
print("Updating skills chart description...")
skills_props = {
    "metadata": {
        "describe": {
            "intro": "Most demanded skills"
        }
    }
}

r = requests.patch(f"{BASE_URL}/charts/FMGJD", headers=headers, json=skills_props)
print(f"Update: {r.status_code}")

r = requests.post(f"{BASE_URL}/charts/FMGJD/publish", headers=headers)
version = r.json().get('version', 3)
print(f"Published: version {version}")

print(f"\nChart URL: https://datawrapper.dwcdn.net/FMGJD/{version}/")
