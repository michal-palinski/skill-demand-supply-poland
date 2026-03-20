#!/usr/bin/env python3
"""Fix tooltips in both charts."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATAWRAPPER_API_KEY")
BASE_URL = "https://api.datawrapper.de/v3"
headers = {"Authorization": f"Bearer {API_KEY}"}

# Fix KZIS chart (U2ao5)
print("Fixing KZIS categories chart...")
kzis_props = {
    "metadata": {
        "visualize": {
            "tooltip": {
                "enabled": True,
                "body": "<strong>{{ Category }}</strong><br>{{ Job Offers }} offers ({{ Percentage }}%)<br><em>e.g. {{ Example }}</em>"
            }
        }
    }
}

r = requests.patch(f"{BASE_URL}/charts/U2ao5", headers=headers, json=kzis_props)
print(f"KZIS update: {r.status_code}")

r = requests.post(f"{BASE_URL}/charts/U2ao5/publish", headers=headers)
version = r.json().get('version', 8)
print(f"KZIS published: version {version}")

# Fix Skills chart (FMGJD)
print("\nFixing skills chart...")
skills_props = {
    "metadata": {
        "visualize": {
            "tooltip": {
                "enabled": True,
                "body": "<strong>{{ Skill }}</strong><br>{{ Job Offers }} offers ({{ Percentage }}%)<br><em>Type: {{ Type }}</em>"
            }
        }
    }
}

r = requests.patch(f"{BASE_URL}/charts/FMGJD", headers=headers, json=skills_props)
print(f"Skills update: {r.status_code}")

r = requests.post(f"{BASE_URL}/charts/FMGJD/publish", headers=headers)
version = r.json().get('version', 2)
print(f"Skills published: version {version}")

print("\nKZIS chart: https://datawrapper.dwcdn.net/U2ao5/8/")
print("Skills chart: https://datawrapper.dwcdn.net/FMGJD/2/")
