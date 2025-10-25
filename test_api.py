# DAY 1: test_api.py
# Tests connectivity to data.gov.in API

import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("DATA_GOV_API_KEY")
BASE_URL = "https://api.data.gov.in/resource"

print("🔧 Testing API Connectivity...\n")

# Search for datasets first
search_url = "https://api.data.gov.in/catalog/package_search"
params = {"q": "crop production area yield", "rows": 5}

try:
    response = requests.get(search_url, params=params, timeout=30)
    data = response.json()
    
    print(f"✅ Search successful! Found {len(data['result']['results'])} datasets\n")
    
    for idx, dataset in enumerate(data['result']['results'][:3], 1):
        print(f"{idx}. {dataset['title']}")
        if dataset.get('resources'):
            for res in dataset['resources'][:1]:
                print(f"   Resource ID: {res.get('id')}")
                print(f"   Format: {res.get('format')}")
        print()
        
except Exception as e:
    print(f"❌ Error: {e}")
