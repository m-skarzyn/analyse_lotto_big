# collect/api_fetcher.py
import requests
import pandas as pd
import time

def fetch_api(api_url: str, params: dict, max_retries: int = 3, delay: float = 5):
    for attempt in range(max_retries):
        try:
            resp = requests.get(api_url, params=params, timeout=30)
            resp.raise_for_status()
            return pd.DataFrame(resp.json())
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    raise RuntimeError("All retries exhausted")
