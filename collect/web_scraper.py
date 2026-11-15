# collect/web_scraper.py
import requests
from bs4 import BeautifulSoup

def scrape(url: str) -> list[dict]:
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(resp.text, 'html.parser')
    # extract what you need
    return [{"title": t.text} for t in soup.find_all('h2')]
