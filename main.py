from collect.api_fetcher import fetch_api
from process.clean import clean
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
    raw = fetch_api("https://api.example.com/data", {"period": "monthly"})
    raw.to_csv("output/raw/raw_data.csv", index=False)
    logger.info(f"Collected {len(raw)} rows")

    processed = clean(raw)
    processed.to_csv("output/processed/clean_data.csv", index=False)
    logger.info(f"Processed {len(processed)} rows")

if __name__ == "__main__":
    main()
