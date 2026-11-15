# analyse_lotto_big

## Project Layout
```
project/
├── config.py          # API keys, URLs, paths (use env vars for secrets)
├── collect/
│   ├── __init__.py
│   ├── api_fetcher.py
│   └── web_scraper.py
├── process/
│   ├── __init__.py
│   ├── clean.py
│   └── transform.py
├── output/
│   ├── raw/           # untouched data
│   └── processed/
├── main.py            # orchestrates the pipeline
└── requirements.txt   
```

## Collection
collect/api_fetcher.py

## Processing
process/clean.py

## Orchestration
main.py

## Key Libraries
| Purpose | Library |
|---|---|
| HTTP / APIs | `requests`, `httpx` |
| Web scraping | `beautifulsoup4`, `playwright` (JS-heavy sites) |
| Data manipulation | `pandas`, `numpy` |
| Scheduling | `APScheduler`, `cron`, or `Airflow` |
| Storage | `sqlite3, `sqlalchemy`, or just CSV/Parquet files |
| Config/secrets | `python-dotenv` |

## Tips
- Idempotency: make each step safe to re-run (upsert, not blind append). 
- Retry with backoff: wrap API calls in a retry loop with exponential delay.
- Log everything: use Python's logging module, not print.
- Separate raw from processed: never overwrite your raw data.
- Schedule it: APScheduler for simple cron-like tasks, Airflow if the DAG grows complex.

