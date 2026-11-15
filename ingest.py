import os
import json
from io import StringIO
import numpy as np
import pandas as pd
import requests

RAW_FILE = "data/raw_events.parquet"
CONFIG_FILE = "config.json"
INGEST_CONFIG_FILE = "ingest_config.json"

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_l_group(df: pd.DataFrame, cfg: dict) -> pd.Series:
    cols = cfg["group_rule"]["columns"]
    div = cfg["group_rule"]["divisor"]
    sep = cfg["group_rule"]["separator"]
    buckets = np.floor(df[cols] / div + 0.5).astype("int32")
    return buckets.astype(str).agg(sep.join, axis=1)

def update_from_csv(url: str | None = None) -> int:
    os.makedirs("data", exist_ok=True)
    cfg = load_json(CONFIG_FILE)
    ingest_cfg = load_json(INGEST_CONFIG_FILE)
    
    target_url = url if url else ingest_cfg["csv_url"]
    csv_opts = ingest_cfg.get("csv_options", {})
    col_map = ingest_cfg["column_mapping"]
    
    response = requests.get(target_url, timeout=20)
    response.raise_for_status()
    response.encoding = csv_opts.get("encoding", "utf-8")
    
    raw_csv = pd.read_csv(
        StringIO(response.text),
        sep=csv_opts.get("separator", ","),
        decimal=csv_opts.get("decimal", ".")
    )
    
    # Odwrócone mapowanie: nazwa_kolumny_w_csv -> nazwa_wewnetrzna
    inv_map = {v: k for k, v in col_map.items()}
    missing_cols = [c for c in inv_map.keys() if c not in raw_csv.columns]
    if missing_cols:
        raise ValueError(f"Brak wymaganych kolumn w pliku CSV: {missing_cols}")
        
    new_df = raw_csv[list(inv_map.keys())].rename(columns=inv_map).copy()
    
    new_df["ID_LOS"] = new_df["ID_LOS"].astype("int64")
    date_fmt = csv_opts.get("date_format", None)
    dt_series = pd.to_datetime(new_df["DATE_LOS"], format=date_fmt)
    new_df["DATE_LOS"] = dt_series.dt.date
    new_df["YEAR_LOS"] = dt_series.dt.year.astype("int32")
    
    for col in ["L_1", "L_2", "L_3", "L_4", "L_5", "L_6"]:
        new_df[col] = new_df[col].astype("int32")
        
    new_df["L_GROUP"] = compute_l_group(new_df, cfg)
    
    ordered_cols = ["ID_LOS", "DATE_LOS", "YEAR_LOS", "L_1", "L_2", "L_3", "L_4", "L_5", "L_6", "L_GROUP"]
    new_df = new_df[ordered_cols]

    if not os.path.exists(RAW_FILE):
        new_df = new_df.sort_values(["DATE_LOS", "ID_LOS"]).reset_index(drop=True)
        new_df.to_parquet(RAW_FILE, engine="pyarrow", index=False)
        return len(new_df)

    existing_df = pd.read_parquet(RAW_FILE, engine="pyarrow")
    existing_keys = set(zip(existing_df["ID_LOS"], existing_df["DATE_LOS"]))
    
    mask_new = [
        (r_id, r_date) not in existing_keys 
        for r_id, r_date in zip(new_df["ID_LOS"], new_df["DATE_LOS"])
    ]
    truly_new = new_df[mask_new]

    if truly_new.empty:
        return 0

    updated_df = pd.concat([existing_df, truly_new], ignore_index=True)
    updated_df = updated_df.sort_values(["DATE_LOS", "ID_LOS"]).reset_index(drop=True)
    updated_df["L_GROUP"] = compute_l_group(updated_df, cfg)
    updated_df.to_parquet(RAW_FILE, engine="pyarrow", index=False)
    return len(truly_new)

if __name__ == "__main__":
    added = update_from_csv()
    print(f"Zakończono ładowanie. Dodano nowych rekordów: {added}")
