import os
import json
import argparse
import numpy as np
import pandas as pd
import requests
from io import StringIO

RAW_FILE = "data/raw_events.parquet"
INGEST_CONFIG_FILE = "ingest_config.json"
CONFIG_FILE = "config.json"

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_l_group(df: pd.DataFrame, cfg: dict) -> pd.Series:
    cols = cfg["group_rule"]["columns"]
    div = cfg["group_rule"]["divisor"]
    sep = cfg["group_rule"]["separator"]
    buckets = np.floor(df[cols] / div + 0.5).astype("int32")
    return buckets.astype(str).agg(sep.join, axis=1)

def update_from_csv(url_override: str = None) -> int:
    os.makedirs("data", exist_ok=True)
    in_cfg = _load_json(INGEST_CONFIG_FILE)
    cfg = _load_json(CONFIG_FILE)
    
    url = url_override or in_cfg["csv_url"]
    sep = in_cfg.get("separator", ",")
    enc = in_cfg.get("encoding", "utf-8")
    date_fmt = in_cfg.get("date_format", None)
    has_header = in_cfg.get("has_header", True)
    
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    response.encoding = enc
    
    if has_header:
        raw_csv = pd.read_csv(StringIO(response.text), sep=sep, header=0)
        mapping = in_cfg["column_mapping_by_name"]
        # Odwracamy słownik: {nazwa_w_csv: nazwa_wewnętrzna}
        rename_dict = {src_col: dst_col for dst_col, src_col in mapping.items()}
        new_df = raw_csv[list(rename_dict.keys())].rename(columns=rename_dict).copy()
    else:
        raw_csv = pd.read_csv(StringIO(response.text), sep=sep, header=None)
        mapping = in_cfg["column_mapping_by_index"]
        # Mapowanie po indeksie kolumny (0-based): {indeks_int: nazwa_wewnętrzna}
        rename_dict = {int(src_idx): dst_col for dst_col, src_idx in mapping.items()}
        new_df = raw_csv[list(rename_dict.keys())].rename(columns=rename_dict).copy()
    
    new_df["ID_LOS"] = new_df["ID_LOS"].astype("int64")
    dt_series = pd.to_datetime(new_df["DATE_LOS"], format=date_fmt)
    new_df["DATE_LOS"] = dt_series.dt.date
    new_df["YEAR_LOS"] = dt_series.dt.year.astype("int32")
    
    for col in ["L_1", "L_2", "L_3", "L_4", "L_5", "L_6"]:
        new_df[col] = new_df[col].astype("int32")
        
    new_df["L_GROUP"] = compute_l_group(new_df, cfg)

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
    parser = argparse.ArgumentParser(description="Pobiera nowe losowania z CSV i aktualizuje Parquet.")
    parser.add_argument("--url", type=str, default=None, help="Opcjonalny adres URL pliku CSV")
    args = parser.parse_args()
    added = update_from_csv(args.url)
    print(f"Zakończono. Dodano nowych rekordów: {added}")