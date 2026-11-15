import os
import pandas as pd
import requests
from io import StringIO

RAW_FILE = "data/raw_events.parquet"

def update_from_csv(url: str) -> int:
    os.makedirs("data", exist_ok=True)
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    
    new_df = pd.read_csv(StringIO(response.text))
    new_df["ID_LOS"] = new_df["ID_LOS"].astype("int64")
    new_df["DATE_LOS"] = pd.to_datetime(new_df["DATE_LOS"]).dt.date
    for col in ["L_1", "L_2", "L_3", "L_4", "L_5", "L_6"]:
        new_df[col] = new_df[col].astype("int32")

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
    updated_df.to_parquet(RAW_FILE, engine="pyarrow", index=False)
    return len(truly_new)
