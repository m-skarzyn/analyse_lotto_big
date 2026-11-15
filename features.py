import json
from datetime import timedelta, date
import numpy as np
import pandas as pd

CONFIG_FILE = "config.json"
FEATURES_FILE = "data/features_cache.parquet"

def load_config() -> dict:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def get_next_draw_date(last_date: date) -> date:
    """Wyznacza kolejną datę losowania (Wtorek=1, Czwartek=3, Sobota=5)."""
    wd = last_date.weekday()
    if wd in (1, 3):
        return last_date + timedelta(days=2)
    elif wd == 5:
        return last_date + timedelta(days=3)
    days_ahead = min((d - wd) % 7 or 7 for d in)
    return last_date + timedelta(days=days_ahead)

def compute_l_group_and_buckets(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Wylicza L_GROUP oraz pomocniczą tabelę samych koszyków (0..5) dla statystyk."""
    cols = cfg["group_rule"]["columns"]
    div = cfg["group_rule"]["divisor"]
    sep = cfg["group_rule"]["separator"]
    
    # Zaokrąglanie matematyczne (połówki w górę)
    buckets = np.floor(df[cols] / div + 0.5).astype("int32")
    l_group = buckets.astype(str).agg(sep.join, axis=1)
    return l_group, buckets

def _apply_feature_definitions(df: pd.DataFrame, buckets: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    res = pd.DataFrame(index=df.index)
    group_codes = df["L_GROUP"].astype("category").cat.codes
    
    # Pomocnicze serie na koszykach (wyliczane tylko dla wierszy, gdzie liczby istnieją)
    b_sum = buckets.sum(axis=1)
    b_span = buckets.max(axis=1) - buckets.min(axis=1)
    b_unique = buckets.nunique(axis=1)
    # Tam gdzie brak liczb (wiersz przyszłości), ustawiamy NaN przed przesunięciem
    no_data_mask = buckets.isna().any(axis=1)
    b_sum[no_data_mask] = np.nan
    b_span[no_data_mask] = np.nan
    b_unique[no_data_mask] = np.nan
    stat_map = {"sum": b_sum, "span": b_span, "unique": b_unique}

    for feat in cfg["features"]:
        fname = feat["name"]
        ftype = feat["type"]
        
        if ftype == "day_of_week":
            res[fname] = pd.to_datetime(df["DATE_LOS"]).dt.weekday.astype("int32")
            
        elif ftype == "lag":
            res[fname] = group_codes.shift(feat["lag"])
            
        elif ftype == "rolling_freq":
            win = feat["window"]
            prev_g = group_codes.shift(1)
            res[fname] = [
                (group_codes.iloc[max(0, i - win):i] == prev_g.iloc[i]).sum() 
                if i > 0 else np.nan
                for i in range(len(df))
            ]
            
        elif ftype == "draws_since_last":
            last_seen = {}
            gaps = []
            for i, g in enumerate(df["L_GROUP"]):
                if i == 0 or pd.isna(df["L_GROUP"].iloc[i - 1]):
                    gaps.append(np.nan)
                else:
                    prev_val = df["L_GROUP"].iloc[i - 1]
                    gaps.append(i - 1 - last_seen.get(prev_val, -1))
                if pd.notna(g):
                    last_seen[g] = i
            res[fname] = gaps
            
        elif ftype == "bucket_stat":
            s = stat_map[feat["stat"]]
            res[fname] = s.shift(feat.get("lag", 1))
            
        elif ftype == "bucket_rolling_mean":
            s = stat_map[feat["stat"]]
            res[fname] = s.rolling(feat["window"]).mean().shift(1)

    return res

def build_historical_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    df = raw_df.copy()
    df["L_GROUP"], buckets = compute_l_group_and_buckets(df, cfg)
    
    feats = _apply_feature_definitions(df, buckets, cfg)
    full_df = pd.concat([df[["ID_LOS", "DATE_LOS", "L_GROUP"]], feats], axis=1)
    full_df.to_parquet(FEATURES_FILE, engine="pyarrow", index=False)
    return full_df

def get_future_row_in_memory(raw_df: pd.DataFrame) -> tuple[date, pd.DataFrame]:
    cfg = load_config()
    last_date = raw_df["DATE_LOS"].iloc[-1]
    next_date = get_next_draw_date(last_date)
    
    df_hist = raw_df.copy()
    df_hist["L_GROUP"], hist_buckets = compute_l_group_and_buckets(df_hist, cfg)
    
    future_row = pd.DataFrame({
        "ID_LOS": [df_hist["ID_LOS"].iloc[-1] + 1],
        "DATE_LOS": [next_date],
        "L_GROUP": [np.nan]
    })
    future_buckets = pd.DataFrame(np.nan, index=, columns=hist_buckets.columns)
    
    temp_df = pd.concat([df_hist[["ID_LOS", "DATE_LOS", "L_GROUP"]], future_row], ignore_index=True)
    temp_buckets = pd.concat([hist_buckets, future_buckets], ignore_index=True)
    
    feats = _apply_feature_definitions(temp_df, temp_buckets, cfg)
    return next_date, feats.iloc[[-1]]
