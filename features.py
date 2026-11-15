import json
from datetime import timedelta, date
import numpy as np
import pandas as pd
from ingest import compute_l_group

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
    days_ahead = min((d - wd) % 7 or 7 for d in [1, 3, 5])
    return last_date + timedelta(days=days_ahead)

def _apply_feature_definitions(df: pd.DataFrame, buckets: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Wylicza L_GROUP oraz pomocniczą tabelę samych koszyków (0..5) dla statystyk."""
    res = pd.DataFrame(index=df.index)
    group_codes = df["L_GROUP"].astype("category").cat.codes
    prev_group = df["L_GROUP"].shift(1)
    
    # Przygotowanie mapy rocznych zliczeń: ile razy dana grupa wystąpiła w każdym roku
    valid_hist = df.dropna(subset=["L_GROUP"])
    yearly_counts = valid_hist.groupby(["YEAR_LOS", "L_GROUP"]).size().unstack(fill_value=0)
    
    # Statystyki na koszykach
    b_sum = buckets.sum(axis=1)
    b_span = buckets.max(axis=1) - buckets.min(axis=1)
    no_data_mask = buckets.isna().any(axis=1)
    b_sum[no_data_mask] = np.nan
    b_span[no_data_mask] = np.nan
    stat_map = {"sum": b_sum, "span": b_span}

    for feat in cfg["features"]:
        fname = feat["name"]
        ftype = feat["type"]
        
        if ftype == "day_of_week":
            res[fname] = pd.to_datetime(df["DATE_LOS"]).dt.weekday.astype("int32")
            
        elif ftype == "lag":
            res[fname] = group_codes.shift(feat["lag"])
            
        elif ftype == "current_year_freq":
            # Narastająca liczba wystąpień grupy z kroku N-1 w bieżącym roku (bez wycieku przyszłości)
            cum_in_year = df.groupby(["YEAR_LOS", "L_GROUP"]).cumcount() + 1
            # Przesuwamy o 1 w dół, aby wiersz N widział stan po losowaniu N-1
            shifted_cum = cum_in_year.shift(1)
            # Jeśli wiersz N to pierwszy dzień nowego roku, licznik z poprzedniego roku zerujemy
            same_year = (df["YEAR_LOS"] == df["YEAR_LOS"].shift(1))
            res[fname] = np.where(same_year, shifted_cum, 0.0)
            res.loc[0, fname] = np.nan
            
        elif ftype == "prev_years_freq":
            # Suma wystąpień grupy (z kroku N-1) w K pełnych latach poprzedzających YEAR_LOS wiersza N
            k_years = feat.get("years_back", 3)
            vals = []
            for idx, row in df.iterrows():
                target_grp = prev_group.iloc[idx]
                curr_yr = row["YEAR_LOS"]
                if pd.isna(target_grp) or target_grp not in yearly_counts.columns:
                    vals.append(0.0 if idx > 0 else np.nan)
                    continue
                
                years_range = range(curr_yr - k_years, curr_yr)
                existing_yrs = [y for y in years_range if y in yearly_counts.index]
                total = yearly_counts.loc[existing_yrs, target_grp].sum() if existing_yrs else 0
                vals.append(float(total))
            res[fname] = vals
            
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
    # Odświeżamy L_GROUP i YEAR_LOS na wypadek zmian w config.json
    df["YEAR_LOS"] = pd.to_datetime(df["DATE_LOS"]).dt.year.astype("int32")
    df["L_GROUP"] = compute_l_group(df, cfg)
    
    cols = cfg["group_rule"]["columns"]
    div = cfg["group_rule"]["divisor"]
    buckets = np.floor(df[cols] / div + 0.5).astype("int32")
    
    feats = _apply_feature_definitions(df, buckets, cfg)
    full_df = pd.concat([df[["ID_LOS", "DATE_LOS", "YEAR_LOS", "L_GROUP"]], feats], axis=1)
    full_df.to_parquet(FEATURES_FILE, engine="pyarrow", index=False)
    return full_df

def get_future_row_in_memory(raw_df: pd.DataFrame) -> tuple[date, pd.DataFrame]:
    cfg = load_config()
    last_date = raw_df["DATE_LOS"].iloc[-1]
    next_date = get_next_draw_date(last_date)
    
    df_hist = raw_df.copy()
    df_hist["YEAR_LOS"] = pd.to_datetime(df_hist["DATE_LOS"]).dt.year.astype("int32")
    df_hist["L_GROUP"] = compute_l_group(df_hist, cfg)
    
    cols = cfg["group_rule"]["columns"]
    div = cfg["group_rule"]["divisor"]
    hist_buckets = np.floor(df_hist[cols] / div + 0.5).astype("int32")
    
    future_row = pd.DataFrame({
        "ID_LOS": [df_hist["ID_LOS"].iloc[-1] + 1],
        "DATE_LOS": [next_date],
        "YEAR_LOS": [np.int32(next_date.year)],
        "L_GROUP": [np.nan]
    })
    future_buckets = pd.DataFrame(np.nan, index=[0], columns=hist_buckets.columns)
    
    temp_df = pd.concat([df_hist[["ID_LOS", "DATE_LOS", "YEAR_LOS", "L_GROUP"]], future_row], ignore_index=True)
    temp_buckets = pd.concat([hist_buckets, future_buckets], ignore_index=True)
    
    feats = _apply_feature_definitions(temp_df, temp_buckets, cfg)
    return next_date, feats.iloc[[-1]]
