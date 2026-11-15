import os
import joblib
import numpy as np
import pandas as pd
from ingest import update_from_csv, RAW_FILE
from features import build_historical_features, get_future_row_in_memory

MODELS_DIR = "models"
PRED_LOG_FILE = "logs/predictions.parquet"

def run_pipeline(csv_url: str):
    os.makedirs("logs", exist_ok=True)
    added_count = update_from_csv(csv_url)
    print(f"Pobrano nowych losowań: {added_count}")

    raw_df = pd.read_parquet(RAW_FILE, engine="pyarrow")
    hist_df = build_historical_features(raw_df)
    
    models = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])
    if not models:
        print("Brak modeli w katalogu models/. Uruchom najpierw train.py.")
        return

    # 1. Weryfikacja wcześniejszych predykcji
    if os.path.exists(PRED_LOG_FILE):
        log_df = pd.read_parquet(PRED_LOG_FILE, engine="pyarrow")
        actual_map = dict(zip(hist_df["DATE_LOS"], hist_df["L_GROUP"]))
        
        for idx, row in log_df.iterrows():
            if pd.isna(row["actual_group"]) and row["target_date"] in actual_map:
                actual_val = actual_map[row["target_date"]]
                top3_list = row["top3_groups"].split(",")
                log_df.at[idx, "actual_group"] = actual_val
                log_df.at[idx, "is_correct_top1"] = int(row["predicted_group"] == actual_val)
                log_df.at[idx, "is_correct_top3"] = int(actual_val in top3_list)
    else:
        log_df = pd.DataFrame(columns=[
            "target_date", "model_version", "predicted_group", "confidence",
            "top3_groups", "actual_group", "is_correct_top1", "is_correct_top3"
        ])

    # 2. Predykcja dla kolejnej daty wyłącznie w RAM
    next_date, future_X = get_future_row_in_memory(raw_df)
    
    new_predictions = []
    for m_file in models:
        bundle = joblib.load(os.path.join(MODELS_DIR, m_file))
        m_ver = bundle["version"]
        
        already_predicted = (
            (log_df["target_date"] == next_date) & 
            (log_df["model_version"] == m_ver)
        ).any()
        
        if not already_predicted:
            model = bundle["model"]
            req_feats = bundle["selected_features"]
            
            probas = model.predict_proba(future_X[req_feats])
            classes = model.classes_
            top3_idx = np.argsort(probas)[::-1][:3]
            
            top1_group = classes[top3_idx]
            top1_conf = float(probas[top3_idx])
            top3_str = ",".join(classes[top3_idx])
            
            new_predictions.append({
                "target_date": next_date,
                "model_version": m_ver,
                "predicted_group": top1_group,
                "confidence": round(top1_conf, 4),
                "top3_groups": top3_str,
                "actual_group": None,
                "is_correct_top1": None,
                "is_correct_top3": None
            })

    if new_predictions:
        log_df = pd.concat([log_df, pd.DataFrame(new_predictions)], ignore_index=True)
        log_df.to_parquet(PRED_LOG_FILE, engine="pyarrow", index=False)

    latest_ver = joblib.load(os.path.join(MODELS_DIR, models[-1]))["version"]
    latest_entry = log_df[
        (log_df["target_date"] == next_date) & (log_df["model_version"] == latest_ver)
    ].iloc[-1]
    
    print(f"Data kolejnego losowania: {next_date}")
    print(f"Top-1 (Model {latest_ver}): {latest_entry['predicted_group']} ({latest_entry['confidence']:.1%})")
    print(f"Top-3 kandydaci: {latest_entry['top3_groups']}")

if __name__ == "__main__":
    run_pipeline("https://twoj-adres-do-pliku.csv")