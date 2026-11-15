import os
import joblib
from datetime import datetime
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from features import build_historical_features, load_config

RAW_FILE = "data/raw_events.parquet"
MODELS_DIR = "models"

def run_training(force: bool = False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    cfg = load_config()
    raw_df = pd.read_parquet(RAW_FILE, engine="pyarrow")
    df = build_historical_features(raw_df).dropna().reset_index(drop=True)
    
    all_feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    
    existing_models = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")])
    if existing_models and not force:
        last_bundle = joblib.load(os.path.join(MODELS_DIR, existing_models[-1]))
        if (last_bundle["available_features"] == all_feat_cols and 
            last_bundle["group_rule"] == cfg["group_rule"]):
            print("Konfiguracja cech bez zmian. Pomijam trenowanie.")
            return

    # Podział czasowy 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # W zbiorze walidacyjnym oceniamy tylko klasy znane z części treningowej
    known_classes = set(train_df["L_GROUP"].unique())
    val_df_safe = val_df[val_df["L_GROUP"].isin(known_classes)]

    X_train, y_train = train_df[all_feat_cols], train_df["L_GROUP"]
    X_val, y_val = val_df_safe[all_feat_cols], val_df_safe["L_GROUP"]

    base_model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    base_model.fit(X_train, y_train)

    perm = permutation_importance(base_model, X_val, y_val, n_repeats=5, random_state=42)
    selected_feats = [
        col for col, imp in zip(all_feat_cols, perm.importances_mean) if imp > 0.0005
    ]
    if not selected_feats:
        selected_feats = all_feat_cols

    # Finalny model trenuje się na 100% historii (w tym na wszystkich rzadkich klasach)
    final_model = LGBMClassifier(n_estimators=150, random_state=42, verbose=-1)
    final_model.fit(df[selected_feats], df["L_GROUP"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.joblib")
    
    bundle = {
        "version": timestamp,
        "model": final_model,
        "selected_features": selected_feats,
        "available_features": all_feat_cols,
        "group_rule": cfg["group_rule"]
    }
    joblib.dump(bundle, model_path)
    print(f"Zapisano nowy model: {model_path} (liczba klas: {df['L_GROUP'].nunique()})")
    print(f"Wybrane cechy ({len(selected_feats)}/{len(all_feat_cols)}): {selected_feats}")

if __name__ == "__main__":
    run_training()