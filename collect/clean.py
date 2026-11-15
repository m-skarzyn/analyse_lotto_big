# process/clean.py
import pandas as pd
import numpy as np

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    return df
