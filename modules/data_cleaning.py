import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

def handle_duplicates(df):
    return df.drop_duplicates()

def handle_missing_values(df, strategy="mean"):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    if strategy == "mean":
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == "median":
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == "drop":
        df_clean = df_clean.dropna()
        
    for col in cat_cols:
        mode_val = df_clean[col].mode()
        df_clean[col] = df_clean[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
        
    return df_clean

def detect_anomalies_zscore(df, threshold=3):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(zscore(df_clean[num_cols].dropna()))
    mask = (z_scores < threshold).all(axis=1)
    df_inliers = df_clean.loc[df_clean[num_cols].dropna().index[mask]]
    return df_inliers

def detect_anomalies_iqr(df):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    Q1 = df_clean[num_cols].quantile(0.25)
    Q3 = df_clean[num_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df_clean[num_cols] < (Q1 - 1.5 * IQR)) | (df_clean[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df_clean[mask]

def detect_anomalies_isolation_forest(df, contamination=0.05):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return df_clean
    df_num = df_clean[num_cols].dropna()
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df_num)
    df_inliers = df_clean.loc[df_num.index[preds == 1]]
    return df_inliers
