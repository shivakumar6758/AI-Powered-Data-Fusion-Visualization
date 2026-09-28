from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import numpy as np

def run_kmeans(df, n_clusters=3):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        return df, None
    df_num = df[num_cols].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_num)
    df_result = df.copy()
    df_result.loc[df_num.index, 'Cluster'] = clusters
    return df_result, kmeans

def run_regression(df, target_col):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if target_col not in num_cols:
        return None, "Target column must be numeric for regression."
    
    df_clean = df[num_cols].dropna()
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    if X.empty:
        return None, "No numeric features available."
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    metrics = {
        'MSE': mean_squared_error(y_test, preds),
        'R2': r2_score(y_test, preds)
    }
    return model, metrics

def run_random_forest(df, target_col, task_type='regression'):
    df_clean = df.dropna(subset=[target_col])
    
    # For simplicity, only using numeric features
    X = df_clean.select_dtypes(include=[np.number])
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
        
    y = df_clean[target_col]
    
    if X.empty:
        return None, "No numeric features available."
        
    # Impute remaining missing values with mean
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task_type == 'regression':
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {'MSE': mean_squared_error(y_test, preds), 'R2': r2_score(y_test, preds)}
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {'Accuracy': accuracy_score(y_test, preds)}
        
    return model, metrics
