"""
Airline Ticket Price Prediction - Machine Learning Training Pipeline
Dataset: dataset/data.xlsx
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

from backend.config import DATASET_PATH, BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, "models")

def parse_duration(d: str) -> int:
    """Parse flight duration string (e.g. '2h 50m', '19h', '45m') into total minutes."""
    if not isinstance(d, str):
        return 0
    h, m = 0, 0
    parts = d.strip().split()
    for part in parts:
        if 'h' in part:
            h = int(part.replace('h', ''))
        elif 'm' in part:
            m = int(part.replace('m', ''))
    return h * 60 + m

def map_stops(stop_str: str) -> int:
    """Convert stop string description into numeric count."""
    stop_map = {
        'non-stop': 0,
        '1 stop': 1,
        '2 stops': 2,
        '3 stops': 3,
        '4 stops': 4
    }
    return stop_map.get(str(stop_str).strip(), 0)

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load dataset from Excel file and clean missing / anomalous records."""
    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_excel(file_path)
    initial_len = len(df)
    logger.info(f"Initial shape: {df.shape}")

    # Drop missing values
    df.dropna(inplace=True)

    # Remove impossible / anomalous duration records (e.g., 5m flight with stops)
    df = df[df['Duration'] != '5m']
    logger.info(f"Cleaned {initial_len - len(df)} anomalous/missing rows. Remaining: {len(df)} rows.")

    # Standardize Additional_Info
    df['Additional_Info'] = df['Additional_Info'].replace({'No Info': 'No info'})

    # Feature Engineering
    # 1. Date of Journey
    journey_dt = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
    df['Journey_day'] = journey_dt.dt.day
    df['Journey_month'] = journey_dt.dt.month
    df['Journey_weekday'] = journey_dt.dt.dayofweek

    # 2. Departure Time
    dep_dt = pd.to_datetime(df['Dep_Time'], format='%H:%M')
    df['Dep_hour'] = dep_dt.dt.hour
    df['Dep_min'] = dep_dt.dt.minute

    # 3. Arrival Time
    arrival_raw = df['Arrival_Time'].str.split(' ').str[0]
    arrival_dt = pd.to_datetime(arrival_raw, format='%H:%M')
    df['Arrival_hour'] = arrival_dt.dt.hour
    df['Arrival_min'] = arrival_dt.dt.minute

    # 4. Duration in Minutes
    df['Duration_mins'] = df['Duration'].apply(parse_duration)

    # 5. Total Stops
    df['Total_Stops'] = df['Total_Stops'].apply(map_stops)

    return df

def train_and_evaluate():
    """Main training workflow."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_and_clean_data(DATASET_PATH)

    cat_cols = ['Airline', 'Source', 'Destination', 'Additional_Info']
    num_cols = ['Total_Stops', 'Journey_day', 'Journey_month', 'Journey_weekday', 
                'Dep_hour', 'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_mins']
    
    feature_cols = cat_cols + num_cols
    target_col = 'Price'

    X = df[feature_cols]
    y = df[target_col]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )

    # Candidate models for comparison
    candidates = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    }

    results = {}
    best_name = None
    best_score = -float('inf')
    best_pipeline = None

    logger.info("=== Benchmarking Candidate Models ===")
    for name, reg in candidates.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', reg)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))

        results[name] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 4)
        }
        logger.info(f"{name:<22} | R2: {r2:.4f} | MAE: INR {mae:.2f} | RMSE: INR {rmse:.2f} | MAPE: {mape*100:.2f}%")

        if r2 > best_score:
            best_score = r2
            best_name = name
            best_pipeline = pipe

    logger.info(f"Selected Champion Model: '{best_name}' with R2 = {best_score:.4f}")

    # Predictions with best model
    best_preds = best_pipeline.predict(X_test)
    final_metrics = {
        "best_model_name": best_name,
        "test_r2_score": round(float(r2_score(y_test, best_preds)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, best_preds)), 2),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, best_preds))), 2),
        "test_mape_percent": round(float(mean_absolute_percentage_error(y_test, best_preds)) * 100, 2),
        "all_models_comparison": results,
        "dataset_statistics": {
            "total_records": int(len(df)),
            "train_records": int(len(X_train)),
            "test_records": int(len(X_test)),
            "price_min": float(df['Price'].min()),
            "price_max": float(df['Price'].max()),
            "price_mean": round(float(df['Price'].mean()), 2),
            "price_median": float(df['Price'].median())
        }
    }

    # Save Model Pipeline
    model_path = os.path.join(MODELS_DIR, "flight_price_model.joblib")
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Saved model pipeline to: {model_path}")

    # Save Metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Extract Feature Metadata for Frontend & Inference
    feature_metadata = {
        "airlines": sorted(list(df['Airline'].unique())),
        "sources": sorted(list(df['Source'].unique())),
        "destinations": sorted(list(df['Destination'].unique())),
        "additional_info_options": sorted(list(df['Additional_Info'].unique())),
        "total_stops_options": [
            {"label": "Non-stop (0 stops)", "value": 0},
            {"label": "1 Stop", "value": 1},
            {"label": "2 Stops", "value": 2},
            {"label": "3 Stops", "value": 3},
            {"label": "4 Stops", "value": 4}
        ],
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_cols": feature_cols
    }

    metadata_path = os.path.join(MODELS_DIR, "feature_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=4)
    logger.info(f"Saved feature metadata to: {metadata_path}")

    # Generate Evaluation Charts
    try:
        generate_plots(best_pipeline, X_test, y_test, best_preds, best_name)
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    logger.info("Training and evaluation completed successfully!")
    return final_metrics

def generate_plots(pipeline, X_test, y_test, y_pred, model_name):
    """Generate and save visual diagnostic charts."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Actual vs Predicted Scatter
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.35, color='#1d4ed8', edgecolors='none', s=35)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    ax1.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', lw=2, label='Ideal 1:1 Line')
    ax1.set_title(f'Actual vs. Predicted Price ({model_name})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Actual Price (INR)', fontsize=11)
    ax1.set_ylabel('Predicted Price (INR)', fontsize=11)
    ax1.legend()

    # 2. Residual Distribution
    residuals = y_test - y_pred
    ax2 = axes[1]
    sns.histplot(residuals, bins=50, kde=True, ax=ax2, color='#059669')
    ax2.axvline(0, color='#dc2626', linestyle='--', lw=1.5)
    ax2.set_title('Residuals Error Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residual (Actual - Predicted) (INR)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)

    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "evaluation_plots.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved evaluation plots to: {plot_path}")

if __name__ == "__main__":
    train_and_evaluate()

