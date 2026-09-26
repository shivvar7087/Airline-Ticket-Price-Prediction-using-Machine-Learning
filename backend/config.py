"""
Configuration loader for Airline Ticket Price Prediction.
Loads environment variables from .env file with zero external dependencies.
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")

def load_env(path: str = ENV_PATH):
    """Load key-value pairs from .env into os.environ if not already set."""
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"Warning: Failed to load .env file from {path}: {e}")

# Load .env upon import
load_env()

# Application Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# Artifact Paths
_raw_model = os.getenv("MODEL_PATH", "models/flight_price_model.joblib")
MODEL_PATH = _raw_model if os.path.isabs(_raw_model) else os.path.join(BASE_DIR, _raw_model)

_raw_metrics = os.getenv("METRICS_PATH", "models/metrics.json")
METRICS_PATH = _raw_metrics if os.path.isabs(_raw_metrics) else os.path.join(BASE_DIR, _raw_metrics)

_raw_meta = os.getenv("METADATA_PATH", "models/feature_metadata.json")
METADATA_PATH = _raw_meta if os.path.isabs(_raw_meta) else os.path.join(BASE_DIR, _raw_meta)

_raw_dataset = os.getenv("DATASET_PATH", "dataset/data.xlsx")
DATASET_PATH = _raw_dataset if os.path.isabs(_raw_dataset) else os.path.join(BASE_DIR, _raw_dataset)

# Security & CORS
_raw_cors = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_cors.split(",")] if "," in _raw_cors else [_raw_cors]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

