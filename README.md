# ✈️ Airline Ticket Price Prediction using Machine Learning

An end-to-end Machine Learning web application and REST API to predict Indian domestic airline ticket prices based on flight routes, carriers, departure/arrival schedules, stops, and dates.

---

## 📊 Model Performance Highlights

We benchmarked 5 regression models on 10,681 flight records:

| Model | $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Extra Trees (Champion)** | **92.15%** | **₹ 583.24** | **₹ 1,287.20** | **6.76%** |
| **Random Forest** | 91.15% | ₹ 621.23 | ₹ 1,366.85 | 7.13% |
| **HistGradientBoosting** | 89.01% | ₹ 894.97 | ₹ 1,523.35 | 10.49% |
| **Ridge Regression** | 70.72% | ₹ 1,736.77 | ₹ 2,486.10 | 20.97% |
| **Linear Regression** | 70.32% | ₹ 1,739.76 | ₹ 2,503.18 | 20.96% |

---

## 📁 Project Architecture

```
Airline Ticket Price Prediction using Machine Learning/
├── dataset/
│   └── data.xlsx               # Source flight records (10,683 rows)
├── models/
│   ├── flight_price_model.joblib # Serialized champion pipeline (Extra Trees)
│   ├── metrics.json            # Model evaluation scores & statistics
│   ├── feature_metadata.json   # Categorical options for airlines, cities, stops
│   └── evaluation_plots.png    # Residual & error diagnostics chart
├── backend/
│   ├── app.py                  # FastAPI web server and REST API endpoints
│   ├── config.py               # Zero-dependency .env environment loader
│   ├── predict.py              # Modular inference engine and preprocessor
│   ├── schemas.py              # Pydantic request and response models
│   └── reuirements.py          # Backend requirements reference
├── frontend/
│   ├── index.html              # Modern responsive flight booking UI
│   ├── style.css               # Clean styling, cards, badge components
│   └── script.js               # Dynamic form interactions and API integration
├── notebooks/
│   └── airline_price_prediction.ipynb # Interactive EDA and experimentation
├── tests/
│   └── test_pipeline.py        # Automated test suite
├── .env                        # Active environment configuration
├── .env.example                # Template for environment variables
├── train.py                    # Reproducible training & benchmarking script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Environment Configuration

The application reads configuration from [`.env`](file:///c:/Users/dell/OneDrive/Desktop/Naviotech/Airline%20Ticket%20Price%20Prediction%20using%20Machine%20Learning/.env). A template is provided in [`.env.example`](file:///c:/Users/dell/OneDrive/Desktop/Naviotech/Airline%20Ticket%20Price%20Prediction%20using%20Machine%20Learning/.env.example).

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Server bind IP address |
| `PORT` | `8000` | Server listen port |
| `ENVIRONMENT` | `development` | Environment mode (`development` / `production`) |
| `DEBUG` | `True` | Auto-reload server in debug mode |
| `MODEL_PATH` | `models/flight_price_model.joblib` | Path to trained model pipeline |
| `METRICS_PATH` | `models/metrics.json` | Path to model metrics JSON |
| `METADATA_PATH` | `models/feature_metadata.json` | Path to feature metadata |
| `DATASET_PATH` | `dataset/data.xlsx` | Path to dataset Excel file |
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | Application log level (`INFO`, `DEBUG`, etc.) |

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Benchmark the Model (Optional)
The pre-trained champion pipeline is already saved in `models/`. To retrain from scratch:
```bash
python train.py
```

### 3. Launch the Backend API & Web Application
Start the FastAPI server:
```bash
uvicorn backend.app:app --reload --port 8000
```
- **Web UI**: Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative**: You can also open `frontend/index.html` directly in any web browser.

### 4. Run Automated Tests
```bash
python tests/test_pipeline.py
```

---

## 🔌 API Reference

### 1. Health Check
`GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "champion_model": "Extra Trees",
  "test_r2_score": 0.9215
}
```

### 2. Get Metadata & Options
`GET /metadata`
Returns lists of available airlines, departure cities, arrival cities, and layover stops.

### 3. Predict Flight Price
`POST /predict`

**Request Body:**
```json
{
  "airline": "IndiGo",
  "source": "Delhi",
  "destination": "Cochin",
  "date_of_journey": "2026-06-15",
  "dep_time": "10:30",
  "arrival_time": "14:15",
  "total_stops": 1,
  "additional_info": "No info"
}
```

**Response:**
```json
{
  "success": true,
  "predicted_price": 5869.90,
  "currency": "INR",
  "model_used": "Extra Trees",
  "estimated_duration_formatted": "3h 45m",
  "duration_mins": 225,
  "flight_summary": {
    "airline": "IndiGo",
    "source": "Delhi",
    "destination": "Cochin",
    "date_of_journey": "2026-06-15",
    "dep_time": "10:30",
    "arrival_time": "14:15",
    "total_stops": 1,
    "additional_info": "No info",
    "journey_day": 15,
    "journey_month": 6,
    "journey_weekday": 0
  }
}
```

---

## 🛠️ Feature Engineering & Data Pipeline

1. **Date Extraction**: Converts `Date_of_Journey` into `Journey_day`, `Journey_month`, and `Journey_weekday`.
2. **Time Parsing**: Extracts departure hour and minute, arrival hour and minute.
3. **Flight Duration**: Parses strings like `2h 50m` or `19h` into total elapsed minutes.
4. **Stops Mapping**: Encodes `non-stop`, `1 stop`, `2 stops`, etc. into numeric integers `(0, 1, 2, 3, 4)`.
5. **Categorical Handling**: Uses `OneHotEncoder(handle_unknown='ignore')` for `Airline`, `Source`, `Destination`, and `Additional_Info` to prevent data leakage and handle unseen categories smoothly.

