"""
FastAPI application for Airline Ticket Price Prediction.
Serves prediction APIs and static web UI.
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import HOST, PORT, DEBUG, CORS_ORIGINS, LOG_LEVEL, BASE_DIR
from backend.schemas import FlightPredictionRequest, FlightPredictionResponse, ModelMetadataResponse
from backend.predict import predictor

# Reconfigure output encoding for console
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Airline Ticket Price Prediction API",
    description="Machine Learning REST API for predicting Indian domestic airline flight fares.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Mount static frontend files if directory exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", summary="Web UI Home")
def read_root():
    """Serve the web application frontend."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Airline Ticket Price Prediction API is live. Visit /docs for documentation."}

@app.get("/health", summary="Health Check")
def health_check():
    """Check API and model status."""
    return {
        "status": "healthy",
        "model_loaded": predictor.pipeline is not None,
        "champion_model": predictor.metrics.get("best_model_name", "Extra Trees"),
        "test_r2_score": predictor.metrics.get("test_r2_score", 0.9215)
    }

@app.get("/metadata", response_model=ModelMetadataResponse, summary="Get Model and Dataset Metadata")
def get_metadata():
    """Return all valid airlines, airports, stops, and benchmark performance metrics."""
    meta = predictor.metadata
    metrics = predictor.metrics

    if not meta:
        raise HTTPException(status_code=500, detail="Metadata not found. Please run train.py first.")

    return {
        "champion_model": metrics.get("best_model_name", "Extra Trees"),
        "metrics": metrics,
        "airlines": meta.get("airlines", []),
        "sources": meta.get("sources", []),
        "destinations": meta.get("destinations", []),
        "additional_info_options": meta.get("additional_info_options", []),
        "total_stops_options": meta.get("total_stops_options", [])
    }

@app.post("/predict", response_model=FlightPredictionResponse, summary="Predict Flight Ticket Price")
def predict_price(request: FlightPredictionRequest):
    """Predict ticket price based on flight itinerary."""
    try:
        result = predictor.predict(
            airline=request.airline,
            source=request.source,
            destination=request.destination,
            date_of_journey=request.date_of_journey,
            dep_time=request.dep_time,
            arrival_time=request.arrival_time,
            total_stops=request.total_stops,
            additional_info=request.additional_info or "No info",
            duration_mins=request.duration_mins
        )
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=DEBUG)

