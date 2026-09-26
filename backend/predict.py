"""
Inference module for Airline Ticket Price Prediction.
Loads saved ML model pipeline and provides clean prediction interface.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import joblib

from backend.config import MODEL_PATH, METRICS_PATH, METADATA_PATH

logger = logging.getLogger(__name__)

class FlightPricePredictor:
    def __init__(self):
        self.pipeline = None
        self.metrics = {}
        self.metadata = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained model pipeline and metadata."""
        if os.path.exists(MODEL_PATH):
            self.pipeline = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded flight price model pipeline.")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}. Training might be needed.")

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    @staticmethod
    def _parse_date(date_str: str):
        """Parse date in various common formats (YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY)."""
        date_str = str(date_str).strip()
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.day, dt.month, dt.weekday()
            except ValueError:
                continue
        # Fallback to today if unparseable
        now = datetime.now()
        return now.day, now.month, now.weekday()

    @staticmethod
    def _parse_time(time_str: str):
        """Parse 24h time in HH:MM format."""
        time_str = str(time_str).strip()
        try:
            dt = datetime.strptime(time_str, "%H:%M")
            return dt.hour, dt.minute
        except ValueError:
            return 12, 0

    @staticmethod
    def _calculate_duration(dep_hour: int, dep_min: int, arr_hour: int, arr_min: int, total_stops: int = 0) -> int:
        """Calculate estimated duration in minutes based on departure and arrival times."""
        dep_total = dep_hour * 60 + dep_min
        arr_total = arr_hour * 60 + arr_min

        diff = arr_total - dep_total
        if diff <= 0:
            # Flight arrives the next day
            diff += 24 * 60

        # Adjust for multiple stops if realistic duration would be larger
        if total_stops > 0 and diff < 120:
            diff += 24 * 60  # Long layover next-day arrival

        return max(diff, 45)

    def predict(
        self,
        airline: str,
        source: str,
        destination: str,
        date_of_journey: str,
        dep_time: str,
        arrival_time: str,
        total_stops: int = 0,
        additional_info: str = "No info",
        duration_mins: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate flight ticket price prediction."""
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded. Please ensure models/flight_price_model.joblib exists.")

        # Extract datetime attributes
        journey_day, journey_month, journey_weekday = self._parse_date(date_of_journey)
        dep_hour, dep_min = self._parse_time(dep_time)
        arr_hour, arr_min = self._parse_time(arrival_time)

        # Determine flight duration
        if duration_mins is None or duration_mins <= 0:
            duration_mins = self._calculate_duration(dep_hour, dep_min, arr_hour, arr_min, total_stops)

        # Standardize Additional Info
        info_cleaned = "No info" if additional_info in ("No info", "No Info", None, "") else additional_info

        # Build feature DataFrame matching trained pipeline
        input_data = {
            "Airline": [airline],
            "Source": [source],
            "Destination": [destination],
            "Additional_Info": [info_cleaned],
            "Total_Stops": [int(total_stops)],
            "Journey_day": [int(journey_day)],
            "Journey_month": [int(journey_month)],
            "Journey_weekday": [int(journey_weekday)],
            "Dep_hour": [int(dep_hour)],
            "Dep_min": [int(dep_min)],
            "Arrival_hour": [int(arr_hour)],
            "Arrival_min": [int(arr_min)],
            "Duration_mins": [int(duration_mins)]
        }

        df_input = pd.DataFrame(input_data)
        raw_pred = float(self.pipeline.predict(df_input)[0])

        # Logical floor (flights rarely cost less than ~₹1,750 base fare in dataset)
        predicted_price = max(round(raw_pred, 2), 1750.0)

        hours = duration_mins // 60
        mins = duration_mins % 60
        duration_formatted = f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

        return {
            "success": True,
            "predicted_price": predicted_price,
            "currency": "INR",
            "model_used": self.metrics.get("best_model_name", "Extra Trees"),
            "estimated_duration_formatted": duration_formatted,
            "duration_mins": duration_mins,
            "flight_summary": {
                "airline": airline,
                "source": source,
                "destination": destination,
                "date_of_journey": date_of_journey,
                "dep_time": dep_time,
                "arrival_time": arrival_time,
                "total_stops": total_stops,
                "additional_info": info_cleaned,
                "journey_day": journey_day,
                "journey_month": journey_month,
                "journey_weekday": journey_weekday
            }
        }

# Global singleton instance
predictor = FlightPricePredictor()

