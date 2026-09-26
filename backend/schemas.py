"""
Pydantic data validation schemas for Airline Ticket Price Prediction API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class FlightPredictionRequest(BaseModel):
    airline: str = Field(..., example="IndiGo", description="Operating airline name")
    source: str = Field(..., example="Delhi", description="Departure airport / city")
    destination: str = Field(..., example="Cochin", description="Arrival airport / city")
    date_of_journey: str = Field(..., example="2026-06-15", description="Date in YYYY-MM-DD or DD/MM/YYYY format")
    dep_time: str = Field(..., example="10:30", description="Departure time in HH:MM format (24h)")
    arrival_time: str = Field(..., example="14:15", description="Arrival time in HH:MM format (24h)")
    total_stops: int = Field(default=0, ge=0, le=4, example=1, description="Number of flight layover stops")
    additional_info: Optional[str] = Field(default="No info", example="No info", description="Ticket class / baggage / meal note")
    duration_mins: Optional[int] = Field(default=None, description="Optional manual duration in minutes (auto-calculated if omitted)")

class FlightPredictionResponse(BaseModel):
    success: bool = True
    predicted_price: float = Field(..., description="Estimated ticket fare in INR")
    currency: str = "INR"
    model_used: str = Field(..., description="Champion ML model identifier")
    estimated_duration_formatted: str = Field(..., description="Formatted duration, e.g. '3h 45m'")
    duration_mins: int = Field(..., description="Total flight duration in minutes")
    flight_summary: Dict[str, Any] = Field(..., description="Summary of parsed flight attributes")

class ModelMetadataResponse(BaseModel):
    champion_model: str
    metrics: Dict[str, Any]
    airlines: List[str]
    sources: List[str]
    destinations: List[str]
    additional_info_options: List[str]
    total_stops_options: List[Dict[str, Any]]

