import os
import sys
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Make sure project root is on path when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.feature_engineering import FeatureEngineeringService, FeatureVector


# ── App lifecycle ─────────────────────────────────────────────────────────────

feature_service: Optional[FeatureEngineeringService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise services once at startup."""
    global feature_service
    feature_service = FeatureEngineeringService(
        opencage_api_key=os.getenv("OPENCAGE_API_KEY"),
        openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
    )
    print("[AgroWare] Services initialised.")
    yield
    print("[AgroWare] Shutting down.")


app = FastAPI(
    title="AgroWare Crop Prediction API",
    description="Collects environmental features from a location + month and returns crop recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class CropPredictionRequest(BaseModel):
    location: str = Field(
        ...,
        min_length=2,
        max_length=100,
        examples=["Punjab", "Amritsar, Punjab", "Nashik, Maharashtra"],
        description="State, district, or city name in India.",
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        examples=[10],
        description="Month of cultivation as an integer (1 = January … 12 = December).",
    )

    @field_validator("location")
    @classmethod
    def strip_location(cls, v: str) -> str:
        return v.strip()


class FeatureVectorResponse(BaseModel):
    """The 7 ML model features plus context metadata."""
    # Core ML features
    N: float = Field(description="Nitrogen (kg/ha)")
    P: float = Field(description="Phosphorus (kg/ha)")
    K: float = Field(description="Potassium (kg/ha)")
    temperature: float = Field(description="Monthly avg temperature (°C)")
    humidity: float = Field(description="Relative humidity (%)")
    ph: float = Field(description="Soil pH")
    rainfall: float = Field(description="Monthly rainfall (mm)")

    # Context
    state: str
    district: str
    month: int
    month_name: str
    season: str
    latitude: float
    longitude: float

    # Metadata
    weather_source: str
    soil_source: str


class CropPredictionResponse(BaseModel):
    success: bool
    location_resolved: str
    features: FeatureVectorResponse
    model_input: list[float] = Field(
        description="The exact 7-value list passed to the ML model [N, P, K, temp, humidity, pH, rainfall]"
    )
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "AgroWare Crop Prediction API v1.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.post(
    "/predict/features",
    response_model=CropPredictionResponse,
    tags=["Prediction"],
    summary="Collect environmental features for a location and month",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Location not found or API failure"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_features(request: CropPredictionRequest):
    """
    **Step 1 of the prediction pipeline.**

    Takes a location name and cultivation month, then automatically:
    1. Geocodes the location to lat/lng
    2. Fetches monthly weather from NASA POWER
    3. Fetches soil data (N, P, K, pH) from SoilGrids
    4. Assembles the 7-feature vector for the ML model

    The returned `model_input` array is what gets passed directly to the trained model.
    """
    try:
        result = feature_service.build_features(
            location=request.location,
            month=request.month,
        )
        fv: FeatureVector = result.feature_vector

        return CropPredictionResponse(
            success=True,
            location_resolved=f"{result.location.district}, {result.location.state}",
            features=FeatureVectorResponse(
                N=fv.N,
                P=fv.P,
                K=fv.K,
                temperature=fv.temperature,
                humidity=fv.humidity,
                ph=fv.ph,
                rainfall=fv.rainfall,
                state=fv.state,
                district=fv.district,
                month=fv.month,
                month_name=fv.month_name,
                season=fv.season,
                latitude=fv.latitude,
                longitude=fv.longitude,
                weather_source=fv.weather_source,
                soil_source=fv.soil_source,
            ),
            model_input=fv.to_model_input(),
            message=(
                f"Features collected for {fv.district}, {fv.state} "
                f"in {fv.month_name} ({fv.season.capitalize()} season). "
                f"Ready for model inference."
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature collection failed: {str(e)}"
        )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)