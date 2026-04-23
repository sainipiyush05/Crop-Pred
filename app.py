# ── Load environment variables first ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.feature_engineering import FeatureEngineeringService, FeatureVector
from prediction.predictor import CropPredictor
import traceback


# ── App lifecycle ────────────────────────────────────────────────────────────

feature_service: Optional[FeatureEngineeringService] = None
crop_predictor: Optional[CropPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise AgroWare services once at startup.
    """

    global feature_service, crop_predictor

    try:
        feature_service = FeatureEngineeringService(
            opencage_api_key=os.getenv("OPENCAGE_API_KEY"),
            openweather_api_key=os.getenv("OPENWEATHER_API_KEY"),
        )
        
        crop_predictor = CropPredictor()

        print("[AgroWare] Services initialised successfully.")

    except Exception as e:
        print(f"[AgroWare] Service initialisation failed: {e}")
        raise RuntimeError("Failed to start AgroWare services")

    yield

    print("[AgroWare] Shutting down services.")


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AgroWare Crop Prediction API",
    description="Collects environmental features from a location + month and returns crop recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS (for frontend connection later) ─────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ──────────────────────────────────────────────

class CropPredictionRequest(BaseModel):
    location: str = Field(
        ...,
        min_length=2,
        max_length=100,
        examples=["Punjab", "Amritsar, Punjab", "Nashik, Maharashtra"],
        description="State, district, or city name in India",
    )

    month: int = Field(
        ...,
        ge=1,
        le=12,
        examples=[10],
        description="Month of cultivation (1=Jan ... 12=Dec)",
    )

    @field_validator("location")
    @classmethod
    def clean_location(cls, v: str) -> str:
        return v.strip()


class FeatureVectorResponse(BaseModel):

    # Core ML features
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

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
    model_inference_input: list[float]
    message: str


class PredictionResult(BaseModel):
    crop: str
    confidence: float


class FinalCropPredictionResponse(BaseModel):
    success: bool
    location_resolved: str
    prediction: str
    confidence: float
    top_suggestions: list[PredictionResult]
    features: FeatureVectorResponse
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "ok",
        "service": "AgroWare Crop Prediction API",
        "version": "1.0"
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


@app.post(
    "/predict/features",
    response_model=CropPredictionResponse,
    tags=["Prediction"],
)
async def get_features(request: CropPredictionRequest):

    if feature_service is None:
        raise HTTPException(
            status_code=500,
            detail="FeatureEngineeringService not initialised"
        )

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

            model_inference_input=fv.to_model_input(),

            message=(
                f"Environmental features collected for "
                f"{fv.district}, {fv.state} in {fv.month_name} "
                f"({fv.season.capitalize()} season). Ready for crop prediction."
            ),
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=422,
            detail=f"Feature collection failed: {str(e)}"
        )


@app.post(
    "/predict/crop",
    response_model=FinalCropPredictionResponse,
    tags=["Prediction"],
)
async def predict_crop(request: CropPredictionRequest):
    """
    Collects environmental features for the location + month and predicts the best crop.
    """

    if feature_service is None or crop_predictor is None:
        raise HTTPException(
            status_code=500,
            detail="AgroWare services not initialised"
        )

    try:
        # 1. Build environmental features
        result = feature_service.build_features(
            location=request.location,
            month=request.month,
        )
        
        fv: FeatureVector = result.feature_vector
        features_list = fv.to_model_input()

        # 2. Perform ML inference
        prediction_output = crop_predictor.predict(features_list)

        return FinalCropPredictionResponse(
            success=True,
            location_resolved=f"{result.location.district}, {result.location.state}",
            prediction=prediction_output["prediction"],
            confidence=prediction_output["confidence"],
            top_suggestions=[
                PredictionResult(crop=s["crop"], confidence=s["confidence"])
                for s in prediction_output["top_suggestions"]
            ],
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
            message=(
                f"Prediction successful for {fv.district}, {fv.state}. "
                f"Recommended crop: {prediction_output['prediction']}."
            )
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[AgroWare] Prediction failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Local Development Runner ────────────────────────────────────────────────

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )