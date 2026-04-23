from dataclasses import dataclass, asdict
from typing import Optional

from services.geo_location import GeoLocationService, LocationData
from services.weather_api import WeatherService, WeatherData
from services.soil_api import SoilService, SoilData


# ── Season mapping (spec §6) ─────────────────────────────────────────────────

SEASON_MAP = {
    1:  "rabi",    # January
    2:  "rabi",    # February
    3:  "zaid",    # March
    4:  "zaid",    # April
    5:  "zaid",    # May
    6:  "kharif",  # June
    7:  "kharif",  # July
    8:  "kharif",  # August
    9:  "kharif",  # September
    10: "kharif",  # October
    11: "rabi",    # November
    12: "rabi",    # December
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


@dataclass
class FeatureVector:
    """
    The exact feature vector passed to the ML model.
    Matches the training dataset columns: N, P, K, temperature, humidity, ph, rainfall.
    Extra fields (lat, lng, season, state) are used by the filtering layer — not the model.
    """
    # ── Core ML features (must match training data column order) ────────────
    N: float            # Nitrogen    (kg/ha)
    P: float            # Phosphorus  (kg/ha)
    K: float            # Potassium   (kg/ha)
    temperature: float  # °C
    humidity: float     # %
    ph: float           # soil pH
    rainfall: float     # mm/month

    # ── Context features (used by filtering layer, not the model) ───────────
    latitude: float
    longitude: float
    state: str
    district: str
    month: int
    month_name: str
    season: str         # kharif / rabi / zaid

    # ── Metadata ────────────────────────────────────────────────────────────
    weather_source: str
    soil_source: str

    def to_model_input(self) -> list:
        """
        Returns only the 7 features the ML model was trained on,
        in the exact column order from the training CSV.
        """
        return [self.N, self.P, self.K, self.temperature, self.humidity, self.ph, self.rainfall]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureEngineeringResult:
    feature_vector: FeatureVector
    location: LocationData
    weather: WeatherData
    soil: SoilData


class FeatureEngineeringService:
    """
    Orchestrates geo_location, weather, and soil services to produce
    a complete FeatureVector ready for the ML model.

    Usage:
        service = FeatureEngineeringService(geo_key="...", ow_key="...")
        result  = service.build_features(location="Punjab", month=10)
        model_input = result.feature_vector.to_model_input()
    """

    def __init__(
        self,
        opencage_api_key: Optional[str] = None,
        openweather_api_key: Optional[str] = None,
    ):
        self.geo_service     = GeoLocationService(api_key=opencage_api_key)
        self.weather_service = WeatherService(openweather_api_key=openweather_api_key)
        self.soil_service    = SoilService()

    def build_features(self, location: str, month: int) -> FeatureEngineeringResult:
        """
        Main entry point. Takes user inputs and returns a complete FeatureVector.

        Args:
            location : state/district name or address string  e.g. "Punjab"
            month    : cultivation month as integer 1–12      e.g. 10

        Returns:
            FeatureEngineeringResult with feature_vector, location, weather, soil
        """
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be between 1 and 12, got {month}.")

        # Step 1 — Geocode location
        print(f"[FeatureEngineering] Geocoding '{location}'...")
        location_data = self.geo_service.get_coordinates(location)
        print(f"  → {location_data.latitude:.4f}, {location_data.longitude:.4f} ({location_data.state})")

        # Step 2 — Fetch weather
        print(f"[FeatureEngineering] Fetching weather for month {month}...")
        weather_data = self.weather_service.get_weather(
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            month=month,
        )
        print(f"  → {weather_data.temperature}°C, {weather_data.humidity}% RH, {weather_data.rainfall}mm [{weather_data.source}]")

        # Step 3 — Fetch soil
        print(f"[FeatureEngineering] Fetching soil data...")
        soil_data = self.soil_service.get_soil(
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            state=location_data.state,
        )
        print(f"  → N={soil_data.nitrogen}, P={soil_data.phosphorus}, K={soil_data.potassium}, pH={soil_data.ph} [{soil_data.source}]")

        # Step 4 — Assemble feature vector
        feature_vector = FeatureVector(
            # ML model features
            N=soil_data.nitrogen,
            P=soil_data.phosphorus,
            K=soil_data.potassium,
            temperature=weather_data.temperature,
            humidity=weather_data.humidity,
            ph=soil_data.ph,
            rainfall=weather_data.rainfall,

            # Context for filtering layer
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            state=location_data.state,
            district=location_data.district,
            month=month,
            month_name=MONTH_NAMES[month],
            season=SEASON_MAP[month],

            # Metadata
            weather_source=weather_data.source,
            soil_source=soil_data.source,
        )

        return FeatureEngineeringResult(
            feature_vector=feature_vector,
            location=location_data,
            weather=weather_data,
            soil=soil_data,
        )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    service = FeatureEngineeringService()
    result  = service.build_features(location="Punjab", month=10)

    print("\n── Feature Vector ──")
    print(json.dumps(result.feature_vector.to_dict(), indent=2))

    print("\n── Model Input (7 features) ──")
    print(result.feature_vector.to_model_input())