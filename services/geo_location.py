import os
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocationData:
    latitude: float
    longitude: float
    state: str
    district: str
    country: str
    source: str


class GeoLocationService:
    """
    Converts a location name (state, district, or address)
    into geographic coordinates using the OpenCage Geocoding API.
    """

    BASE_URL = "https://api.opencagedata.com/geocode/v1/json"

    def __init__(self, api_key: Optional[str] = None):

        self.api_key = api_key or os.getenv("OPENCAGE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenCage API key not found. Set OPENCAGE_API_KEY environment variable."
            )

        # Prevent API blocking
        self.headers = {
            "User-Agent": "AgroWare/1.0 (Crop Prediction Service)"
        }

    # ------------------------------------------------------------------
    # PUBLIC METHOD
    # ------------------------------------------------------------------

    def get_coordinates(self, location: str) -> LocationData:

        if not location:
            raise ValueError("Location cannot be empty")

        params = {
            "q": f"{location}, India",
            "key": self.api_key,
            "limit": 1,
            "language": "en",
            "countrycode": "in",
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=10
            )

            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Geocoding API request failed: {e}")

        data = response.json()

        if not data.get("results"):
            raise ValueError(
                f"Location '{location}' not found. Try a more specific place name."
            )

        result = data["results"][0]

        geometry = result.get("geometry", {})
        components = result.get("components", {})

        latitude = geometry.get("lat")
        longitude = geometry.get("lng")

        if latitude is None or longitude is None:
            raise RuntimeError("Coordinates not returned by OpenCage API")

        # Extract state
        state = (
            components.get("state")
            or components.get("region")
            or components.get("state_district")
            or "Unknown"
        )

        # Extract district / city
        district = (
            components.get("county")
            or components.get("district")
            or components.get("city")
            or components.get("town")
            or components.get("village")
            or "Unknown"
        )

        country = components.get("country", "India")

        return LocationData(
            latitude=float(latitude),
            longitude=float(longitude),
            state=state.strip(),
            district=district.strip(),
            country=country.strip(),
            source="OpenCage"
        )


# ------------------------------------------------------------------
# QUICK TEST
# ------------------------------------------------------------------

if __name__ == "__main__":

    import json

    service = GeoLocationService()

    location = service.get_coordinates("Punjab")

    print(json.dumps(location.__dict__, indent=2))