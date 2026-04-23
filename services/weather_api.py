import os
import requests
from dataclasses import dataclass
from typing import Optional

# NASA month keys
MONTH_KEYS = [
    "JAN","FEB","MAR","APR","MAY","JUN",
    "JUL","AUG","SEP","OCT","NOV","DEC"
]


@dataclass
class WeatherData:
    temperature: float
    humidity: float
    rainfall: float
    source: str


class WeatherService:

    NASA_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    OW_URL   = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, openweather_api_key: Optional[str] = None):

        self.ow_key = openweather_api_key or os.getenv("OPENWEATHER_API_KEY")

        self.headers = {
            "User-Agent": "AgroWare/1.0 (Crop Prediction Service)"
        }

    # ------------------------------------------------------------------
    # PUBLIC METHOD
    # ------------------------------------------------------------------

    def get_weather(self, latitude: float, longitude: float, month: int) -> WeatherData:

        if month < 1 or month > 12:
            raise ValueError("Month must be between 1 and 12")

        try:
            return self._from_nasa(latitude, longitude, month)

        except Exception as nasa_error:

            print(f"[WeatherService] NASA POWER failed: {nasa_error}")

            if self.ow_key:

                try:
                    return self._from_openweather(latitude, longitude)

                except Exception as ow_error:

                    print(f"[WeatherService] OpenWeather also failed: {ow_error}")

            raise RuntimeError(
                "All weather sources failed. Check network or API keys."
            )

    # ------------------------------------------------------------------
    # NASA POWER
    # ------------------------------------------------------------------

    def _from_nasa(self, lat: float, lon: float, month: int) -> WeatherData:

        params = {
            "parameters": "T2M,RH2M,PRECTOTCORR",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "format": "JSON"
        }

        response = requests.get(
            self.NASA_URL,
            params=params,
            headers=self.headers,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        props = data["properties"]["parameter"]

        month_key = MONTH_KEYS[month - 1]

        temperature = props["T2M"][month_key]
        humidity = props["RH2M"][month_key]
        rainfall = props["PRECTOTCORR"][month_key]

        # NASA sometimes returns -999 for missing values
        if temperature == -999 or humidity == -999 or rainfall == -999:
            raise ValueError("NASA returned invalid weather data")

        rainfall = rainfall * self._days_in_month(month)

        return WeatherData(
            temperature=round(temperature,2),
            humidity=round(humidity,2),
            rainfall=round(rainfall,2),
            source="NASA POWER"
        )

    # ------------------------------------------------------------------
    # OPENWEATHER FALLBACK
    # ------------------------------------------------------------------

    def _from_openweather(self, lat: float, lon: float) -> WeatherData:

        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.ow_key,
            "units": "metric"
        }

        response = requests.get(
            self.OW_URL,
            params=params,
            headers=self.headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]

        rainfall = data.get("rain", {}).get("1h", 0.0)

        rainfall = rainfall * 24 * 30

        return WeatherData(
            temperature=round(temperature,2),
            humidity=round(humidity,2),
            rainfall=round(rainfall,2),
            source="OpenWeather (fallback)"
        )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _days_in_month(month: int):

        days = [31,28,31,30,31,30,31,31,30,31,30,31]

        return days[month - 1]


# ----------------------------------------------------------------------
# TEST RUN
# ----------------------------------------------------------------------

if __name__ == "__main__":

    import json

    service = WeatherService()

    weather = service.get_weather(
        latitude=31.1471,
        longitude=75.3412,
        month=10
    )

    print(json.dumps(weather.__dict__, indent=2))