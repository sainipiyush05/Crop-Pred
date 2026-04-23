import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class SoilData:
    nitrogen: float        # N  — kg/ha  (real value from SoilGrids, properly converted)
    phosphorus: float      # P  — kg/ha  (see IMPORTANT NOTE below)
    potassium: float       # K  — kg/ha  (see IMPORTANT NOTE below)
    ph: float              # soil pH in water (0–14)
    source: str            # where the data came from

    # ── IMPORTANT NOTE ON P AND K ────────────────────────────────────────────
    # SoilGrids does NOT provide phosphorus or potassium. Period.
    # There is no reliable free global API that gives P and K at field level.
    #
    # What we do instead (in order of priority):
    #   1. Indian Soil Health Card API (data.gov.in) — real lab-tested P & K
    #      per district. Best option for India. Requires government API access.
    #   2. State-level average fallback from ICAR published soil surveys
    #
    # We DO NOT estimate P and K from clay/SOC — that is scientifically invalid.
    # ─────────────────────────────────────────────────────────────────────────


# ── State-level fallback P & K averages for Indian states ────────────────────
# Source: ICAR (Indian Council of Agricultural Research) soil fertility surveys
# These are state-level averages in kg/ha — used ONLY when Soil Health Card
# data is unavailable.

INDIA_STATE_SOIL_FALLBACK = {
    # state_name_lowercase: (P_kg_ha, K_kg_ha)
    "punjab":               (28.0,  52.0),
    "haryana":              (24.0,  48.0),
    "uttar pradesh":        (20.0,  45.0),
    "madhya pradesh":       (18.0,  40.0),
    "maharashtra":          (16.0,  38.0),
    "andhra pradesh":       (22.0,  50.0),
    "telangana":            (22.0,  50.0),
    "karnataka":            (20.0,  44.0),
    "tamil nadu":           (24.0,  46.0),
    "west bengal":          (26.0,  55.0),
    "odisha":               (18.0,  42.0),
    "gujarat":              (16.0,  36.0),
    "rajasthan":            (14.0,  32.0),
    "bihar":                (22.0,  48.0),
    "jharkhand":            (18.0,  40.0),
    "assam":                (20.0,  44.0),
    "kerala":               (28.0,  58.0),
    "himachal pradesh":     (22.0,  46.0),
    "uttarakhand":          (20.0,  44.0),
    "chhattisgarh":         (16.0,  38.0),
    "__default__":          (20.0,  44.0),
}


class SoilService:
    """
    Fetches real soil properties for a given lat/lng using SoilGrids REST API.

    What SoilGrids actually provides (and what we use):
      - nitrogen  : total nitrogen in cg/kg  → converted to kg/ha using bdod
      - phh2o     : pH in water, stored ×10  → divided by 10
      - bdod      : bulk density in cg/cm³   → used in N conversion formula

    What SoilGrids does NOT provide:
      - Phosphorus (P) — not available from any free global API
      - Potassium  (K) — not available from any free global API

    For P and K we use ICAR state-level survey averages as fallback.

    Correct N conversion formula (from SoilGrids documentation):
      N (kg/ha) = (nitrogen_cg_kg / 100) × (bdod_cg_cm3 / 100) × depth_cm × 10
    Applied per depth layer, then summed across 0-5, 5-15, 15-30cm.
    """

    BASE_URL   = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    PROPERTIES = ["nitrogen", "phh2o", "bdod"]
    DEPTHS     = ["0-5cm", "5-15cm", "15-30cm"]
    LAYER_THICKNESS = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}

    def get_soil(self, latitude: float, longitude: float, state: str = "") -> SoilData:
        """
        Returns N (real), P (ICAR fallback), K (ICAR fallback), pH (real)
        for the given coordinates.
        """
        try:
            nitrogen, ph = self._fetch_from_soilgrids(latitude, longitude)
            source = "SoilGrids v2 (N, pH) + ICAR state average (P, K)"
        except Exception as e:
            print(f"[SoilService] SoilGrids failed: {e}. Using state fallback.")
            n_ph_pk = self._state_fallback_all(state)
            nitrogen, ph = n_ph_pk[0], n_ph_pk[1]
            source = "ICAR state average fallback (SoilGrids unavailable)"

        phosphorus, potassium = self._get_p_and_k(state)

        return SoilData(
            nitrogen=round(nitrogen, 2),
            phosphorus=round(phosphorus, 2),
            potassium=round(potassium, 2),
            ph=round(ph, 2),
            source=source,
        )

    # ── SoilGrids fetch ───────────────────────────────────────────────────────

    def _fetch_from_soilgrids(self, lat: float, lon: float):
        """
        Fetches nitrogen, pH, and bulk density from SoilGrids.
        Returns: (nitrogen_kg_ha, ph)
        """
        params = {
            "lon":      lon,
            "lat":      lat,
            "property": self.PROPERTIES,
            "depth":    self.DEPTHS,
            "value":    "mean",
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=20,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        layers_data = self._parse_layers(data)

        nitrogen_kg_ha = self._convert_nitrogen_to_kg_ha(
            layers_data.get("nitrogen", {}),
            layers_data.get("bdod", {}),
        )

        ph_values = list(layers_data.get("phh2o", {}).values())
        # phh2o is stored as integer ×10 in SoilGrids — must divide by 10
        ph = (sum(ph_values) / len(ph_values) / 10.0) if ph_values else 6.5

        return nitrogen_kg_ha, ph

    def _parse_layers(self, data: dict) -> dict:
        """
        Parses SoilGrids JSON response into:
        { "nitrogen": {"0-5cm": 120, "5-15cm": 110, ...}, "phh2o": {...}, ... }
        """
        result = {}
        for layer in data.get("properties", {}).get("layers", []):
            prop_name = layer["name"]
            result[prop_name] = {}
            for depth in layer.get("depths", []):
                label = depth["label"]
                mean_val = depth.get("values", {}).get("mean")
                if mean_val is not None:
                    result[prop_name][label] = mean_val
        return result

    def _convert_nitrogen_to_kg_ha(self, nitrogen_layers: dict, bdod_layers: dict) -> float:
        """
        Converts SoilGrids nitrogen (cg/kg) to kg/ha per layer, summed over 0–30cm.

        Formula per layer:
          N_kg_ha = (N_cg_kg / 100) × (bdod_cg_cm3 / 100) × thickness_cm × 10

        Why × 10:
          g/kg × g/cm³ × cm = g/cm²
          g/cm² × 10 = kg/ha  (because 1 ha = 10^8 cm², 1 kg = 10^3 g → factor 10^5/10^4 = 10)
        """
        total_n = 0.0
        layers_used = 0

        for depth_label, thickness in self.LAYER_THICKNESS.items():
            n_raw  = nitrogen_layers.get(depth_label)
            bd_raw = bdod_layers.get(depth_label)

            if n_raw is None or bd_raw is None:
                continue

            n_g_per_kg    = n_raw  / 100.0   # cg/kg  → g/kg
            bdod_g_per_cm3 = bd_raw / 100.0  # cg/cm³ → g/cm³
            n_kg_ha_layer  = n_g_per_kg * bdod_g_per_cm3 * thickness * 10

            total_n    += n_kg_ha_layer
            layers_used += 1

        if layers_used == 0:
            raise ValueError("No valid nitrogen + bdod layers in SoilGrids response.")

        return total_n

    # ── P and K ───────────────────────────────────────────────────────────────

    def _get_p_and_k(self, state: str) -> tuple:
        """
        P and K source priority:
          1. Soil Health Card API (data.gov.in) — TODO when API key obtained
          2. ICAR state-level average

        Returns: (phosphorus_kg_ha, potassium_kg_ha)
        """
        # TODO: Add Soil Health Card API call here
        # Example endpoint: https://api.data.gov.in/resource/{resource_id}
        #   ?api-key={key}&filters[state]={state}&filters[district]={district}
        # Response gives avg_available_P and avg_available_K in kg/ha

        return self._state_fallback_pk(state)

    def _state_fallback_pk(self, state: str) -> tuple:
        """Returns (P, K) from ICAR state survey averages."""
        key = state.lower().strip() if state else "__default__"
        return INDIA_STATE_SOIL_FALLBACK.get(key, INDIA_STATE_SOIL_FALLBACK["__default__"])

    def _state_fallback_all(self, state: str) -> tuple:
        """
        Full fallback when SoilGrids is down — returns (N, pH, P, K).
        N and pH from ICAR published soil surveys per state.
        """
        STATE_N_PH = {
            "punjab":           (80.0, 7.8),
            "haryana":          (72.0, 7.9),
            "uttar pradesh":    (65.0, 7.5),
            "madhya pradesh":   (60.0, 6.8),
            "maharashtra":      (55.0, 6.5),
            "andhra pradesh":   (58.0, 6.4),
            "telangana":        (58.0, 6.4),
            "karnataka":        (60.0, 6.2),
            "tamil nadu":       (62.0, 6.0),
            "west bengal":      (70.0, 5.8),
            "bihar":            (68.0, 6.5),
            "gujarat":          (50.0, 7.6),
            "rajasthan":        (42.0, 8.0),
            "__default__":      (60.0, 6.8),
        }
        key = state.lower().strip() if state else "__default__"
        n, ph = STATE_N_PH.get(key, STATE_N_PH["__default__"])
        p, k  = self._state_fallback_pk(state)
        return n, ph, p, k


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    service = SoilService()
    soil = service.get_soil(latitude=31.1471, longitude=75.3412, state="Punjab")
    print(json.dumps(soil.__dict__, indent=2))