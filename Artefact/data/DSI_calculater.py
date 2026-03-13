# ==========================================================
# DSI Calculator
# ==========================================================
# This function aims to calculate the damage score index from historical weather inputs
# This function also computes extra weather reading

# This file was mainly used for testing, development and itterations

# ==========================================================
# IMPORTS
# ==========================================================
import pandas as pd
from pathlib import Path

# ==========================================================
# FILES
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ATHENRY_3months_leadup.csv"
OUTPUT_FILE = BASE_DIR / "ATHENRY_damage_model_output.csv"

# ==========================================================
# WEIGHTS (sum = 1.00)
# ==========================================================
WEIGHTS = {
    "gust": 0.45,
    "mean_wind": 0.25,
    "rain": 0.20,
    "soil": 0.05,
    "pressure": 0.03,
    "temp": 0.01,
    "evap": 0.01,
}

# ==========================================================
# NORMALISATION THRESHOLDS
# ==========================================================
THRESHOLDS = {
    "gust": 80,
    "mean_wind": 45,
    "wind_change": 20,
    "rain_3d": 60,
    "rain_7d": 120,
    "soil": 25,
    "pressure_drop": -25,
    "pressure_min": 960,
    "temp_range": 15,
    "net_moisture": 80,
}

# ==========================================================
# NORMALISATION FUNCTION 
# ==========================================================
def norm(x, limit):
    return (x / limit).clip(0, 1)

# Normalize x to a 0-1 range by dividing by limit, then clamp the result
# between 0 and 1 to handle values outside the expected range


# ==========================================================
# DAMAGE ESTIMATION FROM DSI
# ==========================================================


def estimate_damage_from_dsi(
    dsi,
    dsi_threshold=0.31,
    scale=120_000,
    exponent=2.2
):
    """
    Estimate forest damage (hectares) from a given DSI value.

    Parameters:
    - dsi (float): Damage Severity Index (0-1)
    - dsi_threshold (float): Minimum DSI at which damage begins
    - scale (float): Scaling factor for hectares
    - exponent (float): Controls non-linearity of damage growth

    Returns:
    - Estimated damage in hectares (float)
    """

    if dsi < dsi_threshold:
        return 0.0

    damage = scale * (dsi - dsi_threshold) ** exponent
    return round(damage, 1)



def main():

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df.sort_values("Date", inplace=True)


    # --------------------------
    # Rolling averages
    # --------------------------
    df["Rain_7d"] = (df["Rain (mm)"].rolling(7, min_periods=1).sum().round(3))
    df["Rain_3d"] = (df["Rain (mm)"].rolling(3, min_periods=1).sum().round(3))
    df["MeanWind_3d"] = df["Mean Wind Speed (kt)"].rolling(3, min_periods=1).mean().round(3)
    df["MaxGust_3d"] = df["Max Gust (kt)"].rolling(3, min_periods=1).max().round(3)
    df["MaxGust_5d_mean"] = df["Max Gust (kt)"].rolling(5, min_periods=1).mean().round(3)
    df["WindChange_2d"] = df["Mean Wind Speed (kt)"].diff().rolling(2, min_periods=1).max().round(3)
    df["Soil_7d"] = df["Soil Moisture Deficit - Moderate (mm)"].rolling(7, min_periods=1).mean().round(3)
    df["SoilTrend_5d"] = (-df["Soil Moisture Deficit - Moderate (mm)"].diff().rolling(5, min_periods=1).mean()).round(3)
    df["MinPressure_3d"] = df["Mean Pressure (hPa)"].rolling(3, min_periods=1).min().round(3)
    df["PressureDrop_2d"] = df["Mean Pressure (hPa)"].diff().rolling(2, min_periods=1).min().round(3)
    df["NetMoisture_5d"] = (df["Rain (mm)"] - df["Evaporation (mm)"]).rolling(5, min_periods=1).sum().round(3)
    df["TempRange_3d"] = (df["Max Temp (C)"] - df["Min Temp (C)"]).rolling(3, min_periods=1).mean().round(3)

    # --------------------------
    # Normalised scores
    # --------------------------
    df["GustScore"] = norm(df[["MaxGust_3d", "MaxGust_5d_mean"]].max(axis=1), THRESHOLDS["gust"])
    df["WindScore"] = (norm(df["MeanWind_3d"], THRESHOLDS["mean_wind"]) + 0.5 * norm(df["WindChange_2d"], THRESHOLDS["wind_change"])).clip(0, 1)
    df["RainScore"] = 0.6 * norm(df["Rain_3d"], THRESHOLDS["rain_3d"]) + 0.4 * norm(df["Rain_7d"], THRESHOLDS["rain_7d"])
    df["SoilScore"] = 0.5 * norm(df["Soil_7d"], THRESHOLDS["soil"]) + 0.5 * norm(df["SoilTrend_5d"], THRESHOLDS["soil"])
    df["PressureScore"] = pd.concat([norm(df["PressureDrop_2d"].abs(), abs(THRESHOLDS["pressure_drop"])), norm(THRESHOLDS["pressure_min"] - df["MinPressure_3d"], 40)], axis=1).max(axis=1)
    df["TempScore"] = norm(df["TempRange_3d"], THRESHOLDS["temp_range"])
    df["EvapScore"] = norm(df["NetMoisture_5d"], THRESHOLDS["net_moisture"])

    # --------------------------
    # Final DSI (0–1)
    # --------------------------
    df["DSI"] = (
        WEIGHTS["gust"] * df["GustScore"] +
        WEIGHTS["mean_wind"] * df["WindScore"] +
        WEIGHTS["rain"] * df["RainScore"] +
        WEIGHTS["soil"] * df["SoilScore"] +
        WEIGHTS["pressure"] * df["PressureScore"] +
        WEIGHTS["temp"] * df["TempScore"] +
        WEIGHTS["evap"] * df["EvapScore"]
    ).round(3)


    # Ensure damage_ha is the last column (for viewing purposes)
    if "damage_ha" in df.columns:
        cols = [c for c in df.columns if c != "damage_ha"] + ["damage_ha"]
        df = df[cols]

    df["estimated_damage_ha"] = df["DSI"].apply(estimate_damage_from_dsi)

    df.to_csv(OUTPUT_FILE, index=False)

    print("✔ FINAL forest damage model complete")
    print(f"✔ Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


# ==========================================================