import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# RANDOM GENERATOR
# ==========================================================

rng = np.random.default_rng(42)

# ==========================================================
# HISTORICAL DATA
# ==========================================================

history = pd.DataFrame({
    "Max Gust (kt)": [
        30, 32, 35, 33, 36, 38, 40, 42, 45, 44,
        46, 48, 50, 52, 55, 57, 56, 54, 52, 50,
        48, 47, 45, 43, 42, 40, 38, 36, 34, 32
    ],
    "Mean Wind Speed (kt)": [
        15, 16, 17, 18, 18, 19, 20, 21, 22, 22,
        23, 24, 25, 26, 27, 28, 27, 26, 25, 24,
        23, 22, 21, 20, 20, 19, 18, 17, 16, 15
    ],
    "Rain (mm)": [
        0, 2, 5, 0, 3, 6, 10, 12, 8, 5,
        4, 0, 0, 15, 20, 25, 18, 12, 6, 4,
        2, 0, 0, 3, 5, 7, 4, 2, 1, 0
    ],
    "Soil Moisture Deficit - Moderate (mm)": [
        25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ],
    "Mean Pressure (hPa)": [
        1018, 1016, 1015, 1014, 1012, 1010, 1008, 1005, 1002, 1000,
        998, 995, 992, 990, 988, 985, 987, 989, 991, 993,
        995, 997, 999, 1002, 1005, 1008, 1010, 1012, 1014, 1016
    ],
    "Max Temp (C)": [
        20, 21, 22, 22, 23, 24, 25, 26, 27, 27,
        28, 29, 30, 31, 32, 33, 32, 31, 30, 29,
        28, 27, 26, 25, 24, 23, 22, 21, 21, 20
    ],
    "Min Temp (C)": [
        10, 11, 12, 12, 13, 14, 15, 16, 17, 17,
        18, 19, 20, 21, 22, 23, 22, 21, 20, 19,
        18, 17, 16, 15, 14, 13, 12, 11, 11, 10
    ],
    "Evaporation (mm)": [
        2, 2.5, 3, 3, 3.5, 4, 4.5, 5, 5.5, 5.5,
        6, 6.5, 7, 7.5, 8, 8.5, 8, 7.5, 7, 6.5,
        6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2.5, 2
    ],
})

# ==========================================================
# PARAMETERS
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

THRESHOLDS = {
    "gust": 100,
    "mean_wind": 55,
    "rain_3d": 80,
    "rain_7d": 160,
    "soil": 30,
    "pressure_drop": 35,
    "temp_range": 18,
    "net_moisture": 120,
}

# ==========================================================
# SMOOTH NORMALISATION (NO HARD SPIKES)
# ==========================================================

def norm(x, limit, power=1.6):
    z = np.maximum(0, x / limit)
    return np.minimum(1, z ** power)


# ==========================================================
# Autoregressive mode (1) SIMULATION
# Ideal for modeling short-term correlations in time-series data, where the influence of past observations diminishes over time. 
# ==========================================================

def simulate_ar1(series, N, H, floor_zero=False):
    mu = series.mean()
    sigma = series.std(ddof=1)
    phi = np.clip(np.corrcoef(series[1:], series[:-1])[0, 1], 0.3, 0.9)

    paths = np.zeros((N, H))
    for i in range(N):
        print("Path created....")
        paths[i, 0] = series.iloc[-1]
        for t in range(1, H):
            paths[i, t] = (
                phi * paths[i, t-1]
                + (1 - phi) * mu
                + sigma * rng.normal()
            )
        if floor_zero:
            paths[i] = np.clip(paths[i], 0, None)

    return paths

# ==========================================================
# MONTE CARLO SETTINGS
# ==========================================================

N = 8000  # Number of simulations
H = 7  # Number of days into future

print(f"Running Monte Carlo with N={N}, H={H}")

simulated = {
    col: simulate_ar1(history[col], N, H, floor_zero=(col in ["Rain (mm)", "Evaporation (mm)", "Mean Wind Speed (kt)"]))
    for col in history.columns
}

# ==========================================================
# DSI COMPUTATION (PATH EXTREME)
# Derived from DSI_calculator.py
# ==========================================================

def compute_dsi(sim):
    df = pd.DataFrame(sim)

    df["Rain_7d"] = df["Rain (mm)"].rolling(7, 1).sum()
    df["Rain_3d"] = df["Rain (mm)"].rolling(3, 1).sum()
    df["MeanWind_3d"] = df["Mean Wind Speed (kt)"].rolling(3, 1).mean()
    df["MaxGust_5d"] = df["Max Gust (kt)"].rolling(5, 1).mean()
    df["PressureDrop_2d"] = df["Mean Pressure (hPa)"].diff().abs().rolling(2, 1).mean()
    df["Soil_7d"] = df["Soil Moisture Deficit - Moderate (mm)"].rolling(7, 1).mean()
    df["TempRange_3d"] = (df["Max Temp (C)"] - df["Min Temp (C)"]).rolling(3, 1).mean()
    df["NetMoisture_5d"] = (df["Rain (mm)"] - df["Evaporation (mm)"]).rolling(5, 1).sum()

    dsi = (
        WEIGHTS["gust"] * norm(df["MaxGust_5d"], THRESHOLDS["gust"]) +
        WEIGHTS["mean_wind"] * norm(df["MeanWind_3d"], THRESHOLDS["mean_wind"]) +
        WEIGHTS["rain"] * (
            0.6 * norm(df["Rain_3d"], THRESHOLDS["rain_3d"]) +
            0.4 * norm(df["Rain_7d"], THRESHOLDS["rain_7d"])
        ) +
        WEIGHTS["soil"] * norm(df["Soil_7d"], THRESHOLDS["soil"]) +
        WEIGHTS["pressure"] * norm(df["PressureDrop_2d"], THRESHOLDS["pressure_drop"]) +
        WEIGHTS["temp"] * norm(df["TempRange_3d"], THRESHOLDS["temp_range"]) +
        WEIGHTS["evap"] * norm(df["NetMoisture_5d"], THRESHOLDS["net_moisture"])
    )

    return dsi.max()   

# ==========================================================
# DAMAGE MODEL
# ==========================================================

def estimate_damage_from_dsi(dsi, threshold=0.31, scale=120_000, exponent=2.2):
    if dsi < threshold:
        return 0.0
    return scale * (dsi - threshold) ** exponent

# ==========================================================
# RUN
# ==========================================================


dsi_list = []

for i in range(N):
    
    # Create dictionary for this simulation
    sim_i = {}
    
    for k in simulated:
        sim_i[k] = simulated[k][i]
    
    # Compute DSI for this simulation
    dsi_value = compute_dsi(sim_i)
    
    # Store result
    dsi_list.append(dsi_value)

# Convert to numpy array
dsi_mc = np.array(dsi_list)

damage_mc = np.array([estimate_damage_from_dsi(d) for d in dsi_mc])

# ==========================================================
# PLOTS
# ==========================================================

print("Results:")

threshold = 0.35

print(f"P(DSI ≥ {threshold}) = {np.mean(dsi_mc >= threshold):.3f}")
print(f"Expected damage (ha): {damage_mc.mean():.0f}")

plt.hist(dsi_mc, bins=60, density=True)
plt.axvline(threshold, color="red")
plt.title("Monte Carlo DSI Distribution")
plt.show()

plt.hist(damage_mc, bins=60)
plt.title("Monte Carlo Damage Distribution")
plt.show()
