import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ATHENRY_daily.csv"
OUTPUT_FILE = BASE_DIR / "ATHENRY_daily_cleaned.csv"

# ==========================
# RAW COLUMN STRUCTURE
# (from Met Éireann daily file)
# ==========================
RAW_COLS = [
    "date",
    "ind_maxtp", "maxtp",
    "ind_mintp", "mintp",
    "igmin", "gmin",
    "ind_rain", "rain",
    "cbl",
    "wdsp",
    "ind_hm", "hm",
    "ind_ddhm", "ddhm",
    "ind_hg", "hg",
    "soil",
    "pe",
    "evap",
    "smd_wd",
    "smd_md",
    "smd_pd",
    "glorad",
]

# ==========================
# COLUMNS TO KEEP + RENAME
# ==========================
KEEP_COLS = {
    "date": "Date",
    "rain": "Rain (mm)",
    "maxtp": "Max Temp (C)",
    "mintp": "Min Temp (C)",
    "wdsp": "Mean Wind Speed (kt)",
    "hm": "Max 10-min Wind (kt)",
    "ddhm": "Wind Dir @ Max Wind (deg)",
    "hg": "Max Gust (kt)",
    "evap": "Evaporation (mm)",
    "smd_md": "Soil Moisture Deficit - Moderate (mm)",
    "cbl": "Mean Pressure (hPa)",
}

print("Cleaning daily data...")

def find_header_line(path):
    """Find the header line starting with 'date,'"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("date,"):
                return i
    raise ValueError("CSV header not found")

def main():
    header_idx = find_header_line(INPUT_FILE)

    # Read CSV (skip metadata)
    df = pd.read_csv(
        INPUT_FILE,
        skiprows=header_idx + 1,
        header=None,
        names=RAW_COLS,
        na_values=["", " ", "  "],
        engine="python",
    ).dropna(how="all")

    # Parse and reformat date → dd-mm-yyyy
    df["date"] = pd.to_datetime(
        df["date"],
        format="%d-%b-%Y",
        errors="coerce"
    ).dt.strftime("%d-%m-%Y")

    # Keep only requested columns
    df = df[list(KEEP_COLS.keys())]

    # Rename columns nicely
    df.rename(columns=KEEP_COLS, inplace=True)

    # Force numeric columns
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save cleaned file
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print(f"✔ Cleaned daily file written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
