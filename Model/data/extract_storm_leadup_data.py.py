import pandas as pd
from pathlib import Path

# ==========================
# USER SETTINGS
# ==========================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ATHENRY_daily_cleaned.csv"
OUTPUT_FILE = BASE_DIR / "ATHENRY_3months_leadup.csv"

# Storms and total damage (ha)
STORMS = {
    "2014-02-12": 8000,   # Darwin
    "2024-12-07": 2050,   # Darragh
    "2025-01-24": 24000,  # Éowyn
}

LEADUP_DAYS = 30  # 1 month

# Damage spread kernel
DAMAGE_KERNEL = {
    -1: 0.20,
     0: 0.45,
     1: 0.25,
     2: 0.10,
}

print("Extracting lead-up data and assigning damage...")

def main():
    df = pd.read_csv(INPUT_FILE)

    # Parse date
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    df.sort_values("Date", inplace=True)

    leadup_frames = []

    for storm_date_str, total_damage in STORMS.items():
        storm_date = pd.to_datetime(storm_date_str)
        start_date = storm_date - pd.Timedelta(days=LEADUP_DAYS)

        period_df = df[
            (df["Date"] >= start_date) &
            (df["Date"] <= storm_date + pd.Timedelta(days=2))
        ].copy()

        # Initialise damage column
        period_df["damage_ha"] = 0.0

        # Spread damage across days
        for offset, weight in DAMAGE_KERNEL.items():
            target_date = storm_date + pd.Timedelta(days=offset)
            period_df.loc[
                period_df["Date"] == target_date,
                "damage_ha"
            ] += total_damage * weight

        leadup_frames.append(period_df)

    # Combine all storms
    final_df = pd.concat(leadup_frames, ignore_index=True)

    # Remove duplicate dates (keep max damage if overlap ever occurs)
    final_df = (
        final_df
        .groupby("Date", as_index=False)
        .max()
    )

    # Replace all remaining NaNs with 0
    final_df.fillna(0, inplace=True)

    # Reformat date
    final_df["Date"] = final_df["Date"].dt.strftime("%d-%m-%Y")

    # Save
    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n"
    )

    print(f"✔ Lead-up dataset with damage created: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
