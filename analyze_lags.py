import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/1_blast_furnace_data_first_dataset.xlsx"

SENSOR_SHEET = "data"
SI_SHEET = "Si"

TIME_COLUMN = "dt"

TARGET = "Si"


# Features we want to investigate first.
# These were among the more important features in Baseline 1.

FEATURES = [
    "Tp6",
    "Tp3",
    "Th",
    "Fb",
    "Pt",
    "H2",
    "Tp5",
    "R",
    "CO2",
    "dP"
]


# Hours before the Si measurement
LAGS = [
    0,
    1,
    2,
    3,
    4,
    6,
    8,
    12
]


# ============================================================
# LOAD DATA
# ============================================================

print("[INFO] Loading dataset...")

sensor_df = pd.read_excel(
    DATA_FILE,
    sheet_name=SENSOR_SHEET
)

si_df = pd.read_excel(
    DATA_FILE,
    sheet_name=SI_SHEET
)

print(
    f"[INFO] Sensor data: {sensor_df.shape}"
)

print(
    f"[INFO] Si data: {si_df.shape}"
)


# ============================================================
# PREPARE DATA
# ============================================================

sensor_df[TIME_COLUMN] = pd.to_datetime(
    sensor_df[TIME_COLUMN]
)

si_df[TIME_COLUMN] = pd.to_datetime(
    si_df[TIME_COLUMN]
)

sensor_df = sensor_df.sort_values(
    TIME_COLUMN
).reset_index(drop=True)

si_df = si_df.sort_values(
    TIME_COLUMN
).reset_index(drop=True)


# Make sure numerical columns really are numeric.

for feature in FEATURES:

    sensor_df[feature] = pd.to_numeric(
        sensor_df[feature],
        errors="coerce"
    )


si_df[TARGET] = pd.to_numeric(
    si_df[TARGET],
    errors="coerce"
)


# Remove invalid Si measurements.

si_df = si_df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# CALCULATE LAGGED CORRELATIONS
# ============================================================

results = []


for lag in LAGS:

    print(
        f"\n[INFO] Analyzing lag: {lag} hour(s)"
    )

    # --------------------------------------------------------
    # Shift sensor timestamps forward.
    #
    # Example:
    #
    # Original sensor time = 10:00
    # Lag = 4 hours
    #
    # Shifted time = 14:00
    #
    # Therefore a Si measurement around 14:00
    # can use the sensor state from 10:00.
    # --------------------------------------------------------

    sensor_lagged = sensor_df[
        [TIME_COLUMN] + FEATURES
    ].copy()

    sensor_lagged[TIME_COLUMN] = (
        sensor_lagged[TIME_COLUMN]
        + pd.Timedelta(hours=lag)
    )

    sensor_lagged = sensor_lagged.sort_values(
        TIME_COLUMN
    )

    # --------------------------------------------------------
    # Match each Si measurement with the latest sensor
    # observation corresponding to this lag.
    # --------------------------------------------------------

    merged = pd.merge_asof(
        si_df[
            [TIME_COLUMN, TARGET]
        ].sort_values(TIME_COLUMN),

        sensor_lagged.sort_values(TIME_COLUMN),

        on=TIME_COLUMN,

        direction="backward"
    )

    # --------------------------------------------------------
    # Calculate correlations
    # --------------------------------------------------------

    for feature in FEATURES:

        valid = merged[
            [feature, TARGET]
        ].dropna()

        if len(valid) < 10:

            pearson = np.nan
            spearman = np.nan

        else:

            pearson = valid[
                feature
            ].corr(
                valid[TARGET],
                method="pearson"
            )

            spearman = valid[
                feature
            ].corr(
                valid[TARGET],
                method="spearman"
            )

        results.append({

            "lag_hours": lag,

            "feature": feature,

            "pearson": pearson,

            "abs_pearson": (
                abs(pearson)
                if not pd.isna(pearson)
                else np.nan
            ),

            "spearman": spearman,

            "abs_spearman": (
                abs(spearman)
                if not pd.isna(spearman)
                else np.nan
            ),

            "samples": len(valid)
        })


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# DISPLAY PEARSON RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ABSOLUTE PEARSON CORRELATION")
print("=" * 70)

pearson_table = results_df.pivot(
    index="feature",
    columns="lag_hours",
    values="abs_pearson"
)

print(
    pearson_table.round(4).to_string()
)


# ============================================================
# DISPLAY SPEARMAN RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ABSOLUTE SPEARMAN CORRELATION")
print("=" * 70)

spearman_table = results_df.pivot(
    index="feature",
    columns="lag_hours",
    values="abs_spearman"
)

print(
    spearman_table.round(4).to_string()
)


# ============================================================
# BEST LAG FOR EACH FEATURE
# ============================================================

print("\n")
print("=" * 70)
print("BEST LAG FOR EACH FEATURE")
print("=" * 70)

for feature in FEATURES:

    subset = results_df[
        results_df["feature"] == feature
    ].copy()

    subset = subset.dropna(
        subset=["abs_pearson"]
    )

    if len(subset) == 0:
        continue

    best = subset.loc[
        subset["abs_pearson"].idxmax()
    ]

    print(
        f"{feature:>5}  ->  "
        f"{int(best['lag_hours']):>2} hours  "
        f"(correlation = "
        f"{best['pearson']:.4f})"
    )


# ============================================================
# BEST FEATURES AT EACH LAG
# ============================================================

print("\n")
print("=" * 70)
print("TOP FEATURES AT EACH LAG")
print("=" * 70)

for lag in LAGS:

    subset = results_df[
        results_df["lag_hours"] == lag
    ].copy()

    subset = subset.sort_values(
        "abs_pearson",
        ascending=False
    )

    print(
        f"\nLag {lag} hour(s):"
    )

    for _, row in subset.head(5).iterrows():

        print(
            f"  {row['feature']:>5} : "
            f"{row['pearson']:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = "lag_analysis_results.csv"

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print(
    f"[OK] Results saved to: "
    f"{OUTPUT_FILE}"
)