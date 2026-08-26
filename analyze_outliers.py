import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/1_blast_furnace_data_first_dataset.xlsx"

TIME_COLUMN = "dt"
TARGET = "Si"

FEATURES = [
    "Fb",
    "Ph",
    "Pc",
    "Tc",
    "Fo",
    "dP",
    "dPu",
    "dPl",
    "Pt",
    "Th",
    "CO2",
    "H2",
    "Tt1",
    "Tt2",
    "Tt3",
    "Tt4",
    "Tp1",
    "Tp2",
    "Tp3",
    "Tp4",
    "Tp5",
    "Tp6",
    "Tp7",
    "Tp8",
    "Tp9",
    "Tp10",
    "R"
]


# ============================================================
# LOAD DATA
# ============================================================

print("[INFO] Loading dataset...")

sensor_df = pd.read_excel(
    DATA_FILE,
    sheet_name="data"
)

si_df = pd.read_excel(
    DATA_FILE,
    sheet_name="Si"
)


# ============================================================
# PREPROCESS
# ============================================================

sensor_df[TIME_COLUMN] = pd.to_datetime(
    sensor_df[TIME_COLUMN]
)

si_df[TIME_COLUMN] = pd.to_datetime(
    si_df[TIME_COLUMN]
)

si_df[TARGET] = pd.to_numeric(
    si_df[TARGET],
    errors="coerce"
)

for feature in FEATURES:

    sensor_df[feature] = pd.to_numeric(
        sensor_df[feature],
        errors="coerce"
    )


sensor_df = sensor_df.sort_values(
    TIME_COLUMN
).reset_index(drop=True)

si_df = si_df.sort_values(
    TIME_COLUMN
).reset_index(drop=True)


# ============================================================
# ALIGN SI WITH SENSOR DATA
# ============================================================

print("[INFO] Aligning Si and sensor data...")

dataset = pd.merge_asof(

    si_df[
        [TIME_COLUMN, TARGET]
    ].sort_values(TIME_COLUMN),

    sensor_df[
        [TIME_COLUMN] + FEATURES
    ].sort_values(TIME_COLUMN),

    on=TIME_COLUMN,

    direction="backward"
)


dataset = dataset.dropna(
    subset=[TARGET]
).copy()


print(
    f"[INFO] Final dataset: {dataset.shape}"
)


# ============================================================
# BASIC SI STATISTICS
# ============================================================

print("\n")
print("=" * 75)
print("SI BASIC STATISTICS")
print("=" * 75)

print(
    dataset[TARGET].describe()
)


# ============================================================
# EXTREME VALUE COUNTS
# ============================================================

print("\n")
print("=" * 75)
print("EXTREME SI VALUE COUNTS")
print("=" * 75)

thresholds = [
    0.8,
    1.0,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0
]

for threshold in thresholds:

    high_count = (
        dataset[TARGET] > threshold
    ).sum()

    low_count = (
        dataset[TARGET] < (1.0 - threshold)
        if threshold < 1.0
        else 0
    )

    print(
        f"Si > {threshold:<4} : "
        f"{high_count:5} rows"
    )


# ============================================================
# LOW SI VALUES
# ============================================================

print("\n")
print("=" * 75)
print("LOW SI VALUE COUNTS")
print("=" * 75)

low_thresholds = [
    0.4,
    0.3,
    0.2,
    0.1,
    0.05,
    0.01
]

for threshold in low_thresholds:

    count = (
        dataset[TARGET] < threshold
    ).sum()

    print(
        f"Si < {threshold:<4} : "
        f"{count:5} rows"
    )


# ============================================================
# DEFINE EXTREME OBSERVATIONS
# ============================================================

extreme = dataset[
    (dataset[TARGET] > 1.2) |
    (dataset[TARGET] < 0.3)
].copy()


print("\n")
print("=" * 75)
print("EXTREME OBSERVATIONS")
print("=" * 75)

print(
    f"Total extreme observations: "
    f"{len(extreme)}"
)

print(
    f"Percentage of dataset: "
    f"{len(extreme) / len(dataset) * 100:.3f}%"
)


# ============================================================
# TOP 30 HIGHEST SI VALUES
# ============================================================

print("\n")
print("=" * 75)
print("TOP 30 HIGHEST SI VALUES")
print("=" * 75)

highest = (
    dataset
    .sort_values(TARGET, ascending=False)
    .head(30)
)

print(
    highest[
        [TIME_COLUMN, TARGET] + FEATURES
    ].to_string(index=False)
)


# ============================================================
# LOWEST 30 SI VALUES
# ============================================================

print("\n")
print("=" * 75)
print("30 LOWEST SI VALUES")
print("=" * 75)

lowest = (
    dataset
    .sort_values(TARGET, ascending=True)
    .head(30)
)

print(
    lowest[
        [TIME_COLUMN, TARGET] + FEATURES
    ].to_string(index=False)
)


# ============================================================
# EXTREME SI SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("EXTREME SI FEATURE SUMMARY")
print("=" * 75)

important_features = [
    "Fb",
    "Pt",
    "dP",
    "Th",
    "CO2",
    "H2",
    "Tp3",
    "Tp5",
    "Tp6",
    "Tp8",
    "R"
]


normal = dataset[
    (dataset[TARGET] >= 0.3) &
    (dataset[TARGET] <= 1.0)
].copy()


summary = []

for feature in important_features:

    normal_mean = normal[feature].mean()
    extreme_mean = extreme[feature].mean()

    if abs(normal_mean) > 1e-12:

        difference = (
            (extreme_mean - normal_mean)
            / abs(normal_mean)
        ) * 100

    else:

        difference = np.nan

    summary.append({

        "feature": feature,

        "normal_mean": normal_mean,

        "extreme_mean": extreme_mean,

        "difference_%": difference
    })


summary_df = pd.DataFrame(summary)


print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# EXTREME EVENTS BY TIME
# ============================================================

print("\n")
print("=" * 75)
print("EXTREME EVENTS BY YEAR")
print("=" * 75)

extreme["year"] = (
    extreme[TIME_COLUMN].dt.year
)

year_counts = (
    extreme
    .groupby("year")
    .size()
)

print(
    year_counts.to_string()
)


# ============================================================
# EXTREME EVENTS BY SI RANGE
# ============================================================

print("\n")
print("=" * 75)
print("EXTREME SI RANGES")
print("=" * 75)

bins = [
    -np.inf,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    np.inf
]

labels = [
    "<0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.8",
    "0.8-1.0",
    "1.0-1.2",
    "1.2-1.5",
    "1.5-2.0",
    "2.0-2.5",
    "2.5-3.0",
    "3.0-3.5",
    "3.5-4.0",
    ">4.0"
]

dataset["Si_range"] = pd.cut(
    dataset[TARGET],
    bins=bins,
    labels=labels
)

print(
    dataset["Si_range"]
    .value_counts(sort=False)
    .to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

highest.to_csv(
    "baseline22_highest_si.csv",
    index=False
)

lowest.to_csv(
    "baseline22_lowest_si.csv",
    index=False
)

extreme.to_csv(
    "baseline22_extreme_observations.csv",
    index=False
)

summary_df.to_csv(
    "baseline22_extreme_feature_summary.csv",
    index=False
)


print("\n")
print("=" * 75)
print("[OK] Baseline 2.2 analysis completed.")
print("=" * 75)

print("\nGenerated files:")

print("  baseline22_highest_si.csv")
print("  baseline22_lowest_si.csv")
print("  baseline22_extreme_observations.csv")
print("  baseline22_extreme_feature_summary.csv")