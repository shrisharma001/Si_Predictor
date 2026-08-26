import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/1_blast_furnace_data_first_dataset.xlsx"

SENSOR_SHEET = "data"
SI_SHEET = "Si"

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


# Features that were important in Baseline 2
IMPORTANT_FEATURES = [
    "Tp6",
    "Tp6_lag1",
    "Th",
    "Pt_lag1",
    "Fb_lag1",
    "Fb",
    "Tp8",
    "Tp3",
    "R_lag8",
    "Th_lag6",
    "Pt",
    "Tp3_lag1",
    "CO2_lag12"
]


# Same temporal features used in Baseline 2
LAG_FEATURES = {
    1: [
        "Fb",
        "Pt",
        "dP",
        "Tp3",
        "Tp6",
        "Tp5",
        "CO2",
        "H2"
    ],

    6: [
        "Th"
    ],

    8: [
        "R"
    ],

    12: [
        "CO2"
    ]
}


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


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for feature in FEATURES:

    sensor_df[feature] = pd.to_numeric(
        sensor_df[feature],
        errors="coerce"
    )


si_df[TARGET] = pd.to_numeric(
    si_df[TARGET],
    errors="coerce"
)

si_df = si_df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# CREATE BASE DATASET
# ============================================================

print("[INFO] Creating aligned dataset...")

merged = pd.merge_asof(

    si_df[
        [TIME_COLUMN, TARGET]
    ].sort_values(TIME_COLUMN),

    sensor_df[
        [TIME_COLUMN] + FEATURES
    ].sort_values(TIME_COLUMN),

    on=TIME_COLUMN,

    direction="backward"
)


# ============================================================
# CREATE LAG FEATURES
# ============================================================

print("[INFO] Creating lag features...")

for lag_hours, feature_list in LAG_FEATURES.items():

    lagged_sensor = sensor_df[
        [TIME_COLUMN] + feature_list
    ].copy()

    lagged_sensor[TIME_COLUMN] = (
        lagged_sensor[TIME_COLUMN]
        + pd.Timedelta(hours=lag_hours)
    )

    lagged_sensor = lagged_sensor.rename(
        columns={
            feature: f"{feature}_lag{lag_hours}"
            for feature in feature_list
        }
    )

    merged = pd.merge_asof(

        merged.sort_values(TIME_COLUMN),

        lagged_sensor.sort_values(TIME_COLUMN),

        on=TIME_COLUMN,

        direction="backward"
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

feature_columns = FEATURES.copy()

for lag_hours, feature_list in LAG_FEATURES.items():

    for feature in feature_list:

        feature_columns.append(
            f"{feature}_lag{lag_hours}"
        )


dataset = merged[
    [TIME_COLUMN, TARGET] + feature_columns
].dropna().copy()


print(
    f"[INFO] Final dataset: {dataset.shape}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(dataset)

train_end = int(n * 0.70)
val_end = int(n * 0.85)


train = dataset.iloc[:train_end].copy()

validation = dataset.iloc[
    train_end:val_end
].copy()

test = dataset.iloc[
    val_end:
].copy()


print("\n[INFO] Dataset split")

print(
    f"Training   : {len(train)}"
)

print(
    f"Validation : {len(validation)}"
)

print(
    f"Testing    : {len(test)}"
)


# ============================================================
# TIME RANGES
# ============================================================

print("\n")
print("=" * 75)
print("TIME RANGE OF EACH SPLIT")
print("=" * 75)

for name, df in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:

    print(
        f"{name:12} : "
        f"{df[TIME_COLUMN].min()} "
        f"→ "
        f"{df[TIME_COLUMN].max()}"
    )


# ============================================================
# SI DISTRIBUTION
# ============================================================

print("\n")
print("=" * 75)
print("SI DISTRIBUTION")
print("=" * 75)

for name, df in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:

    si = df[TARGET]

    print(f"\n{name}")

    print(
        f"Mean   : {si.mean():.6f}"
    )

    print(
        f"Median : {si.median():.6f}"
    )

    print(
        f"Std    : {si.std():.6f}"
    )

    print(
        f"Min    : {si.min():.6f}"
    )

    print(
        f"Max    : {si.max():.6f}"
    )


# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 75)
print("FEATURE DISTRIBUTION SHIFT")
print("=" * 75)

distribution_results = []


for feature in IMPORTANT_FEATURES:

    if feature not in dataset.columns:
        continue

    train_mean = train[feature].mean()
    val_mean = validation[feature].mean()
    test_mean = test[feature].mean()

    train_std = train[feature].std()
    val_std = validation[feature].std()
    test_std = test[feature].std()

    # Percentage difference between test and train mean
    if abs(train_mean) > 1e-12:

        mean_shift = (
            (test_mean - train_mean)
            / abs(train_mean)
        ) * 100

    else:

        mean_shift = np.nan

    distribution_results.append({

        "feature": feature,

        "train_mean": train_mean,
        "validation_mean": val_mean,
        "test_mean": test_mean,

        "train_std": train_std,
        "validation_std": val_std,
        "test_std": test_std,

        "test_vs_train_mean_shift_%":
            mean_shift
    })


distribution_df = pd.DataFrame(
    distribution_results
)


print(
    distribution_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SI QUANTILES
# ============================================================

print("\n")
print("=" * 75)
print("SI QUANTILES")
print("=" * 75)

quantiles = [
    0.01,
    0.05,
    0.25,
    0.50,
    0.75,
    0.95,
    0.99
]

for name, df in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:

    print(f"\n{name}")

    print(
        df[TARGET]
        .quantile(quantiles)
        .to_string()
    )


# ============================================================
# CORRELATION OF IMPORTANT FEATURES WITH SI
# ============================================================

print("\n")
print("=" * 75)
print("FEATURE vs SI CORRELATION BY SPLIT")
print("=" * 75)

correlation_rows = []

for feature in IMPORTANT_FEATURES:

    if feature not in dataset.columns:
        continue

    correlation_rows.append({

        "feature": feature,

        "train_corr":
            train[feature].corr(
                train[TARGET]
            ),

        "validation_corr":
            validation[feature].corr(
                validation[TARGET]
            ),

        "test_corr":
            test[feature].corr(
                test[TARGET]
            )
    })


correlation_df = pd.DataFrame(
    correlation_rows
)


print(
    correlation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SI OVER TIME
# ============================================================

print("\n[INFO] Creating Si over time plot...")

plt.figure(figsize=(14, 6))

plt.plot(
    dataset[TIME_COLUMN],
    dataset[TARGET],
    linewidth=0.8
)

plt.axvline(
    train.iloc[-1][TIME_COLUMN],
    linestyle="--",
    label="Train / Validation"
)

plt.axvline(
    validation.iloc[-1][TIME_COLUMN],
    linestyle="--",
    label="Validation / Test"
)

plt.xlabel("Time")
plt.ylabel("Si")
plt.title("Hot Metal Silicon (Si) Over Time")

plt.legend()

plt.tight_layout()

plt.savefig(
    "si_over_time_baseline21.png",
    dpi=150
)

plt.close()


# ============================================================
# SI DISTRIBUTION COMPARISON
# ============================================================

print("[INFO] Creating Si distribution plot...")

plt.figure(figsize=(10, 6))

plt.hist(
    train[TARGET],
    bins=50,
    alpha=0.5,
    label="Training"
)

plt.hist(
    validation[TARGET],
    bins=50,
    alpha=0.5,
    label="Validation"
)

plt.hist(
    test[TARGET],
    bins=50,
    alpha=0.5,
    label="Testing"
)

plt.xlabel("Si")
plt.ylabel("Frequency")

plt.title(
    "Si Distribution Across Dataset Splits"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "si_distribution_baseline21.png",
    dpi=150
)

plt.close()


# ============================================================
# SAVE ANALYSIS
# ============================================================

distribution_df.to_csv(
    "baseline21_feature_distribution.csv",
    index=False
)

correlation_df.to_csv(
    "baseline21_feature_correlations.csv",
    index=False
)


print("\n")
print("=" * 75)
print("[OK] Baseline 2.1 analysis completed.")
print("=" * 75)

print(
    "\nGenerated files:"
)

print(
    "  si_over_time_baseline21.png"
)

print(
    "  si_distribution_baseline21.png"
)

print(
    "  baseline21_feature_distribution.csv"
)

print(
    "  baseline21_feature_correlations.csv"
)