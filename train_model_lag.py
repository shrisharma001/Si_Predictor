import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ============================================================
# CONFIG
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


# Selected temporal features based on lag analysis.
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

print(
    f"[INFO] Sensor data shape: "
    f"{sensor_df.shape}"
)

print(
    f"[INFO] Si data shape: "
    f"{si_df.shape}"
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


# Convert features to numeric.

for col in FEATURES:

    sensor_df[col] = pd.to_numeric(
        sensor_df[col],
        errors="coerce"
    )


si_df[TARGET] = pd.to_numeric(
    si_df[TARGET],
    errors="coerce"
)


# Remove missing target values.

si_df = si_df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# CREATE BASE DATASET
# ============================================================

print("[INFO] Creating base sensor/Si alignment...")


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

    print(
        f"       Adding {lag_hours}-hour lag..."
    )

    lagged_sensor = sensor_df[
        [TIME_COLUMN] + feature_list
    ].copy()


    # Shift timestamp forward so that:
    #
    # sensor(t - lag)
    #
    # becomes available at timestamp t.

    lagged_sensor[TIME_COLUMN] = (

        lagged_sensor[TIME_COLUMN]
        + pd.Timedelta(hours=lag_hours)

    )


    lagged_sensor = lagged_sensor.sort_values(
        TIME_COLUMN
    )


    lagged_sensor = lagged_sensor.rename(

        columns={
            feature:
            f"{feature}_lag{lag_hours}"

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
# PREPARE ML DATA
# ============================================================

feature_columns = FEATURES.copy()


for lag_hours, feature_list in LAG_FEATURES.items():

    for feature in feature_list:

        feature_columns.append(
            f"{feature}_lag{lag_hours}"
        )


dataset = merged[
    [TIME_COLUMN, TARGET] + feature_columns
].copy()


print(
    f"[INFO] Total features: "
    f"{len(feature_columns)}"
)


# Remove missing values.

before = len(dataset)

dataset = dataset.dropna()

after = len(dataset)

print(
    f"[INFO] Removed "
    f"{before - after} rows containing "
    f"missing values."
)

print(
    f"[INFO] Final dataset size: "
    f"{dataset.shape}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(dataset)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)


train = dataset.iloc[
    :train_end
]

validation = dataset.iloc[
    train_end:val_end
]

test = dataset.iloc[
    val_end:
]


X_train = train[feature_columns]
y_train = train[TARGET]


X_val = validation[feature_columns]
y_val = validation[TARGET]


X_test = test[feature_columns]
y_test = test[TARGET]


print("\n[INFO] Dataset split:")

print(
    f"Training   : {len(train)} rows"
)

print(
    f"Validation : {len(validation)} rows"
)

print(
    f"Testing    : {len(test)} rows"
)


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\n[INFO] Training XGBoost...")


model = xgb.XGBRegressor(

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1

)


model.fit(
    X_train,
    y_train
)


print(
    "[OK] XGBoost training completed."
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    name,
    X,
    y
):

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print(f"\n{name} Results")

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    return mae, rmse, r2


train_metrics = evaluate(
    "Training",
    X_train,
    y_train
)

val_metrics = evaluate(
    "Validation",
    X_val,
    y_val
)

test_metrics = evaluate(
    "Test",
    X_test,
    y_test
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance")
print("=================")


importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
).sort_values(
    ascending=False
)


for feature, value in importance.items():

    print(
        f"{feature:>15} : "
        f"{value:.6f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

package = {

    "model": model,

    "feature_columns": feature_columns,

    "lag_features": LAG_FEATURES,

    "time_column": TIME_COLUMN

}


OUTPUT_FILE = (
    "model/xgboost_si_lag_model.pkl"
)


joblib.dump(
    package,
    OUTPUT_FILE
)


print(
    f"\n[OK] Model saved to: "
    f"{OUTPUT_FILE}"
)