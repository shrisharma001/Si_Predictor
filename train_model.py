import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/1_blast_furnace_data_first_dataset.xlsx"
MODEL_FILE = "model/xgboost_si_model.pkl"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# FEATURES
# ============================================================

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

TARGET = "Si"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("[INFO] Loading dataset...")

    sensor_df = pd.read_excel(
        DATA_FILE,
        sheet_name="data"
    )

    si_df = pd.read_excel(
        DATA_FILE,
        sheet_name="Si"
    )

    print(f"[INFO] Sensor data shape: {sensor_df.shape}")
    print(f"[INFO] Si data shape: {si_df.shape}")

    return sensor_df, si_df


# ============================================================
# PREPARE TIMESTAMPS
# ============================================================

def prepare_timestamps(sensor_df, si_df):

    # Display columns so we can verify the actual names
    print("\n[INFO] Sensor columns:")
    print(sensor_df.columns.tolist())

    print("\n[INFO] Si columns:")
    print(si_df.columns.tolist())

    return sensor_df, si_df


# ============================================================
# TIME ALIGNMENT
# ============================================================

def align_data(sensor_df, si_df):

    """
    Match every Si measurement with the most recent
    available sensor measurement.

    This prevents using future sensor information.
    """

    # IMPORTANT:
    # Replace these two column names if the Excel file
    # uses different timestamp names.

    sensor_time_col = "dt"
    si_time_col = "dt"

    sensor_df[sensor_time_col] = pd.to_datetime(
        sensor_df[sensor_time_col]
    )

    si_df[si_time_col] = pd.to_datetime(
        si_df[si_time_col]
    )

    sensor_df = sensor_df.sort_values(
        sensor_time_col
    )

    si_df = si_df.sort_values(
        si_time_col
    )

    aligned = pd.merge_asof(
        si_df,
        sensor_df,
        left_on=si_time_col,
        right_on=sensor_time_col,
        direction="backward"
    )

    return aligned


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n[INFO] Cleaning data...")

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing_columns)
        )

    df = df[required_columns].copy()

    # Convert everything to numeric
    for col in required_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    before = len(df)

    df = df.dropna()

    after = len(df)

    print(
        f"[INFO] Removed {before - after} rows "
        f"containing missing values."
    )

    print(f"[INFO] Final dataset size: {df.shape}")

    return df


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df):

    n = len(df)

    train_end = int(
        n * TRAIN_RATIO
    )

    valid_end = int(
        n * (TRAIN_RATIO + VALID_RATIO)
    )

    train = df.iloc[:train_end]

    valid = df.iloc[
        train_end:valid_end
    ]

    test = df.iloc[
        valid_end:
    ]

    print("\n[INFO] Dataset split:")

    print(
        f"Training   : {len(train)} rows"
    )

    print(
        f"Validation : {len(valid)} rows"
    )

    print(
        f"Testing    : {len(test)} rows"
    )

    return train, valid, test


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_model(X_train, y_train):

    print("\n[INFO] Training XGBoost...")

    model = XGBRegressor(

        n_estimators=500,

        learning_rate=0.05,

        max_depth=6,

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

    print("[OK] XGBoost training completed.")

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X, y, name):

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

    print(
        f"\n{name} Results"
    )

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(model):

    importance = pd.DataFrame({

        "feature": FEATURES,

        "importance":
            model.feature_importances_

    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print("\nFeature Importance")
    print("=================")

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:>6} : "
            f"{row['importance']:.6f}"
        )

    return importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    os.makedirs(
        os.path.dirname(MODEL_FILE),
        exist_ok=True
    )

    package = {

        "model": model,

        "feature_columns": FEATURES,

        "target": TARGET
    }

    joblib.dump(
        package,
        MODEL_FILE
    )

    print(
        f"\n[OK] Model saved to: "
        f"{MODEL_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    sensor_df, si_df = load_data()

    sensor_df, si_df = prepare_timestamps(
        sensor_df,
        si_df
    )

    print("\n[INFO] Aligning sensor and Si data...")

    df = align_data(
        sensor_df,
        si_df
    )

    df = clean_data(df)

    train, valid, test = split_data(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_valid = valid[FEATURES]
    y_valid = valid[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_train,
        y_train,
        "Training"
    )

    evaluate_model(
        model,
        X_valid,
        y_valid,
        "Validation"
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        "Test"
    )

    show_feature_importance(
        model
    )

    save_model(
        model
    )


if __name__ == "__main__":
    main()