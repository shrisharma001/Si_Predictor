from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "model/xgboost_si_model.pkl"

package = joblib.load(
    MODEL_FILE
)

model = package["model"]

FEATURES = package["feature_columns"]


print(
    f"[OK] XGBoost model loaded"
)

print(
    f"[OK] Number of features: "
    f"{len(FEATURES)}"
)


# ============================================================
# STATUS
# ============================================================

def get_status(prediction):

    if prediction < 0.4 or prediction > 1.2:

        return "out_of_range"

    elif prediction < 0.5 or prediction > 1.0:

        return "warning"

    return "normal"


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/predict", methods=["GET"])
def predict_page():

    return render_template(
        "predict.html",
        features=FEATURES
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No input data received"
            }), 400


        # Check that all required features exist

        missing = [
            feature
            for feature in FEATURES
            if feature not in data
        ]

        if missing:

            return jsonify({
                "error":
                    "Missing features",
                "features":
                    missing
            }), 400


        # Convert inputs to numbers

        values = []

        for feature in FEATURES:

            value = float(
                data[feature]
            )

            if not np.isfinite(value):

                raise ValueError(
                    f"Invalid value for {feature}"
                )

            values.append(value)


        # Make prediction

        prediction = model.predict(
            [values]
        )[0]

        prediction = round(
            float(prediction),
            4
        )


        status = get_status(
            prediction
        )


        return jsonify({

            "prediction":
                prediction,

            "status":
                status
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )