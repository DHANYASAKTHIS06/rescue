import joblib
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

model = joblib.load("knn_regressor_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

app = Flask(__name__)

CORS(app)

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Gesture Prediction API Running Successfully"
    })

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        required_features = [
            "ax",
            "ay",
            "az",
            "gx",
            "gy",
            "gz"
        ]

        missing_features = []

        for feature in required_features:

            if feature not in data:
                missing_features.append(feature)

        if len(missing_features) > 0:

            return jsonify({
                "success": False,
                "error": f"Missing Features: {missing_features}"
            }), 400

        input_df = pd.DataFrame([[
            float(data["ax"]),
            float(data["ay"]),
            float(data["az"]),
            float(data["gx"]),
            float(data["gy"]),
            float(data["gz"])
        ]], columns=required_features)

        prediction = model.predict(input_df)

        predicted_encoded = int(round(prediction[0]))

        predicted_encoded = max(
            0,
            min(
                predicted_encoded,
                len(label_encoder.classes_) - 1
            )
        )

        predicted_gesture = label_encoder.inverse_transform(
            [predicted_encoded]
        )[0]

        return jsonify({

            "success": True,

            "input": {
                "ax": data["ax"],
                "ay": data["ay"],
                "az": data["az"],
                "gx": data["gx"],
                "gy": data["gy"],
                "gz": data["gz"]
            },

            "predicted_gesture": str(predicted_gesture),

            "encoded_prediction": int(predicted_encoded)

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
