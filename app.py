"""
app.py
------
Flask web app for the Multiple Disease Prediction System.
Loads the pre-trained models from ./models/ (run train_models.py first)
and serves prediction forms for Diabetes, Heart Disease, and Parkinson's.

Run:
    python train_models.py   # once, to create ./models/*.pkl
    python app.py            # then start the server
"""

import os
import pickle

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_pickle(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ----- Load all artifacts once at startup -----
diabetes_model = load_pickle("diabetes_model.pkl")
diabetes_scaler = load_pickle("diabetes_scaler.pkl")
diabetes_features = load_pickle("diabetes_features.pkl") or [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

heart_model = load_pickle("heart_model.pkl")
heart_scaler = load_pickle("heart_scaler.pkl")
heart_encoders = load_pickle("heart_encoders.pkl") or {}
heart_features = load_pickle("heart_features.pkl") or [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

parkinsons_model = load_pickle("parkinsons_model.pkl")
parkinsons_scaler = load_pickle("parkinsons_scaler.pkl")
parkinsons_features = load_pickle("parkinsons_features.pkl") or [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]

MODEL_ARTIFACTS = (
    diabetes_model, diabetes_scaler, diabetes_features,
    heart_model, heart_scaler, heart_features,
    parkinsons_model, parkinsons_scaler, parkinsons_features,
)
MODELS_READY = all(artifact is not None for artifact in MODEL_ARTIFACTS)
DIABETES_READY = all(artifact is not None for artifact in (diabetes_model, diabetes_scaler, diabetes_features))
HEART_READY = all(artifact is not None for artifact in (heart_model, heart_scaler, heart_features))
PARKINSONS_READY = all(artifact is not None for artifact in (parkinsons_model, parkinsons_scaler, parkinsons_features))

HEART_CATEGORICAL_OPTIONS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["Y", "N"],
    "ST_Slope": ["Up", "Flat", "Down"],
}


def scaled_input(values, features, scaler):
    frame = pd.DataFrame([values], columns=features)
    return scaler.transform(frame)


def prediction_result(model, arr, positive_label, negative_label):
    prediction = int(model.predict(arr)[0])
    probabilities = model.predict_proba(arr)[0] if hasattr(model, "predict_proba") else None
    risk_probability = None
    confidence = None
    if probabilities is not None:
        class_probabilities = dict(zip(model.classes_, probabilities))
        risk_probability = round(float(class_probabilities.get(1, 0)) * 100, 1)
        confidence = round(float(class_probabilities.get(prediction, 0)) * 100, 1)
    return {
        "positive": prediction == 1,
        "label": positive_label if prediction == 1 else negative_label,
        "confidence": confidence,
        "risk_percentage": risk_probability,
        "healthy_percentage": round(100 - risk_probability, 1) if risk_probability is not None else None,
    }


@app.route("/")
def index():
    return render_template("index.html", models_ready=MODELS_READY,
                           diabetes_ready=DIABETES_READY, heart_ready=HEART_READY,
                           parkinsons_ready=PARKINSONS_READY)


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    if request.method == "POST":
        try:
            if not DIABETES_READY:
                raise RuntimeError("The diabetes model is unavailable. Run train_models.py first.")
            values = [float(request.form[f]) for f in diabetes_features]
            arr = scaled_input(values, diabetes_features, diabetes_scaler)
            result = prediction_result(diabetes_model, arr, "Diabetic", "Not Diabetic")
        except (KeyError, TypeError, ValueError):
            result = {"error": "Enter a number in every field before scoring."}
        except Exception as exc:
            result = {"error": str(exc)}
    return render_template("diabetes.html", features=diabetes_features, result=result,
                            models_ready=DIABETES_READY)


@app.route("/heart", methods=["GET", "POST"])
def heart():
    result = None
    if request.method == "POST":
        try:
            if not HEART_READY:
                raise RuntimeError("The heart disease model is unavailable. Run train_models.py first.")
            values = []
            for feat in heart_features:
                raw = request.form[feat]
                if feat in heart_encoders:
                    encoded = heart_encoders[feat].transform([raw])[0]
                    values.append(encoded)
                else:
                    values.append(float(raw))
            arr = scaled_input(values, heart_features, heart_scaler)
            result = prediction_result(heart_model, arr, "Heart Disease Detected", "No Heart Disease")
        except (KeyError, TypeError, ValueError):
            result = {"error": "Enter a valid value in every field before scoring."}
        except Exception as exc:
            result = {"error": str(exc)}
    return render_template("heart.html", features=heart_features, options=HEART_CATEGORICAL_OPTIONS,
                            result=result, models_ready=HEART_READY)


@app.route("/parkinsons", methods=["GET", "POST"])
def parkinsons():
    result = None
    if request.method == "POST":
        try:
            if not PARKINSONS_READY:
                raise RuntimeError("The Parkinson's model is unavailable. Run train_models.py first.")
            values = [float(request.form[f]) for f in parkinsons_features]
            arr = scaled_input(values, parkinsons_features, parkinsons_scaler)
            result = prediction_result(parkinsons_model, arr, "Parkinson's Detected", "Healthy")
        except (KeyError, TypeError, ValueError):
            result = {"error": "Enter a number in every field before scoring."}
        except Exception as exc:
            result = {"error": str(exc)}
    return render_template("parkinsons.html", features=parkinsons_features, result=result,
                            models_ready=PARKINSONS_READY)


if __name__ == "__main__":
    if not MODELS_READY:
        print("⚠️  Models not found in ./models/. Run `python train_models.py` first.")
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(debug=debug)
