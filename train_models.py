"""
train_models.py
----------------
Trains the three disease-prediction models (Diabetes, Heart Disease, Parkinson's)
and saves them into ./models/ as .pkl files for the Flask app to load.

This mirrors the pipeline in Multiple_Disease_Prediction_System.ipynb.
Run this once before starting app.py:

    python train_models.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }


def train_best_model(X_train, y_train, X_test, y_test, label):
    best_name, best_model, best_acc = None, None, -1
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"  {name:20s} -> accuracy: {acc:.4f}")
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc
    print(f"  Best for {label}: {best_name} ({best_acc:.4f})\n")
    return best_model, best_name, best_acc


def save(obj, filename):
    with open(os.path.join(MODELS_DIR, filename), "wb") as f:
        pickle.dump(obj, f)


# ============================================================
# 1. DIABETES
# ============================================================
def train_diabetes():
    print("Training Diabetes model...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(url, names=cols)

    zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_not_allowed] = df[zero_not_allowed].replace(0, np.nan)
    for col in zero_not_allowed:
        df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model, name, acc = train_best_model(X_train_s, y_train, X_test_s, y_test, "Diabetes")

    save(model, "diabetes_model.pkl")
    save(scaler, "diabetes_scaler.pkl")
    save(list(X.columns), "diabetes_features.pkl")
    return name, acc


# ============================================================
# 2. HEART DISEASE
# ============================================================
def train_heart():
    print("Training Heart Disease model...")
    url = "https://raw.githubusercontent.com/akarshsnair/Dataset-cart/main/heart.csv"
    df = pd.read_csv(url)

    # A few rows have blank/missing cells in this dataset; impute before encoding/scaling
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    numeric_cols = [c for c in df.columns if c not in categorical_cols + ["HeartDisease"]]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model, name, acc = train_best_model(X_train_s, y_train, X_test_s, y_test, "Heart Disease")

    save(model, "heart_model.pkl")
    save(scaler, "heart_scaler.pkl")
    save(encoders, "heart_encoders.pkl")
    save(list(X.columns), "heart_features.pkl")
    return name, acc


# ============================================================
# 3. PARKINSON'S
# ============================================================
def train_parkinsons():
    print("Training Parkinson's model...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    df = pd.read_csv(url)
    df = df.drop(columns=["name"])

    X = df.drop("status", axis=1)
    y = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model, name, acc = train_best_model(X_train_s, y_train, X_test_s, y_test, "Parkinson's")

    save(model, "parkinsons_model.pkl")
    save(scaler, "parkinsons_scaler.pkl")
    save(list(X.columns), "parkinsons_features.pkl")
    return name, acc


if __name__ == "__main__":
    results = {}
    results["Diabetes"] = train_diabetes()
    results["Heart Disease"] = train_heart()
    results["Parkinson's"] = train_parkinsons()

    print("=" * 50)
    print("TRAINING COMPLETE")
    for disease, (name, acc) in results.items():
        print(f"{disease:15s} -> {name} (accuracy: {acc:.4f})")
    print(f"\nModels saved to: {MODELS_DIR}")
    print("You can now run: python app.py")
