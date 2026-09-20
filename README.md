# Multiple Disease Prediction — Flask Web App

A Flask front end for the Diabetes / Heart Disease / Parkinson's models built in
`Multiple_Disease_Prediction_System.ipynb`. Enter clinical values in a form and get an
instant prediction with model confidence.

## Setup

```bash
pip install -r requirements.txt
python train_models.py   # fetches the 3 datasets, trains + saves models to ./models/
python app.py             # starts the server at http://127.0.0.1:5000
```

Open **http://127.0.0.1:5000** in your browser. `train_models.py` needs internet access
once, to download the three public datasets — after that everything runs locally.

## Project structure

```
disease_prediction_flask_app/
├── app.py                # Flask routes + prediction logic
├── train_models.py       # fetches data, trains models, saves .pkl files
├── requirements.txt
├── models/                # created after running train_models.py
│   ├── diabetes_model.pkl / diabetes_scaler.pkl / diabetes_features.pkl
│   ├── heart_model.pkl / heart_scaler.pkl / heart_encoders.pkl / heart_features.pkl
│   └── parkinsons_model.pkl / parkinsons_scaler.pkl / parkinsons_features.pkl
├── templates/
│   ├── base.html, index.html
│   ├── diabetes.html, heart.html, parkinsons.html
└── static/
    └── style.css
```

## How predictions work

1. `train_models.py` loads each dataset from its public URL, cleans it, trains six
   classic classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, KNN,
   Naive Bayes), keeps whichever scores best on a held-out test split, and pickles it
   along with its `StandardScaler` (and `LabelEncoder`s for Heart Disease's categorical
   fields).
2. `app.py` loads those pickles once at startup. Each form POST scales the submitted
   values the same way the training data was scaled, then calls `model.predict()` /
   `model.predict_proba()`.

## Notes

- If you see "Models not found" on the home page, you haven't run `train_models.py` yet.
- Re-run `train_models.py` any time to retrain on fresh data or after changing the pipeline.
- This is an educational project, not a medical device — don't use it for real diagnosis.
