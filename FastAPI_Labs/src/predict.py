# predict.py
import joblib


def predict_data(X):
    """
    Predict wine class for given features.
    """
    model = joblib.load("../model/wine_model.pkl")  # load once

    return model.predict(X)
