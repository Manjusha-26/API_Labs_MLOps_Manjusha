# train.py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from data import load_data, split_data
import json

def fit_model(X_train, y_train, X_test, y_test):
    """
    Train a Logistic Regression model and save it.
    """
    log_reg = LogisticRegression(max_iter=1000, random_state=12)
    log_reg.fit(X_train, y_train)

    # Evaluate
    y_pred = log_reg.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc*100:.2f}%")

    # Save model
    joblib.dump(log_reg, "../model/wine_model.pkl")

    # Save metrics
    metrics = {"accuracy": acc}
    with open("../model/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train, X_test, y_test)
