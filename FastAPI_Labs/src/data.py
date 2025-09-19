# data.py
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load Wine dataset and return features and targets.
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y

def split_data(X, y):
    """
    Split data into train/test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test
