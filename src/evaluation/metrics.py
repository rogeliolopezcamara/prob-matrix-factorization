# src/evaluation/metrics.py

import numpy as np

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error between true and predicted ratings.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    """
    Compute Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))