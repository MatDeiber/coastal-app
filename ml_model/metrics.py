import numpy as np

def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def compute_mae(y_pred, y_true):
    return (np.abs(y_pred - y_true)).mean()

def compute_mape(y_pred, y_true):
    return (np.abs(y_pred - y_true) / y_true).mean()