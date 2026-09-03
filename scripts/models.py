import numpy as np


def fit_least_squares(x, y):
    
    x = np.asarray(x)

    X = np.column_stack((
        np.ones(len(x)),
        x
    ))

    beta = np.linalg.pinv(X.T @ X) @ X.T @ y

    intercept = beta[0]
    coefficients = beta[1:]

    if len(coefficients) == 1:
        coefficients = coefficients[0]

    return intercept, coefficients




def fit_ridge(x, y, lambda_value):
    
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y)

    x_centered = x - x_mean
    y_centered = y - y_mean

    n = len(x)

    coefficients = np.linalg.pinv(x_centered.T @ x_centered + n * lambda_value * np.eye(x_centered.shape[1])) @ x_centered.T @ y_centered

    intercept = y_mean - x_mean @ coefficients

    if len(coefficients) == 1:
        coefficients = coefficients[0]

    return intercept, coefficients




def predict(x, intercept, coefficients):
    
    x = np.asarray(x)

    if x.ndim == 1:
        y_pred = intercept + coefficients * x
    else:
        y_pred = intercept + x @ coefficients

    return y_pred