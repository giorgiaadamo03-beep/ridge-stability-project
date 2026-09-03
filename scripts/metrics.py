import numpy as np


def mse(y_true, y_pred):

    errors = y_true - y_pred
    squared_errors = errors ** 2
    mean_error = np.mean(squared_errors)

    return mean_error



def generalization_gap(train_error, test_error):

    gap = test_error - train_error

    return gap