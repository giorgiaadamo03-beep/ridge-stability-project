import numpy as np

from scripts.models import fit_least_squares, fit_ridge, predict


def compute_least_squares_stability(x_train, y_train, x_test):

    intercept_full, coefficients_full = fit_least_squares(x_train, y_train)
    predictions_full = predict(x_test, intercept_full, coefficients_full)

    stability_values = []

    for i in range(len(x_train)):

        x_reduced = np.delete(
            x_train,
            i,
            axis = 0
        )

        y_reduced = np.delete(
            y_train,
            i
        )

        intercept_reduced, coefficients_reduced = fit_least_squares(x_reduced, y_reduced)
        predictions_reduced = predict(x_test, intercept_reduced, coefficients_reduced)

        prediction_difference = np.abs(predictions_full - predictions_reduced)

        change = np.mean(prediction_difference)

        stability_values.append(change)


    overall_stability = np.mean(stability_values)

    return overall_stability, stability_values






def compute_ridge_stability(x_train, y_train, x_test, lambda_value):
   
    intercept_full, coefficients_full = fit_ridge(x_train, y_train, lambda_value)
    predictions_full = predict(x_test, intercept_full, coefficients_full)

    stability_values = []

    for i in range(len(x_train)):

        x_reduced = np.delete(
            x_train,
            i,
            axis = 0
        )

        y_reduced = np.delete(
            y_train,
            i
        )

        intercept_reduced, coefficients_reduced = fit_ridge(x_reduced, y_reduced, lambda_value)
        predictions_reduced = predict(x_test, intercept_reduced, coefficients_reduced)

        prediction_difference = np.abs(predictions_full - predictions_reduced)

        change = np.mean(prediction_difference)

        stability_values.append(change)

    overall_stability = np.mean(stability_values)

    return overall_stability, stability_values