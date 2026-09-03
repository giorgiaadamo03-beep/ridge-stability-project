import numpy as np

from scripts.datasets import synthetic_dataset, real_dataset, real_dataset_base
from scripts.models import fit_least_squares, fit_ridge, predict
from scripts.stability import compute_least_squares_stability, compute_ridge_stability
from scripts.metrics import mse, generalization_gap
from scripts.plots import plot_stability_vs_lambda, plot_test_error_vs_lambda, plot_stability_vs_sample_size
from scripts.eda import dataset_overview, plot_target_distribution, plot_correlation_matrix, plot_rentals_by_hour, plot_rentals_vs_temperature



def print_model_summary(title, train_errors, test_errors, gaps, stabilities):
    print()
    print(title)

    print(
        f"Train MSE: {np.mean(train_errors):.4f} "
        f"(+/- {np.std(train_errors):.4f})"
    )

    print(
        f"Test MSE: {np.mean(test_errors):.4f} "
        f"(+/- {np.std(test_errors):.4f})"
    )

    print(
        f"Generalization gap: {np.mean(gaps):.4f} "
        f"(+/- {np.std(gaps):.4f})"
    )

    print(
        f"Average prediction change: {np.mean(stabilities):.4f} "
        f"(+/- {np.std(stabilities):.4f})"
    )



# =========================================================
# GLOBAL SETTINGS
# =========================================================

n_samples = 200
n_features = 10
test_size = 0.25
n_seeds = 20

lambda_values = [0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

scenarios = {
    "BASE": 0.0,
    "CORRELATED": 0.8
}



# =========================================================
# 1. SYNTHETIC DATASETS
# =========================================================

for scenario_name, correlation in scenarios.items():

    ls_train_errors = []
    ls_test_errors = []
    ls_gaps = []
    ls_stabilities = []

    ridge_results = {}

    for lambda_value in lambda_values:
        ridge_results[lambda_value] = {
            "train_error": [],
            "test_error": [],
            "gap": [],
            "stability": []
        }


    for seed in range(n_seeds):

        X_train, y_train, X_test, y_test, true_beta = synthetic_dataset(
            n_samples = n_samples,
            n_features = n_features,
            test_size = test_size,
            seed = seed,
            correlation = correlation
        )


        # --- LEAST SQUARES ---

        intercept_ls, coefficients_ls = fit_least_squares(X_train, y_train)

        predictions_train_ls = predict(X_train, intercept_ls, coefficients_ls)
        predictions_test_ls = predict(X_test, intercept_ls, coefficients_ls)

        train_error_ls = mse(y_train, predictions_train_ls)
        test_error_ls = mse(y_test, predictions_test_ls)
        gap_ls = generalization_gap(train_error_ls, test_error_ls)
        stability_ls, stability_values_ls = compute_least_squares_stability(X_train, y_train, X_test)

        ls_train_errors.append(train_error_ls)
        ls_test_errors.append(test_error_ls)
        ls_gaps.append(gap_ls)
        ls_stabilities.append(stability_ls)


        # --- RIDGE ---

        for lambda_value in lambda_values:

            intercept_ridge, coefficients_ridge = fit_ridge(X_train, y_train, lambda_value)

            predictions_train_ridge = predict(X_train, intercept_ridge, coefficients_ridge)
            predictions_test_ridge = predict(X_test, intercept_ridge, coefficients_ridge)

            train_error_ridge = mse(y_train, predictions_train_ridge)
            test_error_ridge = mse(y_test, predictions_test_ridge)
            gap_ridge = generalization_gap(train_error_ridge, test_error_ridge)
            stability_ridge, stability_values_ridge = compute_ridge_stability(X_train, y_train, X_test, lambda_value)

            ridge_results[lambda_value]["train_error"].append(train_error_ridge)
            ridge_results[lambda_value]["test_error"].append(test_error_ridge)
            ridge_results[lambda_value]["gap"].append(gap_ridge)
            ridge_results[lambda_value]["stability"].append(stability_ridge)


    print()
    print("=================================================")
    print(f"SCENARIO: {scenario_name}")
    print("=================================================")

    print(f"Correlation: {correlation}")
    print(f"Total observations: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Test size: {test_size}")
    print(f"Number of seeds: {n_seeds}")

    print_model_summary(
        "LEAST SQUARES - MEAN RESULTS",
        ls_train_errors,
        ls_test_errors,
        ls_gaps,
        ls_stabilities
    )

    print()
    print("RIDGE - MEAN RESULTS")

    for lambda_value in lambda_values:

        train_errors = ridge_results[lambda_value]["train_error"]
        test_errors = ridge_results[lambda_value]["test_error"]
        gaps = ridge_results[lambda_value]["gap"]
        stabilities = ridge_results[lambda_value]["stability"]

        print_model_summary(
            f"Lambda = {lambda_value}",
            train_errors,
            test_errors,
            gaps,
            stabilities
        )

    mean_stabilities = []
    std_stabilities = []
    mean_test_errors = []
    std_test_errors = []

    for lambda_value in lambda_values:

        stabilities = ridge_results[lambda_value]["stability"]
        test_errors = ridge_results[lambda_value]["test_error"]

        mean_stabilities.append(np.mean(stabilities))
        std_stabilities.append(np.std(stabilities))
        mean_test_errors.append(np.mean(test_errors))
        std_test_errors.append(np.std(test_errors))

    plot_stability_vs_lambda(lambda_values, mean_stabilities, std_stabilities, np.mean(ls_stabilities), scenario_name)
    plot_test_error_vs_lambda(lambda_values, mean_test_errors, std_test_errors, np.mean(ls_test_errors), scenario_name)




# =========================================================
# 2. EXTENSION: EFFECT OF DATASET SIZE
# =========================================================

sample_sizes = [50, 100, 200, 500]
sample_size_n_features = 10
sample_size_test_size = 0.25
sample_size_n_seeds = 20
sample_size_lambda = 0.01

ls_sample_size_stabilities = []
ridge_sample_size_stabilities = []


for sample_size in sample_sizes:

    ls_stabilities_current_size = []
    ridge_stabilities_current_size = []

    for seed in range(sample_size_n_seeds):

        X_train_size, y_train_size, X_test_size, y_test_size, true_beta_size = synthetic_dataset(
            n_samples = sample_size,
            n_features = sample_size_n_features,
            test_size = sample_size_test_size,
            seed = seed,
            correlation = 0.0
        )

        stability_ls_size, stability_values_ls_size = compute_least_squares_stability(X_train_size, y_train_size, X_test_size)
        stability_ridge_size, stability_values_ridge_size = compute_ridge_stability(X_train_size, y_train_size, X_test_size, sample_size_lambda)

        ls_stabilities_current_size.append(stability_ls_size)
        ridge_stabilities_current_size.append(stability_ridge_size)

    ls_sample_size_stabilities.append(np.mean(ls_stabilities_current_size))
    ridge_sample_size_stabilities.append(np.mean(ridge_stabilities_current_size))


print()
print("===================================")
print("EFFECT OF DATASET SIZE ON STABILITY")
print("===================================")
print(f"Number of features: {sample_size_n_features}")
print(f"Test size: {sample_size_test_size}")
print(f"Number of seeds: {sample_size_n_seeds}")
print(f"Ridge lambda: {sample_size_lambda}")

for i in range(len(sample_sizes)):
    print()
    print(f"Sample size = {sample_sizes[i]}")
    print(f"Least Squares stability: " f"{ls_sample_size_stabilities[i]:.4f}")
    print(f"Ridge stability: " f"{ridge_sample_size_stabilities[i]:.4f}")

plot_stability_vs_sample_size(sample_sizes, ls_sample_size_stabilities, ridge_sample_size_stabilities, sample_size_lambda)




# =========================================================
# 3. REAL DATASET: SEOUL BIKE SHARING
# =========================================================

# --- EXPLORATORY DATA ANALYSIS ---

real_df = real_dataset_base()

dataset_overview(real_df)
plot_target_distribution(real_df)
plot_correlation_matrix(real_df)
plot_rentals_by_hour(real_df)
plot_rentals_vs_temperature(real_df)


real_n_seeds = 10
real_sample_size = 1000

real_ls_train_errors = []
real_ls_test_errors = []
real_ls_gaps = []
real_ls_stabilities = []

ridge_real_results = {}

for lambda_value in lambda_values:
    ridge_real_results[lambda_value] = {
        "train_error": [],
        "test_error": [],
        "gap": [],
        "stability": []
    }


for seed in range(real_n_seeds):

    X_train_real, y_train_real, X_test_real, y_test_real, feature_names = real_dataset(
        test_size = 0.25,
        seed = seed,
        sample_size = real_sample_size
    )

    # --- LEAST SQUARES ---

    intercept_ls_real, coefficients_ls_real = fit_least_squares(X_train_real, y_train_real)

    predictions_train_ls_real = predict(X_train_real, intercept_ls_real, coefficients_ls_real)
    predictions_test_ls_real = predict(X_test_real, intercept_ls_real,coefficients_ls_real)

    train_error_ls_real = mse(y_train_real, predictions_train_ls_real)
    test_error_ls_real = mse(y_test_real, predictions_test_ls_real)
    gap_ls_real = generalization_gap(train_error_ls_real, test_error_ls_real)
    stability_ls_real, stability_values_ls_real = compute_least_squares_stability(X_train_real, y_train_real, X_test_real)


    real_ls_train_errors.append(train_error_ls_real)
    real_ls_test_errors.append(test_error_ls_real)
    real_ls_gaps.append(gap_ls_real)
    real_ls_stabilities.append(stability_ls_real)


    # --- RIDGE ---

    for lambda_value in lambda_values:

        intercept_ridge_real, coefficients_ridge_real = fit_ridge(X_train_real, y_train_real, lambda_value)

        predictions_train_ridge_real = predict(X_train_real, intercept_ridge_real, coefficients_ridge_real)
        predictions_test_ridge_real = predict(X_test_real, intercept_ridge_real, coefficients_ridge_real)

        train_error_ridge_real = mse(y_train_real, predictions_train_ridge_real)
        test_error_ridge_real = mse(y_test_real, predictions_test_ridge_real)
        gap_ridge_real = generalization_gap(train_error_ridge_real, test_error_ridge_real)
        stability_ridge_real, stability_values_ridge_real = compute_ridge_stability(X_train_real, y_train_real, X_test_real, lambda_value)


        ridge_real_results[lambda_value]["train_error"].append(train_error_ridge_real)
        ridge_real_results[lambda_value]["test_error"].append(test_error_ridge_real)
        ridge_real_results[lambda_value]["gap"].append(gap_ridge_real)
        ridge_real_results[lambda_value]["stability"].append(stability_ridge_real)




print()
print("=================================================")
print("REAL DATASET: SEOUL BIKE SHARING")
print("=================================================")
print(f"Sample size: {real_sample_size}")
print(f"Number of seeds: {real_n_seeds}")
print(f"Number of features: {X_train_real.shape[1]}")

print_model_summary(
    "LEAST SQUARES - MEAN RESULTS",
    real_ls_train_errors,
    real_ls_test_errors,
    real_ls_gaps,
    real_ls_stabilities
)

print()
print("RIDGE - MEAN RESULTS")

for lambda_value in lambda_values:

    train_errors_real = ridge_real_results[lambda_value]["train_error"]
    test_errors_real = ridge_real_results[lambda_value]["test_error"]
    gaps_real = ridge_real_results[lambda_value]["gap"]
    stabilities_real = ridge_real_results[lambda_value]["stability"]

    print_model_summary(
        f"Lambda = {lambda_value}",
        train_errors_real,
        test_errors_real,
        gaps_real,
        stabilities_real
    )

real_mean_stabilities = []
real_std_stabilities = []
real_mean_test_errors = []
real_std_test_errors = []

for lambda_value in lambda_values:

    stabilities_real = ridge_real_results[lambda_value]["stability"]
    test_errors_real = ridge_real_results[lambda_value]["test_error"]

    real_mean_stabilities.append(np.mean(stabilities_real))
    real_std_stabilities.append(np.std(stabilities_real))
    real_mean_test_errors.append(np.mean(test_errors_real))
    real_std_test_errors.append(np.std(test_errors_real))


plot_stability_vs_lambda(lambda_values, real_mean_stabilities, real_std_stabilities, np.mean(real_ls_stabilities), "SEOUL BIKE SHARING")
plot_test_error_vs_lambda(lambda_values, real_mean_test_errors, real_std_test_errors, np.mean(real_ls_test_errors), "SEOUL BIKE SHARING")