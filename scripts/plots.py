
import matplotlib.pyplot as plt
from pathlib import Path


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_stability_vs_lambda(lambda_values, mean_stabilities, std_stabilities, ls_stability, scenario_name):

    plt.figure()

    plt.errorbar(
        lambda_values,
        mean_stabilities,
        yerr = std_stabilities,
        marker = "o",
        capsize = 4,
        label = "Ridge"
    )

    plt.axhline(
        y = ls_stability,
        linestyle = "--",
        label = "Least Squares"
    )

    plt.xscale("log")

    plt.xlabel("Lambda")
    plt.ylabel("Average prediction change")

    plt.title(f"Average prediction change vs regularization strength - {scenario_name}")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    safe_name = scenario_name.lower().replace(" ", "_")

    plt.savefig(
        FIGURES_DIR / f"stability_vs_lambda_{safe_name}.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()





def plot_test_error_vs_lambda(lambda_values, mean_test_errors, std_test_errors, ls_test_error, scenario_name):

    plt.figure()

    plt.errorbar(
        lambda_values,
        mean_test_errors,
        yerr = std_test_errors,
        marker = "o",
        capsize = 4,
        label = "Ridge"
    )

    plt.axhline(
        y = ls_test_error,
        linestyle = "--",
        label = "Least Squares"
    )

    plt.xscale("log")

    plt.xlabel("Lambda")
    plt.ylabel("Test MSE")

    plt.title(f"Test error vs regularization strength - {scenario_name}")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    safe_name = scenario_name.lower().replace(" ", "_")

    plt.savefig(
        FIGURES_DIR / f"test_error_vs_lambda_{safe_name}.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()



def plot_stability_vs_sample_size(sample_sizes, ls_mean_stabilities, ridge_mean_stabilities, lambda_value):

    plt.figure()

    plt.plot(
        sample_sizes,
        ls_mean_stabilities,
        marker = "o",
        label = "Least Squares"
    )

    plt.plot(
        sample_sizes,
        ridge_mean_stabilities,
        marker = "o",
        label = f"Ridge (lambda = {lambda_value})"
    )

    plt.xlabel("Dataset size")
    plt.ylabel("Average prediction change")

    plt.title("Average prediction change vs dataset size")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "stability_vs_sample_size.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()