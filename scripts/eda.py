

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def dataset_overview(df):

    print("DATASET SHAPE")
    print(df.shape)

    print()
    print("DATA TYPES")
    print(df.dtypes)

    print()
    print("MISSING VALUES")
    print(df.isnull().sum())

    print()
    print("DESCRIPTIVE STATISTICS")
    print(df.describe())



def plot_target_distribution(df):

    plt.figure()

    plt.hist(
        df["Rented Bike Count"],
        bins = 30
    )

    plt.xlabel("Rented Bike Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Rented Bike Count")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "target_distribution.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()



def plot_correlation_matrix(df):

    numeric_df = df.select_dtypes(include = ["number"])

    corr = numeric_df.corr()

    plt.figure(figsize = (10, 8))

    plt.imshow(
        corr,
        vmin = -1,
        vmax = 1,
        interpolation = "nearest"
    )

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation = 90,
        fontsize = 8
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
        fontsize = 8
    )

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = round(corr.iloc[i, j], 2)

            plt.text(
                j,
                i,
                value,
                ha = "center",
                va = "center",
                fontsize = 7
            )

    plt.title("Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "correlation_matrix.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()



def plot_rentals_by_hour(df):

    hourly_mean = df.groupby("Hour")["Rented Bike Count"].mean()

    plt.figure()

    plt.plot(
        hourly_mean.index,
        hourly_mean.values,
        marker = "o"
    )

    plt.xlabel("Hour")
    plt.ylabel("Average Rented Bike Count")
    plt.title("Average Bike Rentals by Hour")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rentals_by_hour.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()




def plot_rentals_vs_temperature(df):
    
    plt.figure()

    plt.scatter(
        df["Temperature(°C)"],
        df["Rented Bike Count"],
        alpha = 0.4
    )

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Rented Bike Count")
    plt.title("Bike Rentals vs Temperature")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "rentals_vs_temperature.png",
        dpi = 300,
        bbox_inches = "tight"
    )

    plt.show()

