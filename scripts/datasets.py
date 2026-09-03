import numpy as np
import pandas as pd



def synthetic_dataset(n_samples = 200, n_features = 10, test_size = 0.25, seed = 42, correlation = 0.0):

    rng = np.random.default_rng(seed)

    # Correlation matrix between features
    correlation_matrix = np.full(
        (n_features, n_features),
        correlation
    )

    np.fill_diagonal(
        correlation_matrix,
        1.0
    )

    # Matrix of features
    X = rng.multivariate_normal(
        mean = np.zeros(n_features),
        cov = correlation_matrix,
        size = n_samples
    )

    # True coefficients used to generate the target
    true_beta = np.linspace(
        1.0,
        3.0,
        n_features
    )

    # Noise 
    noise = rng.normal(
        loc = 0.0,
        scale = 1.0,
        size = n_samples
    )

    y = X @ true_beta + noise

    indices = rng.permutation(n_samples)    # shuffle the observations 

    X = X[indices]
    y = y[indices]


    # ----------------- TRAIN TEST SPLIT -----------------

    n_test = int(n_samples * test_size)

    X_test = X[:n_test]
    y_test = y[:n_test]

    X_train = X[n_test:]
    y_train = y[n_test:]

    return X_train, y_train, X_test, y_test, true_beta






def real_dataset(test_size = 0.25, seed = 42, sample_size = 1000):

    df = pd.read_csv("data/SeoulBikeData.csv", encoding = "unicode_escape")

    # Only observations where the bike sharing service was active
    df = df[df["Functioning Day"] == "Yes"].copy()

    y = df["Rented Bike Count"].to_numpy()

    date = pd.to_datetime(df["Date"], format = "%d/%m/%Y")
    df["Month"] = date.dt.month
    df["DayOfWeek"] = date.dt.dayofweek


    df["Holiday"] = df["Holiday"].map({"No Holiday": 0, "Holiday": 1})

    df = pd.get_dummies(
        df,
        columns = ["Seasons", "Month", "DayOfWeek", "Hour"],
        drop_first = True,
        dtype = int
    )


    X = df.drop(
        columns = [
            "Date",
            "Rented Bike Count",
            "Functioning Day"
        ]
    )

    feature_names = X.columns.tolist()

    X = X.to_numpy(dtype = float)


    # ----------------- SAMPLE -----------------

    rng = np.random.default_rng(seed)

    if sample_size is not None and sample_size < len(X):
        sample_indices = rng.choice(
            len(X),
            size = sample_size,
            replace = False
        )

        X = X[sample_indices]
        y = y[sample_indices]


    # ----------------- TRAIN TEST SPLIT -----------------

    indices = rng.permutation(len(X))

    X = X[indices]
    y = y[indices]

    n_test = int(len(X) * test_size)

    X_test = X[:n_test]
    y_test = y[:n_test]

    X_train = X[n_test:]
    y_train = y[n_test:]


    # ----------------- STANDARDIZATION -----------------

    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)
    train_std[train_std == 0] = 1.0

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std


    return X_train, y_train, X_test, y_test, feature_names





def real_dataset_base():
    
    df = pd.read_csv(
        "data/SeoulBikeData.csv",
        encoding = "unicode_escape"
    )

    return df