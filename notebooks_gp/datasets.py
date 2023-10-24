import random

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import fetch_california_housing
from ucimlrepo import fetch_ucirepo


def get_concrete():
    concrete_df = pd.read_csv("../datasets/concrete_data.csv")
    X, y = concrete_df[concrete_df.columns[:7]], concrete_df[concrete_df.columns[7]]
    return np.array(X), np.array(y)


def get_wine():
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    return np.array(X), np.array(y)[:, 0]


def get_bio(n=10000):
    df = pd.read_csv("../datasets/CASP.csv")
    if n == "all":
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    elif isinstance(n, int):
        random.seed(42)
        indices = np.random.choice(range(len(df)), n, replace=False)
        df = df.iloc[indices]
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    return X, y


def get_abalone():

    abalone = fetch_ucirepo(id=1)

    # data (as pandas dataframes)
    X = abalone.data.features
    y = abalone.data.targets

    return X.values, y.values


def get_mpg():
    auto_mpg = fetch_ucirepo(id=9)

    # data (as pandas dataframes)
    X = auto_mpg.data.features
    y = auto_mpg.data.targets

    return X.values, y.values


def get_liver():
    liver_disorders = fetch_ucirepo(id=60)

    # data (as pandas dataframes)
    X = liver_disorders.data.features
    y = liver_disorders.data.targets

    return X.values, y.values


def _wing_weight(x, noisy=False):
    t1 = .036 * x[:, 0]**(.758)
    t2 = x[:, 1]**(.0035)
    t3 = (x[:, 2]/(np.cos(x[:, 3])**2))**(.6)
    t4 = x[:, 4]**(.006)
    t5 = x[:, 5]**(.04)
    t6 = ((np.cos(x[:, 3])) / (100 * x[:, 6]))**(.3)
    t7 = (x[:, 7] * x[:, 8])**(.49)
    t8 = x[:, 0] * x[:, 9]
    if noisy:
        noise = np.random.normal(0, 25, x.shape[0])
        return t1 * t2 * t3 * t4 * t5 * t6 * t7 + t8 + noise
    else:
        return t1 * t2 * t3 * t4 * t5 * t6 * t7 + t8


def get_wing_weight(noisy=False):
    nobs = 600
    np.random.seed(42)
    x1 = np.random.uniform(low=150, high=200, size=(nobs, 1))
    x2 = np.random.uniform(low=220, high=300, size=(nobs, 1))
    x3 = np.random.uniform(low=6, high=10, size=(nobs, 1))
    x4 = np.random.uniform(low=-10, high=10, size=(nobs, 1)) * (np.pi/180)
    x5 = np.random.uniform(low=16, high=45, size=(nobs, 1))
    x6 = np.random.uniform(low=.5, high=1, size=(nobs, 1))
    x7 = np.random.uniform(low=.08, high=.18, size=(nobs, 1))
    x8 = np.random.uniform(low=2.5, high=6, size=(nobs, 1))
    x9 = np.random.uniform(low=1700, high=2500, size=(nobs, 1))
    x10 = np.random.uniform(low=0.025, high=.08, size=(nobs, 1))
    X = np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], axis=1)
    y = _wing_weight(X, noisy=noisy)

    return X, y


def _noisy_morokoff(x, d, noisy):
    noise = np.random.normal(0, 1e-4, x.shape[0]) if noisy else 0
    return .5 * (1 + 1 / d)**d * (x ** (1 / d)).prod(axis=1) + noise


def get_morokoff(noisy=False):
    cov = np.array(
        [
            [1, .9, 0, 0, 0, .05, -.3, 0, 0, 0],
            [.9, 1, 0, 0, 0, 0, 0, .1, 0, 0],
            [0, 0, 1, 0, -.3, .1, .4, 0, .05, 0],
            [0, 0, 0, 1, .4, 0, 0, -.35, 0, 0],
            [0, 0, -.3, .4, 1, 0, 0, 0, .1, 0],
            [.05, 0, .1, 0, 0, 1, 0, 0, 0, 0],
            [-.3, 0, .4, 0, 0, 0, 1, 0, 0, -.3],
            [0, .1, 0, -.35, 0, 0, 0, 1, 0, 0],
            [0, 0, .05, 0, .1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, -.3, 0, 0, 1]
        ]
    )

    nobs = 300
    d = cov.shape[0]
    np.random.seed(42)
    Z = np.random.multivariate_normal(np.repeat(0, d), cov, size=nobs)
    X = scipy.stats.norm.cdf(Z)
    y = _noisy_morokoff(X, d, noisy)
    return X, y


def get_california(n="all"):
    data = fetch_california_housing()
    X = data["data"]
    y = data["target"]

    if n == "all":
        return X, y
    elif isinstance(n, int):
        np.random.seed(42)
        indices = np.random.choice(len(X), size=n, replace=False)
        return X[indices], y[indices]
    elif isinstance(n, float):
        if n <= 1:
            np.random.seed(40)
            indices = np.random.choice(len(X), size=int(n * len(X)), replace=False)
            return X[indices], y[indices]
        else:
            raise ValueError("If n is a float, it should be less or equal to 1.")
    else:
        raise ValueError("n must be either equal to 'all', an int or a float")
