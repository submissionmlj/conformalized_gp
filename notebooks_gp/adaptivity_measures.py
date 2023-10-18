import numpy as np
import openturns as ot
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr, permutation_test
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def hsic_ot(errors, widths):
    X = ot.Sample(errors[:, np.newaxis])
    Y = ot.Sample(widths[:, np.newaxis])

    covarianceModelCollection = []
    i = 0
    Xi = X.getMarginal(i)
    Cov = ot.SquaredExponential(1)
    Cov.setScale(Xi.computeStandardDeviation())
    covarianceModelCollection.append(Cov)
    covarianceModelCollection.append(ot.SquaredExponential(Y.computeStandardDeviation()))
    estimatorType_v = ot.HSICVStat()
    hsic_v = ot.HSICEstimatorGlobalSensitivity(covarianceModelCollection, X, Y, estimatorType_v)

    return hsic_v.getR2HSICIndices()[0], hsic_v.getPValuesAsymptotic()[0], hsic_v.getPValuesPermutation()[0]


def bootstrap_correlation(data1, data2, num_iterations=1000):
    correlations = []
    p_values = []
    n = len(data1)

    for _ in range(num_iterations):
        # Randomly sample with replacement
        sample_indices = np.random.choice(n, n, replace=True)
        sampled_data1 = data1[sample_indices]
        sampled_data2 = data2[sample_indices]

        # Calculate correlation coefficient (Pearson's correlation)
        corr, p_value = pearsonr(sampled_data1, sampled_data2)
        correlations.append(corr)
        p_values.append(p_value)

    return correlations, p_values


def my_correlation(data, *args, **kwargs):
    data_split = [i.split('--') for i in data]
    data_float = np.array(data_split).astype(float)
    return pearsonr(data_float[:, 0], data_float[:, 1]).correlation


def spline_mse(data, *args, **kwargs):
    data_split = [i.split('--') for i in data]
    data_float = np.array(data_split).astype(float)
    x, y = data_float[:, 0], data_float[:, 1]
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.2)
    _, unique_index = np.unique(x_t, return_index=True)
    x_t = x_t[unique_index]
    y_t = y_t[unique_index]
    x_t = np.sort(x_t)
    y_t = y_t[np.argsort(x_t)]
    cs = CubicSpline(x_t, y_t)
    y_pred = cs(x_v)

    return mean_squared_error(y_v, y_pred)


def tree_mse(data, *args, **kwargs):
    data_split = [i.split('--') for i in data]
    data_float = np.array(data_split).astype(float)
    x, y = data_float[:, 0], data_float[:, 1]
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=.2)
    tree = DecisionTreeRegressor(min_samples_split=10, max_depth=3).fit(x_t.reshape(-1, 1), y_t)
    y_pred = tree.predict(x_v.reshape(-1, 1))

    return mean_squared_error(y_v, y_pred)


def statistic(x, y):
    return np.mean(x) - np.mean(y)


def recap_permutation_test(models, name, alternative):
    gp = models["GP"][name].bootstrap_distribution
    cv_plus_gp = models["CV+GP"][name].bootstrap_distribution
    cv_minmax_gp = models["CV-minmax-GP"][name].bootstrap_distribution
    cv_plus = models["CV+"][name].bootstrap_distribution
    stat_gp_vs_plus_gp = permutation_test(
        (gp, cv_plus_gp), statistic, vectorized=False,
        alternative=alternative
    )

    stat_gp_vs_minmax = permutation_test(
        (gp, cv_minmax_gp), statistic, vectorized=False,
        alternative=alternative
    )
    stat_gp_vs_cv_plus = permutation_test(
        (gp, cv_plus), statistic, vectorized=False,
        alternative=alternative
    )
    stat_plus_vs_gp_plus = permutation_test(
        (cv_plus, cv_plus_gp), statistic, vectorized=False,
        alternative=alternative
    )
    stat_plus_vs_minmax = permutation_test(
        (cv_plus_gp, cv_minmax_gp), statistic, vectorized=False,
        alternative=alternative
    )
    recap_permutation_test = {
        "gp_vs_gp_plus": {
            "test statistic": stat_gp_vs_plus_gp.statistic,
            "p-value": stat_gp_vs_plus_gp.pvalue
        },
        "gp_vs_cv_plus": {
            "test statistic": stat_gp_vs_cv_plus.statistic,
            "p-value": stat_gp_vs_cv_plus.pvalue
        },
        "cv_plus_vs_gp_plus": {
            "test statistic": stat_plus_vs_gp_plus.statistic,
            "p-value": stat_plus_vs_gp_plus.pvalue
        },
        "gp_vs_gp_minmax": {
            "test statistic": stat_gp_vs_minmax.statistic,
            "p-value": stat_gp_vs_minmax.pvalue
        },
        "gp_plus_vs_gp_minmax": {
            "test statistic": stat_plus_vs_minmax.statistic,
            "p-value": stat_plus_vs_minmax.pvalue
        }
    }

    return pd.DataFrame(recap_permutation_test).T


def q2(y, y_pred):
    return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
