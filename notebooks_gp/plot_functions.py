import numpy as np
import scipy
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

EDF_BLUE = np.array([[26, 54, 105]]) / 255
EDF_ORANGE = np.array([[223, 84, 49]]) / 255


def plot_prediction_intervals(
    title,
    ax,
    y_test,
    y_pred,
    intervals,
    n_points_plot=100,
    fs=15
):
    """
    Plot of the prediction intervals for each different conformal
    method.
    """
    np.random.seed(42)
    points_idx = np.random.choice(range(len(y_test)), n_points_plot, replace=False)
    y_test = y_test[points_idx]
    y_pred = y_pred[points_idx]
    intervals = intervals[points_idx]
    sorted_index = np.argsort(y_test)
    y_test = y_test[sorted_index]

    y_pred = y_pred[sorted_index]
    intervals = intervals[sorted_index]

    lower_bound = intervals[:, 0]
    upper_bound = intervals[:, 1]

    warning1 = y_test > upper_bound
    warning2 = y_test < lower_bound

    warnings = warning1 + warning2

    intervals[:, 0] = np.abs(y_pred - intervals[:, 0])
    intervals[:, 1] = np.abs(y_pred - intervals[:, 1])

    ax.errorbar(
        y_test[~warnings],
        y_pred[~warnings],
        yerr=intervals[~warnings, :].T,
        capsize=5, marker="o", elinewidth=2, linewidth=0,
        c=EDF_BLUE,
        label="Inside prediction interval"
        )
    ax.errorbar(
        y_test[warnings],
        y_pred[warnings],
        yerr=intervals[warnings, :].T,
        capsize=5, marker="o", elinewidth=2, linewidth=0, color=EDF_ORANGE,
        label="Outside prediction interval"
        )
    ax.axis("equal")
    ax.set_xlim(min(y_test), max(y_test))
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, '--', alpha=0.75, color="black", label="x=y")
    ax.set_xlabel("$y_{true}$", fontsize=fs)
    ax.set_ylabel("$y_{pred}$", fontsize=fs)
    ax.set_title((title))


def plot_distribution(distribution, model_name, xlabel, ax):
    sns.histplot(
        distribution, kde=True,
        bins=100, color='darkblue', ax=ax
    )
    ax.set_xlabel(xlabel, size=15)
    ax.set_title(model_name)
    ax.set_xlim(0, max(1, max(distribution)))


def get_max_width(models, index_confidence):
    max_width = 0
    for model in models.values():
        max_width = max(max_width, model["width"][index_confidence])
    return max_width


def plot_width_error(model, model_name, ax, r, c, max_width=None, y=None, plot_tree=True, plot_rank=True):
    if y is not None:
        error = model["errors"] / y
    else:
        error = model["errors"]
    if plot_rank:
        ax[r, c].scatter(
            scipy.stats.rankdata(model["width"]),
            scipy.stats.rankdata(error),
            s=10,
            c=model["color"]
        )
    else:
        ax[r, c].scatter(
            model["width"],
            error,
            s=10,
            c=model["color"]
        )
    if max_width:
        ax[r, c].set_xlim(0, max_width * 1.1)
    ci_pearson_correlation = model["pearson_correlation_to_error"].confidence_interval
    mean_pearson_correlation = np.mean(model["pearson_correlation_to_error"].bootstrap_distribution)
    ci_spearman_correlation = model["spearman_correlation_to_error"].confidence_interval
    mean_spearman_correlation = np.mean(model["spearman_correlation_to_error"].bootstrap_distribution)
    ci_mse = model["tree_mse"].confidence_interval
    mean_mse = model["tree_mse"].bootstrap_distribution.mean()
    ax[r, c].set_xlabel("Width of the prediction interval\n", size=15)
    ax[r, 0].set_ylabel("Relative error of the metamodel", size=15)
    ax[r, c].set_title(
        model_name + "\n" +
        r"$\bf{Pearson correlation\in [}$" + r"$\bf{" + str(round(ci_pearson_correlation[0], 2)) +
        ", " + str(round(ci_pearson_correlation[1], 2)) + "], mean = " +
        str(round(mean_pearson_correlation, 2)) + "}$" + "\n" +

        r"$\bf{Spearman correlation\in [}$" + r"$\bf{" + str(round(ci_spearman_correlation[0], 2)) +
        ", " + str(round(ci_spearman_correlation[1], 2)) + "], mean = "
        + str(round(mean_spearman_correlation, 2)) + "}$" + "\n" +

        r"$\bf{MSE\in [}$" + r"$\bf{" + str(round(ci_mse[0], 2)) +
        ", " + str(round(ci_mse[1], 2)) + "], mean = " + str(round(mean_mse, 2)) + "}$" + "\n" +

        r"$\bf{Global MSE =" + str(round(np.mean(model["errors"] ** 2), 2)) + "}$"
    )
    if plot_tree:
        _plot_tree(model, r, c, ax)


def _plot_tree(model, r, c, ax):
    X_temp = model["width"]
    y_temp = model["errors"]
    X_temp_train, _, y_temp_train, _ = train_test_split(X_temp, y_temp, test_size=.3)
    tree = DecisionTreeRegressor(min_samples_split=3, max_depth=3).fit(X_temp_train.reshape(-1, 1), y_temp_train)
    pred = tree.predict(np.arange(0, X_temp.max(), .1).reshape(-1, 1))
    ax[r, c].plot(np.arange(0, X_temp.max(), .1), pred)
