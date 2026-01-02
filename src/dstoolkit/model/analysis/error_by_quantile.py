import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_error_by_quantile(y_true, y_score, q=5, ax=None):
    """
    Plot the absolute prediction error grouped by target quantiles.

    The target variable is split into quantiles, and the absolute
    prediction error is visualized for each quantile using a boxplot.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_score : array-like
        Predicted target values.
    q : int, default=5
        Number of quantiles to compute.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.

    Raises
    ------
    ValueError
        If y_true and y_score have different lengths.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length.")

    df = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_score
    })

    df["quantile"] = pd.qcut(df["y_true"], q=q, duplicates="drop")
    df["abs_error"] = np.abs(df["y_true"] - df["y_score"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sns.boxplot(
        x="quantile",
        y="abs_error",
        data=df,
        ax=ax
    )

    ax.set_title("Absolute Error by Target Quantile")
    ax.set_xlabel("Target Quantile")
    ax.set_ylabel("Absolute Error")
    ax.tick_params(axis="x", rotation=45)

    return fig, ax