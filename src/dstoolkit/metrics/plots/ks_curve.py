import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def plot_ks_curve(y_true, y_score, ax=None):
    """
    Plot the KS (Kolmogorov–Smirnov) cumulative distribution curves for a
    binary classification model using predicted probabilities.

    The function computes the KS statistic using the two-sample KS test
    from `scipy.stats.ks_2samp`, comparing the distribution of predicted
    probabilities for the positive and negative classes. It also plots the
    cumulative distributions and visually highlights the KS distance.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).

    y_score : array-like
        Predicted probabilities for the positive class.

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
        If `y_true` is not binary.
    """
    y_true = pd.Series(y_true)
    y_score = pd.Series(y_score)

    if set(y_true.unique()) - {0, 1}:
        raise ValueError("y_true must be binary (0/1).")

    # Separate predicted probabilities by class
    scores_1 = y_score[y_true == 1]
    scores_0 = y_score[y_true == 0]

    # Compute KS statistic
    ks_value, _ = ks_2samp(scores_1, scores_0)

    # Prepare cumulative distributions
    df = pd.DataFrame({"target": y_true, "score": y_score}).sort_values("score")

    total_1 = (df["target"] == 1).sum()
    total_0 = (df["target"] == 0).sum()

    df["cum_1"] = (df["target"] == 1).cumsum() / total_1
    df["cum_0"] = (df["target"] == 0).cumsum() / total_0

    # Point of maximum difference (KS threshold)
    df["diff"] = (df["cum_1"] - df["cum_0"]).abs()
    ks_idx = df["diff"].idxmax()
    ks_threshold = df.loc[ks_idx, "score"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Plot CDFs
    ax.plot(
        df["score"],
        df["cum_1"],
        label="Class 1 (CDF)",
    )
    ax.plot(
        df["score"],
        df["cum_0"],
        label="Class 0 (CDF)",
    )

    # KS vertical line
    ax.vlines(
        ks_threshold,
        ymin=df.loc[ks_idx, "cum_0"],
        ymax=df.loc[ks_idx, "cum_1"],
        linestyles="--",
        label=f"KS = {ks_value:.3f}",
    )

    ax.set_title("Kolmogorov–Smirnov Curve")
    ax.set_xlabel("Predicted Probability of Class 1")
    ax.set_ylabel("Cumulative Proportion")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    return fig, ax