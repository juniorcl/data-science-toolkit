import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_waste_distribution(y_true, y_score, ax=None):
    """
    Plot the distribution of residuals (waste) for regression models.

    The residuals are computed as (y_true - y_score) and visualized using
    a histogram with an optional kernel density estimate (KDE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_score : array-like of shape (n_samples,)
        Predicted target values.

    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    residuals = y_true - y_score

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sns.histplot(
        residuals,
        kde=True,
        ax=ax,
    )

    ax.set_title("Residual (Waste) Distribution")
    ax.set_xlabel("Residual (y_true − y_pred)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

    return fig, ax