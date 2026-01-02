import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_true_vs_pred(y_true, y_score, ax=None):
    """
    Plot predicted values against true values using a scatter plot.

    This visualization helps assess model fit and bias by comparing
    predictions to the ideal y = x reference line.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_score : array-like
        Predicted target values.
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sns.scatterplot(x=y_true, y=y_score, ax=ax)

    min_val = min(y_true.min(), y_score.min())
    max_val = max(y_true.max(), y_score.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="red"
    )

    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title("y_true vs y_pred")

    return fig, ax