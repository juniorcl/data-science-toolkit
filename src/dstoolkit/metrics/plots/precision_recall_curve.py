import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_precision_recall_curve(y_true, y_score, ax=None):
    """
    Plot the Precision–Recall curve for a binary classification model.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).

    y_score : array-like of shape (n_samples,)
        Predicted probabilities or decision scores for the positive class.

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
        If y_true is not binary.
    """
    y_true = np.asarray(y_true)

    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary (0/1).")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {ap:.3f}",
    )

    baseline = np.mean(y_true)
    ax.hlines(
        baseline,
        xmin=0,
        xmax=1,
        linestyles="--",
        alpha=0.6,
        label=f"Baseline = {baseline:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    return fig, ax