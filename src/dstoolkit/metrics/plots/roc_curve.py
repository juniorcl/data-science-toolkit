import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_curve(y_true, y_score, ax=None):
    """
    Plot the ROC curve for a binary classification model.

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

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        alpha=0.6,
        label="Random",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    return fig, ax