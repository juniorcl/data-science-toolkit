import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_curve(y_true, y_score, figsize=(8, 5)):
    """
    Plot the ROC curve for a binary classification model without using
    sklearn's RocCurveDisplay.

    The function computes the False Positive Rate (FPR), True Positive Rate (TPR)
    and the ROC AUC, and plots the ROC curve with consistent styling and size.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).

    y_score : array-like of shape (n_samples,)
        Predicted probabilities or decision scores for the positive class.

    figsize : tuple, default=(8, 5)
        Size of the matplotlib figure.

    Returns
    -------
    float
        ROC AUC score.
    """

    # Safety check
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary (0/1).")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Area under ROC curve
    auc = roc_auc_score(y_true, y_score)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}",
        color="tab:blue",
        linewidth=2,
    )

    # Baseline (random classifier)
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        alpha=0.6,
        label="Random",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return auc