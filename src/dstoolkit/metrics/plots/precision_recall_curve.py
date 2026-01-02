import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_precision_recall_curve(y_true, y_score, figsize=(8, 5)):
    """
    Plot the Precision–Recall curve for a binary classification model
    without using sklearn's PrecisionRecallDisplay.

    The function computes precision, recall and the Average Precision (AP),
    and plots the Precision–Recall curve with consistent styling and size.

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
        Average Precision (area under the Precision–Recall curve).
    """

    # Safety check
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true must be binary (0/1).")

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # Area under PR curve
    ap = average_precision_score(y_true, y_score)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(
        recall,
        precision,
        label=f"AP = {ap:.3f}",
        color="tab:blue",
        linewidth=2,
    )

    # Baseline (prevalence)
    baseline = np.mean(y_true)
    plt.hlines(
        baseline,
        xmin=0,
        xmax=1,
        colors="gray",
        linestyles="--",
        alpha=0.6,
        label=f"Baseline = {baseline:.3f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()