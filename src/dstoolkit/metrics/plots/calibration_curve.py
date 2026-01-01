import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_score, n_bins=10, strategy='uniform', model_name='Model', figsize=(8, 5)):
    """
    Plots the calibration curve for a binary classifier and displays
    the Brier Score on the plot.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_score : array-like
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Binning strategy.
    model_name : str, default='Model'
        Name to show in the legend.
    figsize : tuple, default=(8, 5)
        Figure size.
    """

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy=strategy,
    )

    brier = brier_score_loss(y_true, y_score)

    plt.figure(figsize=figsize)

    plt.plot(
        prob_pred,
        prob_true,
        marker='o',
        linewidth=2,
        label=f"{model_name} (Brier = {brier:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='gray',
        label='Perfectly calibrated',
    )

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()