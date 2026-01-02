import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_score, n_bins=10, strategy="uniform", model_name="Model", ax=None):
    """
    Plot the calibration curve for a binary classifier and display
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

    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    prob_true, prob_pred = calibration_curve(
        y_true=y_true,
        y_prob=y_score,
        n_bins=n_bins,
        strategy=strategy,
    )

    brier = brier_score_loss(y_true, y_score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=2,
        label=f"{model_name} (Brier = {brier:.3f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfectly calibrated",
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    return fig, ax