import numpy as np
import matplotlib.pyplot as plt


def plot_lift_score(y_true, y_score, n_bins=10, ax=None):
    """
    Plots a bar chart showing the Lift Score partitioned into custom ranges (bins).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0, 1 or False, True).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities or decision scores for the positive class.
    n_bins : int, optional
        Number of ranges/bins to split the population (e.g., 10 for deciles, 5 for quintiles). Default is 10.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true variable must contain only 0 or 1 values.")

    idx = np.argsort(y_score)[::-1] # invert
    y_true_sorted = y_true[idx]
    
    baseline = np.mean(y_true)

    bins = np.array_split(y_true_sorted, n_bins)
    
    lift_per_bin = [(np.mean(b) / baseline) for b in bins]
    
    lift_score_bin_1 = lift_per_bin[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    
    else:
        fig = ax.figure

    range_labels = [f"{i+1}º" for i in range(n_bins)]
    x_positions = np.arange(n_bins)

    ax.bar(
        x_positions,
        lift_per_bin,
        width=0.75,
        color="#4682B4",
        edgecolor="black",
        linewidth=0.5,
        label=f"ML Model (Lift at first group = {lift_score_bin_1:.3f}x)"
    )

    ax.axhline(
        y=1.0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Baseline (Random = 1.000)"
    )

    ax.set_title(f"Lift Score per Range ({n_bins} bins)", fontsize=14, pad=12)
    ax.set_xlabel(f"Population Split in {n_bins} Parts (Sorted by Score)", fontsize=11)
    ax.set_ylabel("Lift Score (by range)", fontsize=11)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(range_labels)
    
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    
    max_lift = max(lift_per_bin)
    ax.set_ylim(0, max(max_lift * 1.2, 2.0))

    return fig, ax