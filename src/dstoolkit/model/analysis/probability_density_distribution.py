import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_probability_density_distribution(y_true, y_score, ax=None):
    """
    Plots the KDE and histogram distribution of predicted probabilities
    separated by the true target class (0 vs 1), matching the style of image_1.png.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Input validation
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("y_true variable must contain only 0 or 1 values.")

    # Figure and Axes setup
    if ax is None:
        # Match the aspect ratio and size of the reference image
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    else:
        fig = ax.figure

    # --- Match the visual style of image_1.png ---
    # 1. Colors using standard Seaborn cycle
    colors = {0: "C0", 1: "C1" }
    labels = {0: "Target 0", 1: "Target 1"}

    # 2. Main Plotting Loop (Histogram + KDE)
    for target_val in [1, 0]: # Plot Target 1 first so Target 0 sits on top
        mask = (y_true == target_val)
        scores_target = y_score[mask]
        
        if len(scores_target) == 0:
            continue

        # Plot Histogram + Smoothed KDE, adding classic pyplot style details
        sns.histplot(
            x=scores_target,
            kde=True,
            ax=ax,
            color=colors[target_val],
            label=f"{labels[target_val]} (n={len(scores_target):,})",
            stat="density",       # Area under curve is 1
            common_norm=False,    # Classes don't flatten each other
            alpha=0.6,            # Set transparency to match the overlay feel
            edgecolor="black",    # Add sharp black outlines like in image_1.png
            linewidth=0.5,        # Thin outlines
            line_kws={"linewidth": 2.5, "color": colors[target_val]}  # Use specific class color for the KDE line
        )

    # --- Match the specific chart labels and style ---
    ax.set_title("Probability Distribution per Target Class", fontsize=16, fontweight='normal', pad=15)
    ax.set_xlabel("Predicted Probability (Score)", fontsize=14)
    ax.set_ylabel("Density (by range)", fontsize=14)
    
    # 3. Axis Limits and Ticks
    ax.set_xlim(0, 1) # Standard probability scope
    # Ticks for standard 0.0 to 1.0 probability range in steps of 0.1
    xticks = np.arange(0.0, 1.1, 0.1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks], fontsize=12)
    
    # Increase the font size for Y-axis ticks
    ax.tick_params(axis='y', labelsize=12)

    # 4. Specific grid lines like in the reference image
    ax.grid(axis='both', which='major', color='#E0E0E0', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True) # Ensure grid is behind the bars

    # 5. Legend setup with 'best' location
    ax.legend(
        loc="best",
        fontsize=11, 
        frameon=True, 
        framealpha=1, 
        edgecolor="gray"
    )

    # 6. Sharp frame/border for the entire plot
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    return fig, ax