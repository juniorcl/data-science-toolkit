import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_permutation_importance(model, X, y, scoring, n_repeats=5, random_state=42, figsize=None, ax=None):
    """
    Plot permutation feature importance for a fitted model.

    This function computes permutation importance by randomly shuffling
    each feature and measuring the decrease in model performance. The
    distribution of importance values across multiple repeats is shown
    using box plots.

    Parameters
    ----------
    model : estimator object
        A fitted scikit-learn compatible estimator.

    X : pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix used for computing permutation importance.

    y : array-like of shape (n_samples,)
        True target values corresponding to `X`.

    scoring : str or callable
        Scoring metric used to evaluate the decrease in model performance
        (e.g., 'accuracy', 'roc_auc').

    n_repeats : int, default=5
        Number of times a feature is randomly permuted.

    random_state : int, default=42
        Random seed for reproducibility.

    figsize : tuple, optional
        Size of the matplotlib figure. Used only if `ax` is None.
        If None, the height is automatically adjusted based on the number
        of features.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If None, a new figure and axes
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the plot.

    Notes
    -----
    - Permutation importance is model-agnostic and can be applied to any
      fitted estimator.
    - Importance values reflect the decrease in the chosen scoring metric
      after permuting a feature.
    - Highly correlated features may share importance, leading to lower
      individual importance values.
    """

    # --- Compute permutation importance ---
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    # --- Sort features by mean importance ---
    sorted_idx = np.argsort(result.importances_mean)

    importances = result.importances[sorted_idx]
    feature_names = X.columns[sorted_idx]

    # --- Axes / Figure ---
    if ax is None:
        if figsize is None:
            figsize = (8, max(4, len(feature_names) * 0.4))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- Boxplot ---
    ax.boxplot(
        importances.T,
        vert=False,
        labels=feature_names,
        patch_artist=True,
        boxprops=dict(facecolor="tab:blue", alpha=0.6),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel(f"Decrease in {scoring}")
    ax.set_title("Permutation Feature Importance")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    return fig, ax