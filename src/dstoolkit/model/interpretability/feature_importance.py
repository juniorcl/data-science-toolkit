import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(model, top_n=20, figsize=None, ax=None):
    """
    Plot feature importance for tree-based models.

    This function visualizes the most important features of a fitted
    tree-based model (e.g., LightGBM, XGBoost, CatBoost, RandomForest),
    helping to understand which variables contribute most to the model's
    predictions.

    Parameters
    ----------
    model : estimator object
        A fitted model that exposes feature importance information through
        either a `feature_importances_` attribute or a `get_score` method
        (XGBoost native models).

    top_n : int, default=20
        Number of top features to display, ordered by importance.

    figsize : tuple, optional
        Size of the matplotlib figure. Used only if `ax` is None.
        If None, the height is automatically adjusted based on `top_n`.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If None, a new figure and axes
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the plot.

    Raises
    ------
    AttributeError
        If feature names or feature importance values cannot be inferred
        from the model.

    Notes
    -----
    - Feature importances are model-specific and should not be interpreted
      as causal effects.
    - Importance values from different models or training runs are not
      directly comparable.
    """

    # --- Feature names ---
    if hasattr(model, "feature_name_"):
        feature_names = model.feature_name_
    elif hasattr(model, "feature_names_"):
        feature_names = model.feature_names_
    elif hasattr(model, "feature_names_in_"):
        feature_names = model.feature_names_in_
    else:
        raise AttributeError("Unable to infer feature names from the model.")

    # --- Feature importances ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_score"):  # XGBoost native API
        importance_dict = model.get_score(importance_type="weight")
        feature_names = np.array(list(importance_dict.keys()))
        importances = np.array(list(importance_dict.values()))
    else:
        raise AttributeError("The provided model does not expose feature importance information.")

    # --- Prepare data ---
    df_imp = (
        pd.DataFrame(
            {
                "Feature": feature_names, 
                "Importance": importances
            }
        )
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    # --- Axes / Figure ---
    if ax is None:
        if figsize is None:
            figsize = (8, max(4, top_n * 0.4))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- Plot ---
    ax.barh(
        df_imp["Feature"][::-1],
        df_imp["Importance"][::-1],
        color="tab:blue",
        alpha=0.85,
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    return fig, ax