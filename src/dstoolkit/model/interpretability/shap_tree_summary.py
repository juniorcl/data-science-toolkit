import shap
import matplotlib.pyplot as plt


def plot_shap_tree_summary(model, X, figsize=(8, 5), ax=None):
    """
    Plot a SHAP summary plot for tree-based models.

    This function computes SHAP values using a TreeExplainer and displays
    a summary plot showing the global feature importance and the direction
    of each feature's impact on the model predictions.

    Parameters
    ----------
    model : estimator object
        A fitted tree-based model compatible with SHAP's TreeExplainer
        (e.g., LightGBM, XGBoost, CatBoost, RandomForest).

    X : pandas.DataFrame of shape (n_samples, n_features)
        Input data used to compute SHAP values.

    figsize : tuple, default=(8, 5)
        Size of the matplotlib figure. Used only if `ax` is None.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If None, a new figure and axes
        are created internally.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the SHAP summary plot.

    Notes
    -----
    - SHAP summary plots are global explanations and reflect the average
      impact of each feature across the dataset.
    - Feature importance values are expressed in SHAP units (impact on
      the model output).
    - For multiclass models, SHAP returns one explanation per class.
    """

    # --- Create explainer and compute SHAP values ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # --- Create axes if needed ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        plt.sca(ax)  # set current axes for SHAP

    # --- SHAP summary plot (matplotlib backend) ---
    shap.summary_plot(
        shap_values,
        X,
        show=False,
    )

    fig.tight_layout()

    return fig, ax