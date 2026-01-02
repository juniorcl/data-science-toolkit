import shap
import matplotlib.pyplot as plt


def plot_shap_linear_summary(model, X, max_display=20, ax=None):
    """
    Plot SHAP summary for linear models.

    This function computes SHAP values using a LinearExplainer and
    visualizes feature contributions for linear models such as
    LogisticRegression, LinearRegression, Ridge, Lasso, and ElasticNet.

    Parameters
    ----------
    model : estimator object
        A fitted linear model compatible with SHAP's LinearExplainer.

    X : pandas.DataFrame
        Input data used to compute SHAP values.

    max_display : int, default=20
        Maximum number of features to display in the summary plot.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.

    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the plot.

    Notes
    -----
    - For standardized features, SHAP values are directly comparable.
    - For non-standardized features, interpretation depends on feature scale.
    """

    # SHAP explainer
    explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
    shap_values = explainer.shap_values(X)

    # Axes handling
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, max_display * 0.35)))
    else:
        fig = ax.figure

    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False
    )

    fig.tight_layout()

    return fig, ax