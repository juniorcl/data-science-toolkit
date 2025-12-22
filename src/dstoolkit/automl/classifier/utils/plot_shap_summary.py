import shap


def plot_shap_summary(model, X):
    """
    Plot SHAP summary for the given model and dataset.

    This function computes SHAP values for the provided model and dataset,
    and visualizes the feature importance using a summary plot.

    Parameters
    ----------
    model : object
        The trained model for which to plot SHAP values.
    X : pd.DataFrame
        The input data for which to compute SHAP values.

    Returns
    -------
    None
        This function does not return a value.
        It displays a SHAP summary plot.

    Raises
    ------
    None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)