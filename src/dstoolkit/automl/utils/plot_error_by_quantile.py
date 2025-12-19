import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_error_by_quantile(y, target, pred_col="pred"):
    """
    Plots the absolute error of predictions against quantiles of the target variable.

    Each quantile is represented on the x-axis, while the absolute error is on the y-axis.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame containing the target variable and predictions.
    pred_col : str
        Name of the column containing the predictions.
    target : str
        Name of the column containing the true target values.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the input DataFrame is invalid or if the plotting fails.
    """
    y_copy = y.copy()
    y_copy["quantile"] = pd.qcut(y_copy[target], q=5)
    y_copy["abs_error"] = abs(y_copy[target] - y_copy[pred_col])
    
    sns.boxplot(x="quantile", y="abs_error", data=y_copy)
    plt.title("Absolute Error by Target Quantile")
    plt.xticks(rotation=45)
    plt.show()