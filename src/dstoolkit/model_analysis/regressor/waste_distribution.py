import seaborn as sns
import matplotlib.pyplot as plt


def plot_waste_distribution(y, target, pred_col="pred"):
    """
    Plot the distribution of residuals for given true labels and predicted values.

    This function computes the residuals (differences between true and predicted values)
    and visualizes their distribution using a histogram.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame containing true labels and predicted values.
    pred_col : str
        Name of the column in `y` that contains predicted values.
    target : str
        Name of the column in `y` that contains true labels.

    Returns
    -------
    None
        This function does not return a value.
        It displays a residuals plot.
    
    Raises
    ------
    KeyError
        If `pred_col` or `target` are not found in the DataFrame `y`.
    """
    residuals = y[target] - y[pred_col]
    sns.histplot(residuals, kde=True)
    plt.title("Waste Distribution")
    plt.xlabel("Erro (y_true - y_pred)")
    plt.show()