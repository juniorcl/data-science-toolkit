import seaborn as sns
import matplotlib.pyplot as plt


def plot_pred_vs_true(y, target, pred_col="pred"):
    """
    Plots predicted values against true values using a scatter plot.

    This function creates a scatter plot to visualize the relationship
    between true and predicted values, helping to assess the model's
    performance.

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
        It displays a scatter plot.
    
    Raises
    ------
    KeyError
        If `pred_col` or `target` are not found in the DataFrame `y
    """
    sns.scatterplot(x=y[target], y=y[pred_col])
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], '--r')
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("y_true vs y_pred")
    plt.show()