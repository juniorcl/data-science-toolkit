import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc_curve(y, target, prob_col="prob"):
    """
    Plot the ROC AUC curve for given true labels and predicted probabilities.

    This function computes the ROC curve and AUC for the specified true labels
    and predicted probabilities, and visualizes the results using a line plot.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame containing true labels and predicted probabilities.
    prob_col : str
        Name of the column in `y` that contains predicted probabilities.
    target : str
        Name of the column in `y` that contains true labels.

    Returns
    -------
    None
        This function does not return a value.
        It displays a ROC AUC curve plot.

    Raises
    ------
    KeyError
        If `prob_col` or `target` are not found in the DataFrame `y`.
    """
    fpr, tpr, _ = roc_curve(y[target], y[prob_col])
    plt.plot(fpr, tpr)
    plt.title('ROC AUC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()