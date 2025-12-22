import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_precision_recall_curve(y, target, prob_col="prob"):
    """
    Plots the Precision-Recall curve for the given true labels and predicted probabilities.

    It visualizes the trade-off between precision and recall for different probability thresholds.

    Parameters
    ----------
    y : DataFrame
        The input data containing true labels and predicted probabilities.
    target : str
        The name of the target column in the DataFrame.
    prob_col : str, optional
        The name of the column containing predicted probabilities (default is "prob").

    Returns
    -------
    None
        This function does not return a value.
    
    Raises
    ------
    ValueError
        If the target or prob_col does not exist in the DataFrame.
    """
    precision, recall, _ = precision_recall_curve(y[target], y[prob_col])
    
    plt.plot(precision, recall)
    plt.title('Precision Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.show()