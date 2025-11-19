import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y, target, n_bins=10, strategy='uniform'):
    """
    Plots the calibration curve for a binary classifier.

    This function visualizes the relationship between predicted probabilities and observed frequencies.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame containing the true labels and predicted probabilities.
    target : str
        The name of the column in `y` that contains the true binary labels.
    n_bins : int, optional
        Number of bins to use for the calibration curve (default is 10).
    strategy : {'uniform', 'quantile'}, optional
        Strategy to define the bin edges (default is 'uniform').
    
    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """
    prob_true, prob_pred = calibration_curve(y[target], y['prob'], n_bins=n_bins, strategy=strategy)
    
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()