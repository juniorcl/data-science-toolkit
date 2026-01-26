from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .average_precision_lift_score import average_precision_lift_score
from .ks_score import ks_score


def get_classifier_metrics(y_true, y_pred, y_score):
    """
    Calculate a set of classification metrics.

    This function computes various classification metrics based on the true and predicted values.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame containing the true labels and predictions.
    pred_col : str
        Name of the column with predicted class labels.
    prob_col : str
        Name of the column with predicted probabilities for the positive class.
    target : str
        Name of the column with true class labels.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics:
        - 'Balanced Accuracy'
        - 'Precision'
        - 'Recall'
        - 'F1'
        - 'AUC'
        - 'KS'
        - 'Brier'
        - 'LogLoss'
        - 'Average Precision Lift'

    Raises
    ------
    ValueError
        If the input data is not valid for classification.
    """

    return {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "KS": ks_score(y_true, y_score),
        "Brier": brier_score_loss(y_true, y_score),
        "LogLoss": log_loss(y_true, y_score),
        "Avg Precision Lift": average_precision_lift_score(y_true, y_score),
    }
