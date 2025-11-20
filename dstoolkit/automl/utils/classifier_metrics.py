from .ks_score import ks_score
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

def get_classifier_metrics(y, target, pred_col="pred", prob_col="prob"):
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

    Raises
    ------
    ValueError
        If the input data is not valid for classification.
    """
    y_true = y[target]
    y_pred = y[pred_col]
    y_prob = y[prob_col]
    
    return {
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'KS': ks_score(y_true, y_prob),
        'Brier': brier_score_loss(y_true, y_prob),
        'LogLoss': log_loss(y_true, y_prob)
    }