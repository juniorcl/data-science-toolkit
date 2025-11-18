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

def get_classifier_metrics(y, pred_col, prob_col, target):
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