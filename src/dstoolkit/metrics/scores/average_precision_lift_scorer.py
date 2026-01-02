from .average_precision_lift_score import average_precision_lift_score
from sklearn.metrics import make_scorer


average_precision_lift_scorer = make_scorer(
    average_precision_lift_score, greater_is_better=True, response_method='predict_proba')