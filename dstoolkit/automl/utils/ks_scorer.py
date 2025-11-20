from .ks_score import ks_score
from sklearn.metrics import make_scorer


ks_scorer = make_scorer(ks_score, greater_is_better=True, response_method='predict_proba')