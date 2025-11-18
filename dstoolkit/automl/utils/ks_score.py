from scipy.stats import ks_2samp

def ks_score(y_true, y_prob):
    pos_prob = y_prob[y_true == 1]
    neg_prob = y_prob[y_true == 0]
    if len(pos_prob) == 0 or len(neg_prob) == 0:
        return 0.0
    result = ks_2samp(pos_prob, neg_prob)
    return result.statistic