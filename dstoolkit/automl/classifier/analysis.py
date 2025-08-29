import shap

import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats             import ks_2samp
from sklearn.metrics         import roc_curve, precision_recall_curve
from sklearn.inspection      import permutation_importance
from sklearn.calibration     import calibration_curve
from sklearn.model_selection import learning_curve, StratifiedKFold

from sklearn.metrics import (
    f1_score, 
    make_scorer,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)


def plot_permutation_importance(model, features, X, y, scoring):

    permu_results = permutation_importance(model, X[features], y, scoring=scoring, n_repeats=5, random_state=42)

    sorted_importances_idx = permu_results.importances_mean.argsort()
    
    df_results = pd.DataFrame(permu_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    
    ax = df_results.plot.box(vert=False, whis=10, patch_artist=True, boxprops={'facecolor':'skyblue', 'color':'blue'})
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel(f"Decrease in {scoring}")

    plt.show()

def plot_feature_importance(model):

    feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else model.feature_names_ # for catboost
    
    df_imp = pd.DataFrame(model.feature_importances_, feature_names).reset_index()
    df_imp.columns = ["Variable", "Importance"]
    df_imp = df_imp.sort_values("Importance", ascending=False)

    sns.barplot(x="Importance", y="Variable", color="#006e9cff", data=df_imp[:20])

    plt.title(f"Importance of Variables")
    plt.show()

def plot_shap_summary(model, features, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[features])
    
    shap.summary_plot(shap_values, X[features])

def plot_residuals(y, pred_col, target):
        
    residuals = y[target] - y[pred_col]
    
    sns.histplot(residuals, kde=True)
    
    plt.title("Waste Distribution")
    plt.xlabel("Erro (y_true - y_pred)")
    plt.show()

def plot_roc_curve(y, prob_col, target):
    
    fpr, tpr, _ = roc_curve(y[target], y[prob_col])

    plt.plot(fpr, tpr)
    plt.title('ROC AUC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()

def plot_precision_recall_curve(y, prob_col, target):

    precision, recall, _ = precision_recall_curve(y[target], y[prob_col])

    plt.plot(precision, recall)
    plt.title('Precision Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.show()

def plot_learning_curve(model, X, y, target, scoring, cv=5):

    splitter = StratifiedKFold(cv, shuffle=True, random_state=42)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y[target],
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=splitter,
        scoring=scoring
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Train')
    plt.fill_between(
        train_sizes_abs, train_scores_mean - train_scores_std, 
        train_scores_mean + train_scores_std, alpha=0.1, color='r')

    plt.plot(train_sizes_abs, val_scores_mean, 'o-', color='b', label='Validation')
    plt.fill_between(
        train_sizes_abs, val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std, alpha=0.1, color='b')

    plt.title("Learning Curve")
    plt.xlabel("Train Size")
    plt.ylabel(f"{scoring}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_calibration_curve(y, model_name, target, n_bins=11, strategy='uniform'):

    prob_true, prob_pred = calibration_curve(y[target], y[f"{model_name}_prob"], n_bins=n_bins, strategy=strategy)

    # plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def prob_to_score(probs, min_score=0, max_score=1000, inverse=False):
    
    """
    Convert probabilities to scores in intervals [min_score, max_score].

    Args:
        probs (array-like): List of probabilities between 0 and 1.
        min_score (int): Scor min (default=1).
        max_score (int): Scor max (default=1000).
        inverse (bool): If True, probability is high => score is lower (ex: risk).

    Returns:
        np.array: Scores between min_score and max_score.
    """

    probs = np.asarray(probs)

    if inverse:
        probs = 1 - probs
    
    scores = min_score + (max_score - min_score) * probs
    
    return np.round(scores).astype(int)

def calc_rating_limits(y, n_ratings=5, min_score=0, max_score=1000):
    
    scores = np.asarray(y)
    
    quantiles = np.linspace(0, 1, n_ratings + 1)[1:-1]
    inner_thresholds = np.quantile(scores, quantiles)
    
    thresholds = [min_score] + inner_thresholds.tolist() + [max_score]

    return thresholds

def apply_ratings(y, thresholds, labels=None):
  
    return pd.cut(y, bins=thresholds, labels=labels, include_lowest=True)

def plot_rating_distribution(y, model_name, target):

    """
    Plot the distribution of ratings.

    Args:
        df (pd.DataFrame): DataFrame with the rating and score.
    """
    
    y.groupby([f'{model_name}_rating'], observed=True)[target].count().plot(kind='bar')
    
    plt.title(f'Rating Distribution - {model_name}')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

def plot_rating(y, model_name, target):
    
    """
    Plot the average score for each rating.

    Args:
        df (pd.DataFrame): DataFrame with the rating and score.
    """
    
    y.groupby([f'{model_name}_rating'], observed=True)[target].mean().plot(kind='bar')
    
    plt.title(f'Rating vs Target - {model_name}')
    plt.xlabel('Rating')
    plt.ylabel('Average Target')
    plt.xticks(rotation=45)
    plt.show()

def summarize_metric_results(results):
        
    rows = []
    
    for dataset, metrics_dict in results.items():
        row = {"Dataset": dataset}
        row.update(metrics_dict)
        rows.append(row)
    
    return pd.DataFrame(rows)

def analyze_model(model_name, model, results, features, X_train, y_train, y_test, target, scoring):
    
    print(f"{model_name} Results")

    display(summarize_metric_results(results))

    prob_col = f"{model_name}_prob"
        
    plot_rating(y_test, model_name, target)
    plot_rating_distribution(y_test, model_name, target)
    plot_roc_curve(y_test, prob_col, target)
    plot_precision_recall_curve(y_test, prob_col, target)
    plot_calibration_curve(y_test, model_name, target, strategy='uniform')
    plot_learning_curve(model, X_train, y_train, target, scoring=scoring)

    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model)
    
    plot_permutation_importance(model, features, X_train, y_train[target], scoring=scoring)
    # plot_shap_summary(model, features, X_train)

def ks_score(y_true, y_prob):
    
    pos_prob = y_prob[y_true == 1]
    neg_prob = y_prob[y_true == 0]
    
    if len(pos_prob) == 0 or len(neg_prob) == 0:
        return 0.0
    
    result = ks_2samp(pos_prob, neg_prob)
    
    return result.statistic

ks_scorer = make_scorer(ks_score, greater_is_better=True, response_method="predict_proba")

def get_eval_scoring(scoring, return_func=True):
    
    scorers = {
        'ks': ks_scorer,
        'roc_auc': 'roc_auc',
        'brier': 'neg_brier_score',
        'log_loss': 'neg_log_loss',
    }

    functions = {
        'ks': ks_score,
        'brier': lambda y_true, y_pred: -1 * brier_score_loss(y_true, y_pred),
        'roc_auc': roc_auc_score,
        'log_loss': lambda y_true, y_pred: -1 * log_loss(y_true, y_pred)
    }
    
    if scoring not in scorers:
        raise ValueError(f"Metric '{scoring}' is not supported.")

    if scoring not in functions:
        raise ValueError(f"Metric '{scoring}' is not supported.")

    return functions[scoring] if return_func else scorers[scoring]

def get_best_threshold_for_f1(y_true, y_probs):
    
    thresholds = np.linspace(0, 1, 200)
    best_thresh = 0
    best_f1 = 0
    
    for thresh in thresholds:
        
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def get_metrics(y, model_name, target):
        
    best_threshold = get_best_threshold_for_f1(y[target], y[f'{model_name}_prob'])

    y[f'{model_name}_pred'] = 0
    y.loc[y[f'{model_name}_prob'] >= best_threshold, f'{model_name}_pred'] = 1
    
    return {
        'Treshold': best_threshold,
        'Balanced Accuracy': balanced_accuracy_score(y[target], y[f'{model_name}_pred']),
        'Precision': precision_score(y[target], y[f'{model_name}_pred']),
        'Recall': recall_score(y[target], y[f'{model_name}_pred']),
        'F1': f1_score(y[target], y[f'{model_name}_pred']),
        'AUC': roc_auc_score(y[target], y[f'{model_name}_prob']),
        'KS': ks_score(y[target], y[f'{model_name}_prob']),
        'Brier': brier_score_loss(y[target], y[f'{model_name}_prob'])
    }
