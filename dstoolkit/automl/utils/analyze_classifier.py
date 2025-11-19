from .plot_ks_curve import plot_ks_curve
from .plot_roc_curve import plot_roc_curve
from .plot_shap_summary import plot_shap_summary
from .plot_learning_curve import plot_learning_curve
from .plot_calibration_curve import plot_calibration_curve
from .plot_feature_importance import plot_feature_importance
from .plot_permutation_importance import plot_permutation_importance
from .plot_precision_recall_curve import plot_precision_recall_curve


def analyze_classifier(model, X_train, y_train, y_test, target, scoring, prob_col='prob'):
    """
    Analyze and visualize classifier performance using various plots.

    This function orchestrates the various plotting functions to provide a comprehensive
    analysis of the classifier's performance.

    Parameters
    ----------
    model : classifier object
        The trained classifier model to be analyzed.
    X_train : pd.DataFrame
        Training feature dataset.
    y_train : pd.DataFrame
        Training target dataset.
    y_test : pd.DataFrame
        Testing target dataset.
    target : str
        Name of the target variable.
    scoring : str
        Scoring metric used for evaluation.
    prob_col : str, optional
        Name of the column containing predicted probabilities in y_test, by default 'prob'.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the input data is not valid for classification.
    """
    plot_roc_curve(y_test, prob_col, target)
    plot_ks_curve(y_test, target)
    plot_precision_recall_curve(y_test, prob_col, target)
    plot_calibration_curve(y_test, target, strategy='uniform')
    plot_learning_curve(model, X_train, y_train[target], scoring=scoring)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model)
    plot_permutation_importance(model, X_train, y_train[target], scoring=scoring)
    plot_shap_summary(model, X_train)