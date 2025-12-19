from .plot_residuals import plot_residuals
from .plot_pred_vs_true import plot_pred_vs_true
from .plot_shap_summary import plot_shap_summary
from .plot_learning_curve import plot_learning_curve
from .plot_error_by_quantile import plot_error_by_quantile
from .plot_feature_importance import plot_feature_importance
from .plot_permutation_importance import plot_permutation_importance


def analyze_regressor(model, X_train, y_train, y_test, target, scoring, pred_col='pred'):
    """
    Analyze and visualize regressor performance using various plots.

    This function orchestrates the various plotting functions to provide a comprehensive
    analysis of the regressor's performance.
    
    Parameters
    ----------
    model : regressor object
        The trained regressor model to be analyzed.
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
    pred_col : str, optional
        Name of the column containing predicted values in y_test, by default 'pred'.

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the input data is not valid for regression.
    """
    plot_residuals(y_test, pred_col, target)
    plot_pred_vs_true(y_test, pred_col, target)
    plot_error_by_quantile(y_test, pred_col, target)
    plot_learning_curve(model, X_train, y_train[target], scoring=scoring)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model)
    plot_permutation_importance(model, X_train, y_train[target], scoring=scoring)
    plot_shap_summary(model, X_train)