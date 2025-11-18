from .plot_residuals import plot_residuals
from .plot_pred_vs_true import plot_pred_vs_true
from .plot_error_by_quantile import plot_error_by_quantile
from .plot_learning_curve import plot_learning_curve
from .plot_feature_importance import plot_feature_importance
from .plot_permutation_importance import plot_permutation_importance
from .plot_shap_summary import plot_shap_summary

def analyze_regressor(model, X_train, y_train, y_test, target, scoring):
    pred_col = 'pred'
    plot_residuals(y_test, pred_col, target)
    plot_pred_vs_true(y_test, pred_col, target)
    plot_error_by_quantile(y_test, pred_col, target)
    plot_learning_curve(model, X_train, y_train[target], scoring=scoring)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model)
    plot_permutation_importance(model, X_train, y_train[target], scoring=scoring)
    plot_shap_summary(model, X_train)