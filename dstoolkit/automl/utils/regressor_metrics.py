from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)


def get_regressor_metrics(y, pred_col, target):
    y_true = y[target]
    y_pred = y[pred_col]
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MadAE': median_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred)
    }