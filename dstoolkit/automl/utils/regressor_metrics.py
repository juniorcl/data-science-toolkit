from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)


def get_regressor_metrics(y, target, pred_col="pred"):
    """
    Calculate various regression metrics between true and predicted values.

    Parameters
    ----------
    y : pd.DataFrame
        Dataframe or array-like containing true and predicted values.
    pred_col : str
        The name of the column containing predicted values.
    target : str
        The name of the column containing true target values.

    Returns
    -------
    dict
        A dictionary containing the calculated regression metrics:
        - R2
        - MAE (Mean Absolute Error)
        - MadAE (Median Absolute Error)
        - MAPE (Mean Absolute Percentage Error)
        - RMSE (Root Mean Squared Error)
        - Explained Variance

    Raises
    ------
    KeyError
        If `pred_col` or `target` are not found in the DataFrame `y`.
    """
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