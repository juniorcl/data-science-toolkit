import numpy as np

from sklearn import metrics


def get_regression_metrics(y, col_target='target', col_pred='pred', decimals=0):

    """
    Function to calculate the regression metrics: R2, RMSE, MAE, MAPE, MedAE

    Parameters
    ----------
    y : DataFrame
        Data frame with target and prediction.

    col_target : str
        The name of the columns with the target.

    col_pred : str
        The name of the columns with the prediction.

    decimals : int
        Number of decimal places to round.
    
    Returns
    -------
    dict_results : Dict
        Dictionary with metric results.
    """

    r2 = metrics.r2_score(y[col_target], y[col_pred])

    rmse = metrics.root_mean_squared_error(y[col_target], y[col_pred])

    mae = metrics.mean_absolute_error(y[col_target], y[col_pred])

    mape = metrics.mean_absolute_percentage_error(y[col_target], y[col_pred])

    medae = metrics.median_absolute_error(y[col_target], y[col_pred])

    dict_results = {
        "R2": np.round(r2, decimals), "RMSE": np.round(rmse, decimals), 
        "MAE": np.round(mae, decimals), "MAPE": np.round(mape, decimals), "MedAE": np.round(medae, decimals)}

    return dict_results