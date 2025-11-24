from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def get_model_instance(model_name):
    """
    Get an instance of the specified regression model.

    Parameters
    ----------
    model_name : str
        The name of the regression model to use.

    Returns
    -------
    model : object
        The regression model instance.

    Raises
    ------
    ValueError
        If the input data is not valid for regression.
    """
    match model_name:
        case 'LightGBM':
            return LGBMRegressor
        case 'CatBoost':
            return CatBoostRegressor
        case 'HistGradientBoosting':
            return HistGradientBoostingRegressor
        case _:
            raise ValueError(f"Model '{model_name}' is not suported.")