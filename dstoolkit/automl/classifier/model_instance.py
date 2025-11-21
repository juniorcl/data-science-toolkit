from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def get_model_instance(model_name):
    """
    Get an instance of the specified classification model.

    Parameters
    ----------
    model_name : str
        The name of the classification model to use.

    Returns
    -------
    model : object
        The classifier model instance.

    Raises
    ------
    ValueError
        If the input data is not valid for classification.
    """
    match model_name:
        case 'LightGBM':
            return LGBMClassifier
        case 'CatBoost':
            return CatBoostClassifier
        case 'HistGradientBoosting':
            return HistGradientBoostingClassifier
        case _:
            raise ValueError(f"Model '{model_name}' is not suported.")