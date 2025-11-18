from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def get_model_instance(model_name):
    match model_name:
        case 'LightGBM':
            return LGBMRegressor
        case 'CatBoost':
            return CatBoostRegressor
        case 'HistGradientBoosting':
            return HistGradientBoostingRegressor
        case _:
            raise ValueError(f"Model '{model_name}' is not suported.")