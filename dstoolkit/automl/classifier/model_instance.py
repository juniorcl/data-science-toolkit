from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def get_model_instance(model_name):
    match model_name:
        case 'LightGBM':
            return LGBMClassifier
        case 'CatBoost':
            return CatBoostClassifier
        case 'HistGradientBoosting':
            return HistGradientBoostingClassifier
        case _:
            raise ValueError(f"Model '{model_name}' is not suported.")