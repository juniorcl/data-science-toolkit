from sklearn.inspection import permutation_importance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_importance(model, top_n=20):
    if hasattr(model, 'feature_name_'):
        feature_names = model.feature_name_
    elif hasattr(model, 'feature_names_'):
        feature_names = model.feature_names_
    elif hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        raise AttributeError("It was not possible to identify the feature names in the model.")

    if hasattr(model, 'feature_importances_'):  # LightGBM, XGBoost, CatBoost (modo sklearn)
        importances = model.feature_importances_
    elif hasattr(model, 'get_score'):  # XGBoost nativo (booster)
        importances_dict = model.get_score(importance_type='weight')
        feature_names = list(importances_dict.keys())
        importances = list(importances_dict.values())
    else:
        raise AttributeError("Model doesn't support featue importance attribute.")

    df_imp = pd.DataFrame({"Variable": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, min(top_n, 20) * 0.4 + 2))
    sns.barplot(x="Importance", y="Variable", data=df_imp.head(top_n), color="#006e9cff")
    plt.title("Importance of Variables")
    plt.tight_layout()
    plt.show()