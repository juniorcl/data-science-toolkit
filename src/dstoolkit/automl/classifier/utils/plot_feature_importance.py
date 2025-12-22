import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(model, top_n=20):
    """
    Plots the feature importance of a given model.

    This function visualizes the importance of each feature in the model,
    helping to understand which features are driving predictions.

    Parameters
    ----------
    model : object
        A trained model object that contains feature importance attributes.
    top_n : int, optional
        The number of top features to display in the plot (default is 20).

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the model does not have feature importance attributes.
    """
    if hasattr(model, 'feature_name_'):
        feature_names = model.feature_name_
    elif hasattr(model, 'feature_names_'):
        feature_names = model.feature_names_
    elif hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        raise AttributeError("It was not possible to identify the feature names in the model.")

    if hasattr(model, 'feature_importances_'):  # LightGBM, XGBoost, CatBoost (sklearn model)
        importances = model.feature_importances_
    elif hasattr(model, 'get_score'):  # XGBoost native model
        importances_dict = model.get_score(importance_type='weight')
        feature_names = list(importances_dict.keys())
        importances = list(importances_dict.values())
    else:
        raise AttributeError("Model doesn't support feature importance attribute.")

    df_imp = pd.DataFrame({"Variable": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, min(top_n, 20) * 0.4 + 2))
    sns.barplot(x="Importance", y="Variable", data=df_imp.head(top_n), color="#006e9cff")
    plt.title("Importance of Variables")
    plt.tight_layout()
    plt.show()