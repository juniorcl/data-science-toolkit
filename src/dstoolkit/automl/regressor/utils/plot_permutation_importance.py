import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_permutation_importance(model, X, y, scoring):
    """
    Plots the permutation importance of features for a given model.

    This function computes the permutation importance of each feature by shuffling the 
    feature values and measuring the impact on the model's performance. 
    It then visualizes the results using a box plot.

    Parameters
    ----------
    model : object
        A scikit-learn compatible estimator (model) to evaluate.
    X : DataFrame
        Feature dataset used for training and validation.
    y : Series
        Target labels corresponding to the feature dataset.
    scoring : str
        A string representing the scoring metric to use (e.g., 'accuracy', 'roc_auc').

    Returns
    -------
    None
        This function does not return a value.

    Raises
    ------
    ValueError
        If the model is not fitted or if there is an issue with the data.
    """
    permu_results = permutation_importance(model, X, y, scoring=scoring, n_repeats=5, random_state=42)
    sorted_importances_idx = permu_results.importances_mean.argsort()
    
    df_results = pd.DataFrame(permu_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    
    ax = df_results.plot.box(vert=False, whis=10, patch_artist=True, boxprops={'facecolor':'skyblue', 'color':'blue'})
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel(f"Decrease in {scoring}")
    
    plt.show()