import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold


def plot_learning_curve(model, X, y, scoring):
    """
    Plot the learning curve of a machine learning model.

    This function generates a learning curve plot, which shows the relationship
    between the size of the training dataset and the model's performance on both
    the training and validation sets.

    Parameters
    ----------
    model : object
        A scikit-learn compatible estimator (model) to evaluate.
    X : array-like
        Feature dataset used for training and validation.
    y : array-like
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
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
        cv=KFold(n_splits=3, shuffle=True, random_state=42), scoring=scoring
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='r', label='Train')
    plt.fill_between(
        train_sizes_abs, train_scores_mean - train_scores_std, 
        train_scores_mean + train_scores_std, alpha=0.1, color='r'
    )

    plt.plot(train_sizes_abs, val_scores_mean, 'o-', color='b', label='Validation')
    plt.fill_between(
        train_sizes_abs, val_scores_mean - val_scores_std, 
        val_scores_mean + val_scores_std, alpha=0.1, color='b'
    )

    plt.title("Learning Curve")
    plt.xlabel("Train Size")
    plt.ylabel(f"{scoring}")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()