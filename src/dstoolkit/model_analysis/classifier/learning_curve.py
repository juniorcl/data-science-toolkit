import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold


def plot_learning_curve(model, X, y, scoring, cv=None, n_jobs=-1):
    """
    Plot the learning curve of a machine learning estimator.

    This function visualizes how the model performance evolves as the size
    of the training dataset increases. It plots both the training and
    cross-validation scores, including their variability across folds,
    which helps diagnose bias, variance, and data sufficiency issues.

    Parameters
    ----------
    model : estimator object
        A scikit-learn compatible estimator implementing `fit`.
        The model is trained repeatedly on increasing subsets of the data.

    X : array-like of shape (n_samples, n_features)
        Feature matrix used for training and validation.

    y : array-like of shape (n_samples,)
        Target values corresponding to `X`.

    scoring : str or callable
        Scoring metric to evaluate the model performance (e.g., 'accuracy',
        'roc_auc', 'neg_log_loss').

    cv : int, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy.
        If None, a 3-fold `KFold` with shuffling and a fixed random state
        is used.

    n_jobs : int, default=-1
        Number of jobs to run in parallel during cross-validation.
        `-1` means using all available processors.

    Returns
    -------
    None
        This function does not return any value. It produces a matplotlib
        figure displaying the learning curve.

    Notes
    -----
    - A small gap between training and validation curves suggests good
      generalization.
    - A large gap indicates high variance (overfitting).
    - Low scores for both curves indicate high bias (underfitting).
    - If validation performance keeps improving with more data, the model
      may benefit from additional training samples.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier(random_state=42)
    >>> plot_learning_curve(
    ...     model,
    ...     X_train,
    ...     y_train,
    ...     scoring="roc_auc"
    ... )
    """
    
    if cv is None:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv, scoring=scoring, n_jobs=n_jobs
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))

    # Train
    plt.plot(
        train_sizes_abs,
        train_scores_mean,
        'o-',
        color='tab:blue',
        label='Train',
    )
    plt.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.15,
        color='tab:blue',
    )

    # Validation
    plt.plot(
        train_sizes_abs,
        val_scores_mean,
        'o-',
        color='tab:orange',
        label='Validation',
    )
    plt.fill_between(
        train_sizes_abs,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.15,
        color='tab:orange',
    )

    plt.title("Learning Curve")
    plt.xlabel("Train Size")
    plt.ylabel(scoring)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()