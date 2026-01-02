import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold


def plot_learning_curve(model, X, y, scoring, cv=None, n_jobs=-1, ax=None):
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

    X : array-like of shape (n_samples, n_features)
        Feature matrix used for training and validation.

    y : array-like of shape (n_samples,)
        Target values corresponding to `X`.

    scoring : str or callable
        Scoring metric to evaluate the model performance.

    cv : int, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy.
        If None, a 3-fold `KFold` with shuffling and a fixed random state
        is used.

    n_jobs : int, default=-1
        Number of jobs to run in parallel during cross-validation.

    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object.
    """
    if cv is None:
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        train_sizes=np.linspace(0.1, 1.0, 5),
        estimator=model, 
        scoring=scoring, 
        n_jobs=n_jobs,
        cv=cv,
        X=X, 
        y=y, 
    )

    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    val_scores_mean = val_scores.mean(axis=1)
    val_scores_std = val_scores.std(axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Train curve
    ax.plot(
        train_sizes_abs,
        train_scores_mean,
        marker="o",
        label="Train",
    )
    ax.fill_between(
        train_sizes_abs,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.15,
    )

    # Validation curve
    ax.plot(
        train_sizes_abs,
        val_scores_mean,
        marker="o",
        label="Validation",
    )
    ax.fill_between(
        train_sizes_abs,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.15,
    )

    ax.set_title("Learning Curve")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(scoring)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    return fig, ax