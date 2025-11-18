import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, KFold

def plot_learning_curve(model, X, y, scoring):
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