import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier, plot_tree


def plot_tree_ovo(
    X,
    labels,
    max_depth=4,
    figsize=(16, 12),
    class_names=None,
):
    """
    Generates One-vs-One (OvO) decision trees to explain pairwise class differences.

    For each pair of classes (i, j), a binary decision tree is trained using
    only samples belonging to those two classes.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        Feature set.

    labels : array-like of shape (n_samples,)
        Class or cluster labels.

    max_depth : int, default=4
        Maximum depth of the decision trees.

    figsize : tuple, default=(16, 12)
        Figure size for each tree plot.

    class_names : dict, optional
        Optional mapping {label: friendly_name}.

    Returns
    -------
    trees : dict
        Dictionary where keys are tuples (class_i, class_j) and values are dicts:
        {
            "model": DecisionTreeClassifier,
            "fig": matplotlib.figure.Figure,
            "ax": matplotlib.axes.Axes
        }
    """
    labels = np.asarray(labels)
    unique_classes = np.unique(labels)
    trees = {}

    for cls_i, cls_j in combinations(unique_classes, 2):

        # Filter samples belonging only to the pair
        mask = (labels == cls_i) | (labels == cls_j)
        X_pair = X.loc[mask]
        y_pair = labels[mask]

        # Binary encoding
        y_binary = (y_pair == cls_j).astype(int)

        # Train tree
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
        )
        tree.fit(X_pair, y_binary)

        # Friendly names
        name_i = class_names.get(cls_i, str(cls_i)) if class_names else str(cls_i)
        name_j = class_names.get(cls_j, str(cls_j)) if class_names else str(cls_j)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Decision Tree — One vs One ({name_i} × {name_j})")

        plot_tree(
            tree,
            feature_names=X.columns,
            class_names=[name_i, name_j],
            filled=True,
            proportion=True,
            rounded=True,
            ax=ax,
        )

        fig.tight_layout()

        trees[(cls_i, cls_j)] = {
            "model": tree,
            "fig": fig,
            "ax": ax,
        }

    return trees