import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def plot_tree_ovr(X, labels, max_depth=4, class_names=None):
    """
    Generate One-vs-Rest (OvR) decision trees to explain each label individually.

    For each unique label, a binary decision tree is trained to distinguish
    that label versus all others, and a corresponding tree visualization
    is generated.

    Parameters
    ----------
    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix.

    labels : array-like of shape (n_samples,)
        Class or cluster labels.

    max_depth : int, default=4
        Maximum depth of the decision trees.

    class_names : dict, optional
        Optional mapping {label: friendly_name}.

    Returns
    -------
    results : dict
        Dictionary indexed by label, where each value is a dict with:
        - "tree": trained DecisionTreeClassifier
        - "fig": matplotlib.figure.Figure
        - "ax": matplotlib.axes.Axes
    """
    labels = np.asarray(labels)
    unique_classes = np.unique(labels)

    results = {}

    for cls in unique_classes:
        # One-vs-Rest target
        y_binary = (labels == cls).astype(int)

        # Train model
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
        )
        tree.fit(X, y_binary)

        # Friendly name
        cls_name = class_names.get(cls, str(cls)) if class_names else str(cls)

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(f"Decision Tree – One vs Rest (Label {cls_name})")

        plot_tree(
            tree,
            feature_names=X.columns if hasattr(X, "columns") else None,
            class_names=[f"Not {cls_name}", cls_name],
            filled=True,
            proportion=True,
            rounded=True,
            ax=ax,
        )

        results[cls] = {
            "tree": tree,
            "fig": fig,
            "ax": ax,
        }

    return results