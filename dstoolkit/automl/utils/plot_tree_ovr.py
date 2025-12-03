import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def plot_tree_ovr(X, labels, max_depth=4, figsize=(16, 12), class_names=None):
    """
    Generates One-vs-Rest (OvR) decision trees to explain each label individually.

    Parameters
    ----------
    X : pd.DataFrame
        Feature set.
    labels : array-like
        Classification or cluster labels.
    max_depth : int, optional
        Maximum depth of the decision tree.
    figsize : tuple, optional
        Figure size for the plots.
    class_names : dict, optional
        Optional mapping {label: friendly_name}.

    Returns
    -------
    trees : dict
        Key: label  
        Value: trained decision tree model explaining that label versus all others.
    """
    
    unique_classes = np.unique(labels)
    labels = np.array(labels)
    trees = {}

    for cls in unique_classes:

        # One-vs-Rest
        y_binary = (labels == cls).astype(int)

        # Model
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        tree.fit(X, y_binary)

        # Nome amigável
        cls_name = class_names[cls] if class_names else str(cls)

        # Plot
        plt.figure(figsize=figsize)
        plt.title(f"Decision Tree - One vs Rest (Label {cls_name})")

        plot_tree(
            tree,
            feature_names=X.columns,
            class_names=[f"Not {cls_name}", cls_name],
            filled=True,
            proportion=True,
            rounded=True
        )

        plt.show()

        trees[cls] = tree

    return trees
