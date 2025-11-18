from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

def plot_permutation_importance(model, X, y, scoring):
    permu_results = permutation_importance(model, X, y, scoring=scoring, n_repeats=5, random_state=42)
    sorted_importances_idx = permu_results.importances_mean.argsort()
    df_results = pd.DataFrame(permu_results.importances[sorted_importances_idx].T, columns=X.columns[sorted_importances_idx])
    ax = df_results.plot.box(vert=False, whis=10, patch_artist=True, boxprops={'facecolor':'skyblue', 'color':'blue'})
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel(f"Decrease in {scoring}")
    plt.show()