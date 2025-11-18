import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_error_by_quantile(y, pred_col, target):
    y_copy = y.copy()
    y_copy["quantile"] = pd.qcut(y_copy[target], q=5)
    y_copy["abs_error"] = abs(y_copy[target] - y_copy[pred_col])
    sns.boxplot(x="quantile", y="abs_error", data=y_copy)
    plt.title("Absolute Error by Target Quantile")
    plt.xticks(rotation=45)
    plt.show()