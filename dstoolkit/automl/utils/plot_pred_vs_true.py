import seaborn as sns
import matplotlib.pyplot as plt

def plot_pred_vs_true(y, pred_col, target):
    sns.scatterplot(x=y[target], y=y[pred_col])
    plt.plot([y[target].min(), y[target].max()], [y[target].min(), y[target].max()], '--r')
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("y_true vs y_pred")
    plt.show()