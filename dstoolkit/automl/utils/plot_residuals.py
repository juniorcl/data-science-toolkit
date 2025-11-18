import seaborn as sns
import matplotlib.pyplot as plt

def plot_residuals(y, pred_col, target):
    residuals = y[target] - y[pred_col]
    sns.histplot(residuals, kde=True)
    plt.title("Waste Distribution")
    plt.xlabel("Erro (y_true - y_pred)")
    plt.show()