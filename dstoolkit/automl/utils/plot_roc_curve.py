from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y, prob_col, target):
    fpr, tpr, _ = roc_curve(y[target], y[prob_col])
    plt.plot(fpr, tpr)
    plt.title('ROC AUC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()