from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_precision_recall_curve(y, prob_col, target):
    precision, recall, _ = precision_recall_curve(y[target], y[prob_col])
    plt.plot(precision, recall)
    plt.title('Precision Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.show()