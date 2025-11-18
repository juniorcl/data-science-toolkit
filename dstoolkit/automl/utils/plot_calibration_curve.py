from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_curve(y, target, n_bins=10, strategy='uniform'):
    prob_true, prob_pred = calibration_curve(y[target], y['prob'], n_bins=n_bins, strategy=strategy)
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()