import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def plot_ks_curve(y_true, y_score):
    """
    Plot the KS (Kolmogorov-Smirnov) cumulative distribution curves for a
    binary classification model using predicted probabilities.

    The function computes the KS statistic using the two-sample KS test
    from `scipy.stats.ks_2samp`, comparing the distribution of predicted
    probabilities for the positive and negative classes. It also plots the
    cumulative distributions and visually highlights the KS distance.

    Parameters
    ----------
    y : pandas.DataFrame
        DataFrame containing at least two columns:
        - `target`: the true binary labels (0 or 1)
        - `'prob'`: predicted probability for the positive class.
    target : str
        Name of the target column inside `y`.

    Returns
    -------
    float
        The KS statistic computed using `scipy.stats.ks_2samp`.

    Raises
    ------
    ValueError
        If required columns are missing or if the target is not binary.
    """

    if set(y_true.unique()) - {0, 1}:
        raise ValueError("Target column must be binary (0/1).")

    # Separate predicted probabilities by class
    scores_1 = y_score[y_true == 1]
    scores_0 = y_score[y_true == 0]

    # Compute KS statistic using SciPy
    ks_value, _ = ks_2samp(scores_1, scores_0)

    # Prepare cumulative distributions for plotting
    df = pd.DataFrame({'target': y_true, 'score': y_score})
    df = df.sort_values('score')

    total_1 = (df['target'] == 1).sum()
    total_0 = (df['target'] == 0).sum()

    df["cum_1"] = (df['target'] == 1).cumsum() / total_1
    df["cum_0"] = (df['target'] == 0).cumsum() / total_0

    # Point of maximum difference → not from scipy (scipy does not return threshold)
    df["diff"] = np.abs(df["cum_1"] - df["cum_0"])
    ks_idx = df["diff"].idxmax()
    ks_threshold = df.loc[ks_idx, 'score']

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['score'], df["cum_1"], label="Classe 1 (CDF)", color="tab:blue")
    plt.plot(df['score'], df["cum_0"], label="Classe 0 (CDF)", color="tab:orange")

    plt.vlines(
        ks_threshold,
        ymin=df.loc[ks_idx, "cum_0"],
        ymax=df.loc[ks_idx, "cum_1"],
        colors="red",
        linestyles="--",
        label=f"KS = {ks_value:.3f}",
    )

    plt.title("Kolmogorov–Smirnov Curve")
    plt.xlabel("Predicted Probability of Class 1")
    plt.ylabel("Cumulative Proportion")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()