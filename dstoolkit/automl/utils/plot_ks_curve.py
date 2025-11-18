import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_ks_curve(y, target, bins=100):
    df = pd.DataFrame({"y_true": y[target], "y_proba": y['prob']})
    df = df.sort_values("y_proba")
    
    total_1 = (df["y_true"] == 1).sum()
    total_0 = (df["y_true"] == 0).sum()
    
    df["cum_1"] = (df["y_true"] == 1).cumsum() / total_1
    df["cum_0"] = (df["y_true"] == 0).cumsum() / total_0
    
    df["diff"] = np.abs(df["cum_1"] - df["cum_0"])
    ks_value = df["diff"].max()
    ks_idx = df["diff"].idxmax()
    ks_threshold = df.loc[ks_idx, "y_proba"]

    plt.figure(figsize=(8, 5))
    plt.plot(df["y_proba"], df["cum_1"], label="Classe 1 (Acumulada)", color="tab:blue")
    plt.plot(df["y_proba"], df["cum_0"], label="Classe 0 (Acumulada)", color="tab:orange")
    plt.vlines(ks_threshold, ymin=df.loc[ks_idx, "cum_0"], ymax=df.loc[ks_idx, "cum_1"],
               colors="red", linestyles="--", label=f"KS = {ks_value:.3f}")
    
    plt.title("Curvas de Distribuição Acumulada (KS Test)")
    plt.xlabel("Probabilidade prevista da classe 1")
    plt.ylabel("Proporção acumulada")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()