import pytest
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.metrics.plots import plot_roc_curve

matplotlib.use("Agg")


def test_plot_roc_curve_returns_fig_and_ax():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_roc_curve(y_true, y_score)

    assert fig is not None
    assert ax is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

def test_plot_roc_curve_uses_given_ax():
    y_true = [0, 1, 1, 0]
    y_score = [0.2, 0.8, 0.7, 0.1]

    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_roc_curve(
        y_true, y_score, ax=ax
    )

    assert returned_ax is ax
    assert returned_fig is fig

def test_plot_roc_curve_adds_curve_and_random_line():
    y_true = [0, 1, 0, 1]
    y_score = [0.05, 0.9, 0.3, 0.8]

    fig, ax = plot_roc_curve(y_true, y_score)

    # 2 linhas: ROC + diagonal
    assert len(ax.get_lines()) == 2

def test_plot_roc_curve_labels_and_title():
    y_true = [0, 1, 0, 1]
    y_score = [0.2, 0.7, 0.3, 0.9]

    fig, ax = plot_roc_curve(y_true, y_score)

    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate (Recall)"
    assert ax.get_title() == "ROC Curve"

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("AUC =" in txt for txt in legend_texts)
    assert "Random" in legend_texts

def test_plot_roc_curve_non_binary_target_raises():
    y_true = [0, 1, 2, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    with pytest.raises(ValueError, match="binary"):
        plot_roc_curve(y_true, y_score)

@pytest.mark.parametrize("container", [list, np.array])
def test_plot_roc_curve_accepts_different_input_types(container):
    y_true = container([0, 1, 0, 1])
    y_score = container([0.2, 0.8, 0.3, 0.7])

    fig, ax = plot_roc_curve(y_true, y_score)

    assert len(ax.get_lines()) == 2

def test_plot_roc_curve_auc_in_legend():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.7, 0.9])

    fig, ax = plot_roc_curve(y_true, y_score)

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("AUC = 1.000" in txt for txt in legend_texts)