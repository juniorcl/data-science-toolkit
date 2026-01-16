import pytest
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.metrics.plots import plot_precision_recall_curve

matplotlib.use("Agg")


def test_plot_pr_curve_returns_fig_and_ax():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_precision_recall_curve(y_true, y_score)

    assert fig is not None
    assert ax is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

def test_plot_pr_curve_uses_given_ax():
    y_true = [0, 1, 1, 0]
    y_score = [0.2, 0.8, 0.7, 0.1]

    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_precision_recall_curve(
        y_true, y_score, ax=ax
    )

    assert returned_ax is ax
    assert returned_fig is fig

def test_plot_pr_curve_adds_curve_and_baseline():
    y_true = [0, 1, 0, 1]
    y_score = [0.05, 0.9, 0.3, 0.8]

    fig, ax = plot_precision_recall_curve(y_true, y_score)

    # 1 line (curva PR)
    assert len(ax.get_lines()) == 1

    # 1 collection (hlines create LineCollection)
    assert len(ax.collections) == 1

def test_plot_pr_curve_labels_and_title():
    y_true = [0, 1, 0, 1]
    y_score = [0.2, 0.7, 0.3, 0.9]

    fig, ax = plot_precision_recall_curve(y_true, y_score)

    assert ax.get_xlabel() == "Recall"
    assert ax.get_ylabel() == "Precision"
    assert ax.get_title() == "Precision–Recall Curve"

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("AP =" in txt for txt in legend_texts)
    assert any("Baseline =" in txt for txt in legend_texts)

def test_plot_pr_curve_non_binary_target_raises():
    y_true = [0, 1, 2, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    with pytest.raises(ValueError, match="binary"):
        plot_precision_recall_curve(y_true, y_score)

@pytest.mark.parametrize("container", [list, np.array])
def test_plot_pr_curve_accepts_different_input_types(container):
    y_true = container([0, 1, 0, 1])
    y_score = container([0.2, 0.8, 0.3, 0.7])

    fig, ax = plot_precision_recall_curve(y_true, y_score)

    assert len(ax.get_lines()) == 1

def test_plot_pr_curve_baseline_value():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])

    fig, ax = plot_precision_recall_curve(y_true, y_score)

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Baseline = 0.500" in txt for txt in legend_texts)