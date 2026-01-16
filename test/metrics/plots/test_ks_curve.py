import pytest
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dstoolkit.metrics.plots import plot_ks_curve

matplotlib.use("Agg")  # backend não interativo


def test_plot_ks_curve_returns_fig_and_ax():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.8, 0.3, 0.7, 0.9, 0.2])

    fig, ax = plot_ks_curve(y_true, y_score)

    assert fig is not None
    assert ax is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

def test_plot_ks_curve_uses_given_ax():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.9, 0.3, 0.8])

    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_ks_curve(
        y_true, y_score, ax=ax
    )

    assert returned_ax is ax
    assert returned_fig is fig

def test_plot_ks_curve_adds_expected_lines():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.3, 0.8, 0.7, 0.2]

    fig, ax = plot_ks_curve(y_true, y_score)

    assert len(ax.get_lines()) == 2
    assert len(ax.collections) == 1

def test_plot_ks_curve_labels_and_title():
    y_true = [0, 1, 1, 0]
    y_score = [0.2, 0.7, 0.9, 0.1]

    fig, ax = plot_ks_curve(y_true, y_score)

    assert ax.get_title() == "Kolmogorov–Smirnov Curve"
    assert ax.get_xlabel() == "Predicted Probability of Class 1"
    assert ax.get_ylabel() == "Cumulative Proportion"

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Class 1 (CDF)" in legend_texts
    assert "Class 0 (CDF)" in legend_texts
    assert any("KS =" in txt for txt in legend_texts)

def test_plot_ks_curve_ks_value_in_legend():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    fig, ax = plot_ks_curve(y_true, y_score)

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("KS =" in txt for txt in legend_texts)

def test_plot_ks_curve_ks_value_range():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_score = np.array([0.05, 0.95, 0.2, 0.8, 0.9, 0.1, 0.7, 0.3])

    fig, ax = plot_ks_curve(y_true, y_score)

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    ks_text = next(txt for txt in legend_texts if "KS =" in txt)
    ks_value = float(ks_text.split("KS =")[1])

    assert 0.0 <= ks_value <= 1.0

def test_plot_ks_curve_non_binary_target_raises():
    y_true = [0, 1, 2, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    with pytest.raises(ValueError, match="y_true must be binary"):
        plot_ks_curve(y_true, y_score)

def test_plot_ks_curve_mismatched_input_length_raises():
    y_true = [0, 1, 0]
    y_score = [0.2, 0.8]

    with pytest.raises(ValueError):
        plot_ks_curve(y_true, y_score)

@pytest.mark.parametrize("container", [list, np.array, pd.Series])
def test_plot_ks_curve_accepts_different_input_types(container):
    y_true = container([0, 1, 0, 1])
    y_score = container([0.2, 0.8, 0.3, 0.7])

    fig, ax = plot_ks_curve(y_true, y_score)

    assert len(ax.get_lines()) == 2

def test_plot_ks_curve_constant_scores():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score = np.array([0.5] * len(y_true))

    fig, ax = plot_ks_curve(y_true, y_score)

    assert len(ax.get_lines()) == 2
    assert len(ax.collections) == 1