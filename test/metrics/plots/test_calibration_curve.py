import pytest
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.metrics.plots import plot_calibration_curve

matplotlib.use("Agg")


def test_plot_calibration_curve_returns_fig_and_ax():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.1, 0.8, 0.3, 0.7, 0.9, 0.2])

    fig, ax = plot_calibration_curve(y_true, y_score)

    assert fig is not None
    assert ax is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

def test_plot_calibration_curve_uses_given_ax():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.2, 0.9, 0.8, 0.1])

    fig, ax = plt.subplots()

    returned_fig, returned_ax = plot_calibration_curve(
        y_true, y_score, ax=ax
    )

    assert returned_ax is ax
    assert returned_fig is fig

def test_plot_calibration_curve_adds_two_lines():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])

    fig, ax = plot_calibration_curve(y_true, y_score)

    lines = ax.get_lines()
    assert len(lines) == 2

def test_plot_calibration_curve_labels_and_title():
    y_true = np.array([0, 1, 1, 0])
    y_score = np.array([0.2, 0.7, 0.9, 0.1])

    fig, ax = plot_calibration_curve(
        y_true, y_score, model_name="MyModel"
    )

    assert ax.get_xlabel() == "Mean Predicted Probability"
    assert ax.get_ylabel() == "Fraction of Positives"
    assert ax.get_title() == "Calibration Curve"

    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("MyModel" in txt for txt in legend_texts)
    assert any("Perfectly calibrated" in txt for txt in legend_texts)

@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_plot_calibration_curve_with_different_strategies(strategy):
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.9, 0.3, 0.7, 0.8, 0.2, 0.6, 0.4])

    fig, ax = plot_calibration_curve(
        y_true, y_score, n_bins=4, strategy=strategy
    )

    assert len(ax.get_lines()) == 2

def test_plot_calibration_curve_brier_score_in_legend():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.25, 0.75, 0.3, 0.9])

    fig, ax = plot_calibration_curve(y_true, y_score)

    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("Brier =" in txt for txt in legend_texts)

def test_plot_calibration_curve_invalid_input_raises():
    y_true = np.array([0, 1, 0])
    y_score = np.array([0.2, 0.8])  # tamanho diferente 

    with pytest.raises(ValueError):
        plot_calibration_curve(y_true, y_score)