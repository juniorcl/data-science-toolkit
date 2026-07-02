import pytest
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.model.analysis import plot_lift_score_by_bins, plot_lift_curve

matplotlib.use("Agg")


def test_plot_lift_score_by_bins_returns_fig_and_ax():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_lift_score_by_bins(y_true, y_score)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_lift_score_by_bins_uses_given_ax():
    y_true = [0, 1, 1, 0]
    y_score = [0.2, 0.8, 0.7, 0.1]

    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_lift_score_by_bins(y_true, y_score, ax=ax)

    assert returned_ax is ax
    assert returned_fig is fig


def test_plot_lift_score_by_bins_creates_bars():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15]

    fig, ax = plot_lift_score_by_bins(y_true, y_score)

    assert len(ax.containers) >= 1
    bars = ax.containers[0]
    assert len(bars) == 10


def test_plot_lift_score_by_bins_baseline_horizontal_line():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_lift_score_by_bins(y_true, y_score)

    lines = ax.get_lines()
    assert any(line.get_linestyle() == "--" for line in lines)


def test_plot_lift_score_by_bins_labels_and_title():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_lift_score_by_bins(y_true, y_score)

    assert "Lift Score per Range" in ax.get_title()
    assert "Population Split" in ax.get_xlabel()
    assert "Lift Score" in ax.get_ylabel()

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("lift at first group" in txt.lower() for txt in legend_texts)
    assert any("baseline" in txt.lower() or "random" in txt.lower() for txt in legend_texts)


def test_plot_lift_score_by_bins_custom_n_bins():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15, 0.6, 0.4]

    fig, ax = plot_lift_score_by_bins(y_true, y_score, n_bins=5)

    bars = ax.containers[0]
    assert len(bars) == 5


def test_plot_lift_score_by_bins_non_binary_target_raises():
    y_true = [0, 1, 2, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    with pytest.raises(ValueError, match="must contain only 0 or 1"):
        plot_lift_score_by_bins(y_true, y_score)


@pytest.mark.parametrize("container", [list, np.array])
def test_plot_lift_score_by_bins_accepts_different_input_types(container):
    y_true = container([0, 1, 0, 1])
    y_score = container([0.2, 0.8, 0.3, 0.7])

    fig, ax = plot_lift_score_by_bins(y_true, y_score)

    assert len(ax.containers) >= 1


def test_plot_lift_curve_returns_fig_and_ax():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_lift_curve(y_true, y_score)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_lift_curve_uses_given_ax():
    y_true = [0, 1, 1, 0]
    y_score = [0.2, 0.8, 0.7, 0.1]

    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_lift_curve(y_true, y_score, ax=ax)

    assert returned_ax is ax
    assert returned_fig is fig


def test_plot_lift_curve_has_line_and_hline():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15]

    fig, ax = plot_lift_curve(y_true, y_score)

    lines = ax.get_lines()
    assert len(lines) >= 2
    assert any(line.get_linestyle() == "--" for line in lines)


def test_plot_lift_curve_labels_and_title():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_lift_curve(y_true, y_score)

    assert "Curva de Lift" in ax.get_title()
    assert "Proporção" in ax.get_xlabel()
    assert "Lift Score" in ax.get_ylabel()

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("lift" in txt.lower() for txt in legend_texts)
    assert any("baseline" in txt.lower() for txt in legend_texts)


def test_plot_lift_curve_non_binary_target_raises():
    y_true = [0, 1, 2, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    with pytest.raises(ValueError, match="must be binary"):
        plot_lift_curve(y_true, y_score)


@pytest.mark.parametrize("container", [list, np.array])
def test_plot_lift_curve_accepts_different_input_types(container):
    y_true = container([0, 1, 0, 1])
    y_score = container([0.2, 0.8, 0.3, 0.7])

    fig, ax = plot_lift_curve(y_true, y_score)

    assert isinstance(fig, matplotlib.figure.Figure)
