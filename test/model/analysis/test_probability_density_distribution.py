import pytest
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.model.analysis import plot_probability_density_distribution

matplotlib.use("Agg")


def test_plot_probability_density_distribution_returns_fig_and_ax():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_probability_density_distribution(y_true, y_score)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_probability_density_distribution_uses_given_ax():
    y_true = [0, 1, 1, 0]
    y_score = [0.2, 0.8, 0.7, 0.1]

    fig, ax = plt.subplots()
    returned_fig, returned_ax = plot_probability_density_distribution(
        y_true, y_score, ax=ax
    )

    assert returned_ax is ax
    assert returned_fig is fig


def test_plot_probability_density_distribution_labels_and_title():
    y_true = [0, 1, 0, 1, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    fig, ax = plot_probability_density_distribution(y_true, y_score)

    assert "Probability Distribution" in ax.get_title()
    assert ax.get_xlabel() == "Predicted Probability (Score)"
    assert "Density" in ax.get_ylabel()


def test_plot_probability_density_distribution_legend():
    y_true = [0, 1, 0, 1, 1, 0, 1, 0]
    y_score = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15]

    fig, ax = plot_probability_density_distribution(y_true, y_score)

    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Target 0" in txt for txt in legend_texts)
    assert any("Target 1" in txt for txt in legend_texts)


def test_plot_probability_density_distribution_xlim():
    y_true = [0, 1, 0, 1]
    y_score = [0.2, 0.8, 0.3, 0.7]

    fig, ax = plot_probability_density_distribution(y_true, y_score)

    xlim = ax.get_xlim()
    assert xlim[0] >= 0
    assert xlim[1] <= 1


def test_plot_probability_density_distribution_single_class():
    y_true = [1, 1, 1]
    y_score = [0.7, 0.8, 0.9]

    fig, ax = plot_probability_density_distribution(y_true, y_score)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_probability_density_distribution_non_binary_target_raises():
    y_true = [0, 1, 2, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    with pytest.raises(ValueError, match="must contain only 0 or 1"):
        plot_probability_density_distribution(y_true, y_score)


@pytest.mark.parametrize("container", [list, np.array])
def test_plot_probability_density_distribution_accepts_different_input_types(container):
    y_true = container([0, 1, 0, 1])
    y_score = container([0.2, 0.8, 0.3, 0.7])

    fig, ax = plot_probability_density_distribution(y_true, y_score)

    assert isinstance(fig, matplotlib.figure.Figure)
