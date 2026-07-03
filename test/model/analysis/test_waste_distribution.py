import pytest
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.waste_distribution import plot_waste_distribution


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    y_true = np.array([3, -0.5, 2, 7, 4, 2, 3])
    y_score = np.array([2.5, 0.0, 2, 8, 4, 2.5, 3])
    return y_true, y_score


def test_plot_waste_distribution_returns_fig_and_ax(regression_data):
    """Test that plot_waste_distribution returns figure and axes."""
    y_true, y_score = regression_data
    fig, ax = plot_waste_distribution(y_true, y_score)
    assert fig is not None
    assert ax is not None


def test_plot_waste_distribution_creates_valid_plot(regression_data):
    """Test that the function creates a valid matplotlib plot."""
    y_true, y_score = regression_data
    fig, ax = plot_waste_distribution(y_true, y_score)
    assert hasattr(ax, 'figure')
    assert hasattr(ax, 'set_title')


def test_plot_waste_distribution_with_custom_ax(regression_data):
    """Test that plot_waste_distribution works with provided axes."""
    y_true, y_score = regression_data
    fig_custom, ax_custom = plt.subplots()
    fig, ax = plot_waste_distribution(y_true, y_score, ax=ax_custom)
    assert ax is ax_custom


def test_plot_waste_distribution_ax_labels_set(regression_data):
    """Test that axis labels are properly set."""
    y_true, y_score = regression_data
    fig, ax = plot_waste_distribution(y_true, y_score)
    assert ax.get_xlabel() == "Residual (y_true − y_pred)"
    assert ax.get_ylabel() == "Frequency"


def test_plot_waste_distribution_with_list_input():
    """Test that plot_waste_distribution works with list input."""
    y_true = [1, 2, 3, 4, 5]
    y_score = [1.1, 2.1, 2.9, 4.1, 4.9]
    fig, ax = plot_waste_distribution(y_true, y_score)
    assert fig is not None
    assert ax is not None
