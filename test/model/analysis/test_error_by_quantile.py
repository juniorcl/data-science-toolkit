import pytest
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.error_by_quantile import plot_error_by_quantile


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    y_true = np.array([3, -0.5, 2, 7, 4, 2, 3, 5, 6, 8])
    y_score = np.array([2.5, 0.0, 2, 8, 4, 2.5, 3, 4.5, 6.5, 7.5])
    return y_true, y_score


def test_plot_error_by_quantile_returns_fig_and_ax(regression_data):
    """Test that plot_error_by_quantile returns figure and axes."""
    y_true, y_score = regression_data
    fig, ax = plot_error_by_quantile(y_true, y_score)
    assert fig is not None
    assert ax is not None


def test_plot_error_by_quantile_creates_valid_plot(regression_data):
    """Test that the function creates a valid matplotlib plot."""
    y_true, y_score = regression_data
    fig, ax = plot_error_by_quantile(y_true, y_score)
    assert hasattr(ax, 'figure')
    assert hasattr(ax, 'set_title')


def test_plot_error_by_quantile_with_custom_ax(regression_data):
    """Test that plot_error_by_quantile works with provided axes."""
    y_true, y_score = regression_data
    fig_custom, ax_custom = plt.subplots()
    fig, ax = plot_error_by_quantile(y_true, y_score, ax=ax_custom)
    assert ax is ax_custom


def test_plot_error_by_quantile_custom_quantiles(regression_data):
    """Test plot_error_by_quantile with custom number of quantiles."""
    y_true, y_score = regression_data
    fig, ax = plot_error_by_quantile(y_true, y_score, q=3)
    assert fig is not None
    assert ax is not None


def test_plot_error_by_quantile_raises_on_length_mismatch():
    """Test that plot_error_by_quantile raises ValueError on length mismatch."""
    y_true = np.array([1, 2, 3])
    y_score = np.array([1, 2])
    
    with pytest.raises(ValueError, match="y_true and y_score must have the same length"):
        plot_error_by_quantile(y_true, y_score)


def test_plot_error_by_quantile_with_list_input():
    """Test that plot_error_by_quantile works with list input."""
    y_true = [1, 2, 3, 4, 5, 6, 7, 8]
    y_score = [1.1, 2.1, 2.9, 4.1, 4.9, 6.1, 6.9, 8.1]
    fig, ax = plot_error_by_quantile(y_true, y_score)
    assert fig is not None
    assert ax is not None
