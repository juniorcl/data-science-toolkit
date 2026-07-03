import pytest
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.cluster_size import plot_cluster_sizes


@pytest.fixture
def cluster_labels():
    """Create sample cluster labels."""
    return np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])


def test_plot_cluster_sizes_returns_fig_and_ax(cluster_labels):
    """Test that plot_cluster_sizes returns figure and axes."""
    fig, ax = plot_cluster_sizes(cluster_labels)
    assert fig is not None
    assert ax is not None


def test_plot_cluster_sizes_creates_matplotlib_plot(cluster_labels):
    """Test that the function creates a valid matplotlib plot."""
    fig, ax = plot_cluster_sizes(cluster_labels)
    assert hasattr(ax, 'figure')
    assert hasattr(ax, 'set_title')


def test_plot_cluster_sizes_with_custom_ax(cluster_labels):
    """Test that plot_cluster_sizes works with provided axes."""
    fig_custom, ax_custom = plt.subplots()
    fig, ax = plot_cluster_sizes(cluster_labels, ax=ax_custom)
    assert ax is ax_custom


def test_plot_cluster_sizes_raises_on_empty_labels():
    """Test that plot_cluster_sizes raises ValueError on empty labels."""
    with pytest.raises(ValueError, match="labels must be a non-empty array-like"):
        plot_cluster_sizes([])


def test_plot_cluster_sizes_raises_on_none_labels():
    """Test that plot_cluster_sizes raises ValueError on None labels."""
    with pytest.raises(ValueError, match="labels must be a non-empty array-like"):
        plot_cluster_sizes(None)


def test_plot_cluster_sizes_with_list_input():
    """Test that plot_cluster_sizes works with list input."""
    labels = [0, 0, 1, 1, 2]
    fig, ax = plot_cluster_sizes(labels)
    assert fig is not None
    assert ax is not None


def test_plot_cluster_sizes_ax_labels_set(cluster_labels):
    """Test that axis labels are properly set."""
    fig, ax = plot_cluster_sizes(cluster_labels)
    assert ax.get_xlabel() == "Cluster"
    assert ax.get_ylabel() == "Number of Observations"
