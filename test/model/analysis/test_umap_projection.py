import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.umap_projection import plot_umap_projection


@pytest.fixture
def clustering_data():
    """Create sample clustering data."""
    np.random.seed(42)
    X = np.random.randn(20, 5)
    labels = np.array([0]*7 + [1]*7 + [2]*6)
    return X, labels


def test_plot_umap_projection_returns_fig_and_ax(clustering_data):
    """Test that plot_umap_projection returns figure and axes."""
    X, labels = clustering_data
    fig, ax = plot_umap_projection(X, labels)
    assert fig is not None
    assert ax is not None


def test_plot_umap_projection_creates_valid_plot(clustering_data):
    """Test that the function creates a valid matplotlib plot."""
    X, labels = clustering_data
    fig, ax = plot_umap_projection(X, labels)
    assert hasattr(ax, 'figure')
    assert hasattr(ax, 'set_title')


def test_plot_umap_projection_ax_labels_set(clustering_data):
    """Test that axis labels are properly set."""
    X, labels = clustering_data
    fig, ax = plot_umap_projection(X, labels)
    assert ax.get_xlabel() == "Dimension 1 (UMAP)"
    assert ax.get_ylabel() == "Dimension 2 (UMAP)"


def test_plot_umap_projection_with_custom_figsize(clustering_data):
    """Test plot_umap_projection with custom figure size."""
    X, labels = clustering_data
    fig, ax = plot_umap_projection(X, labels, figsize=(10, 8))
    assert fig.get_figwidth() == 10
    assert fig.get_figheight() == 8


def test_plot_umap_projection_with_custom_n_neighbors(clustering_data):
    """Test plot_umap_projection with custom n_neighbors."""
    X, labels = clustering_data
    fig, ax = plot_umap_projection(X, labels, n_neighbors=10)
    assert fig is not None
    assert ax is not None


def test_plot_umap_projection_with_custom_min_dist(clustering_data):
    """Test plot_umap_projection with custom min_dist."""
    X, labels = clustering_data
    fig, ax = plot_umap_projection(X, labels, min_dist=0.05)
    assert fig is not None
    assert ax is not None


def test_plot_umap_projection_with_custom_metric(clustering_data):
    """Test plot_umap_projection with custom distance metric."""
    X, labels = clustering_data
    fig, ax = plot_umap_projection(X, labels, metric='manhattan')
    assert fig is not None
    assert ax is not None


def test_plot_umap_projection_with_dataframe(clustering_data):
    """Test plot_umap_projection with pandas DataFrame."""
    X, labels = clustering_data
    X_df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(X.shape[1])])
    fig, ax = plot_umap_projection(X_df, labels)
    assert fig is not None
    assert ax is not None
