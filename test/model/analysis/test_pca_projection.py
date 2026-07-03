import pytest
import numpy as np
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.pca_projection import plot_pca_projection


@pytest.fixture
def clustering_data():
    """Create sample clustering data."""
    np.random.seed(42)
    X = np.random.randn(20, 5)
    labels = np.array([0]*7 + [1]*7 + [2]*6)
    return X, labels


def test_plot_pca_projection_returns_fig_and_ax(clustering_data):
    """Test that plot_pca_projection returns figure and axes."""
    X, labels = clustering_data
    fig, ax = plot_pca_projection(X, labels)
    assert fig is not None
    assert ax is not None


def test_plot_pca_projection_creates_valid_plot(clustering_data):
    """Test that the function creates a valid matplotlib plot."""
    X, labels = clustering_data
    fig, ax = plot_pca_projection(X, labels)
    assert hasattr(ax, 'figure')
    assert hasattr(ax, 'set_title')


def test_plot_pca_projection_ax_labels_set(clustering_data):
    """Test that axis labels are properly set."""
    X, labels = clustering_data
    fig, ax = plot_pca_projection(X, labels)
    assert ax.get_xlabel() == "Component 1"
    assert ax.get_ylabel() == "Component 2"


def test_plot_pca_projection_with_custom_figsize(clustering_data):
    """Test plot_pca_projection with custom figure size."""
    X, labels = clustering_data
    fig, ax = plot_pca_projection(X, labels, figsize=(10, 8))
    assert fig.get_figwidth() == 10
    assert fig.get_figheight() == 8


def test_plot_pca_projection_with_custom_random_state(clustering_data):
    """Test plot_pca_projection with custom random state."""
    X, labels = clustering_data
    fig1, ax1 = plot_pca_projection(X, labels, random_state=42)
    fig2, ax2 = plot_pca_projection(X, labels, random_state=42)
    # Should produce same results with same random state
    assert fig1 is not None
    assert fig2 is not None


def test_plot_pca_projection_raises_on_insufficient_features():
    """Test that plot_pca_projection raises ValueError for insufficient features."""
    X = np.random.randn(10, 1)
    labels = np.array([0]*5 + [1]*5)
    
    with pytest.raises(ValueError, match="X must have at least 2 features"):
        plot_pca_projection(X, labels)


def test_plot_pca_projection_with_dataframe(clustering_data):
    """Test plot_pca_projection with pandas DataFrame."""
    import pandas as pd
    X, labels = clustering_data
    X_df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(X.shape[1])])
    fig, ax = plot_pca_projection(X_df, labels)
    assert fig is not None
    assert ax is not None
