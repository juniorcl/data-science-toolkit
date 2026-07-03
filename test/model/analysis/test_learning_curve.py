import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from dstoolkit.model.analysis.learning_curve import plot_learning_curve


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    return X, y


def test_plot_learning_curve_returns_fig_and_ax(classification_data):
    """Test that plot_learning_curve returns figure and axes."""
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    fig, ax = plot_learning_curve(model, X, y, scoring='accuracy')
    assert fig is not None
    assert ax is not None


def test_plot_learning_curve_creates_valid_plot(classification_data):
    """Test that the function creates a valid matplotlib plot."""
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    fig, ax = plot_learning_curve(model, X, y, scoring='accuracy')
    assert hasattr(ax, 'figure')
    assert hasattr(ax, 'set_title')


def test_plot_learning_curve_with_custom_ax(classification_data):
    """Test that plot_learning_curve works with provided axes."""
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    fig_custom, ax_custom = plt.subplots()
    fig, ax = plot_learning_curve(model, X, y, scoring='accuracy', ax=ax_custom)
    assert ax is ax_custom


def test_plot_learning_curve_with_custom_cv(classification_data):
    """Test plot_learning_curve with custom cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    fig, ax = plot_learning_curve(model, X, y, scoring='accuracy', cv=cv)
    assert fig is not None
    assert ax is not None


def test_plot_learning_curve_labels_set(classification_data):
    """Test that axis labels are set."""
    X, y = classification_data
    model = LogisticRegression(random_state=42)
    fig, ax = plot_learning_curve(model, X, y, scoring='accuracy')
    # Check that the plot has been configured
    assert ax.get_xlabel() != ""
    assert ax.get_ylabel() != ""
