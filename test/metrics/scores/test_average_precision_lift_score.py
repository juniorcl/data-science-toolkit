import pytest
import numpy as np
from dstoolkit.metrics.scores import average_precision_lift_score


def test_average_precision_lift_score_returns_float():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.9, 0.2, 0.8]

    lift = average_precision_lift_score(y_true, y_score)

    assert isinstance(lift, float)

def test_average_precision_lift_score_default_baseline():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    # baseline = 0.5
    lift = average_precision_lift_score(y_true, y_score)

    assert lift > 0

def test_average_precision_lift_score_with_explicit_baseline():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.7, 0.3, 0.9])

    lift = average_precision_lift_score(
        y_true,
        y_score,
        baseline=0.4,
    )

    assert isinstance(lift, float)

def test_average_precision_lift_score_zero_when_equal_to_baseline():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])

    lift = average_precision_lift_score(y_true, y_score)

    assert np.isclose(lift, 0.0)

def test_average_precision_lift_score_negative_when_worse_than_random():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])  # scores invertidos

    lift = average_precision_lift_score(y_true, y_score)

    assert lift < 0

def test_average_precision_lift_score_y_true_not_1d_raises():
    y_true = np.array([[0, 1], [1, 0]])
    y_score = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="y_true must be a 1D array"):
        average_precision_lift_score(y_true, y_score)

def test_average_precision_lift_score_y_score_not_1d_raises():
    y_true = np.array([0, 1])
    y_score = np.array([[0.2], [0.8]])

    with pytest.raises(ValueError, match="y_score must be a 1D array"):
        average_precision_lift_score(y_true, y_score)

def test_average_precision_lift_score_mismatched_length_raises():
    y_true = np.array([0, 1, 0])
    y_score = np.array([0.2, 0.8])

    with pytest.raises(ValueError, match="same length"):
        average_precision_lift_score(y_true, y_score)

@pytest.mark.parametrize("baseline", [0, -0.1])
def test_average_precision_lift_score_invalid_baseline_raises(baseline):
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.8, 0.3, 0.7])

    with pytest.raises(ValueError, match="Baseline must be greater than 0"):
        average_precision_lift_score(
            y_true,
            y_score,
            baseline=baseline,
        )

@pytest.mark.parametrize("container", [list, np.array])
def test_average_precision_lift_score_accepts_different_types(container):
    y_true = container([0, 1, 0, 1])
    y_score = container([0.2, 0.8, 0.3, 0.7])

    lift = average_precision_lift_score(y_true, y_score)

    assert isinstance(lift, float)