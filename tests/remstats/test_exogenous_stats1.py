"""Directed actor-covariate statistic regression tests."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


def _attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [1, 2, 3, 1, 2, 3],
            "time": [0, 0, 0, 3, 3, 3],
            "x1": [10, 20, 30, 100, 200, 300],
            "x2": [0, 1, 1, 1, 1, 0],
        }
    )


def _history(*, simultaneous: bool = False):
    times = [1, 2, 3, 3, 4] if simultaneous else range(1, 6)
    return remify(
        pd.DataFrame(
            {
                "time": times,
                "actor1": [1, 1, 2, 2, 3],
                "actor2": [2, 3, 1, 3, 2],
            }
        ),
        model="tie",
        riskset="active",
    )


def _tensor(result, name: str) -> np.ndarray:
    column = result.names.index(name)
    return np.stack([matrix[:, column] for matrix in result.stats])


def _statistics():
    with pytest.warns(DeprecationWarning, match="attr_actors"):
        return remstats(
            _history(),
            tie_effects=(
                '~ send(variable="x1") + receive(variable="x1") + '
                'average(variable="x1") + difference(variable="x1") + '
                'maximum(variable="x1") + minimum(variable="x1") + '
                'same(variable="x2")'
            ),
            attr_actors=_attributes(),
            first=1,
        )


def test_time_varying_actor_covariates_match_expected_matrices():
    result = _statistics()
    expected = {
        "send_x1": [
            [10, 10, 20, 20, 30],
            [10, 10, 20, 20, 30],
            [100, 100, 200, 200, 300],
            [100, 100, 200, 200, 300],
            [100, 100, 200, 200, 300],
        ],
        "receive_x1": [
            [20, 30, 10, 30, 20],
            [20, 30, 10, 30, 20],
            [200, 300, 100, 300, 200],
            [200, 300, 100, 300, 200],
            [200, 300, 100, 300, 200],
        ],
        "average_x1": [
            [15, 20, 15, 25, 25],
            [15, 20, 15, 25, 25],
            [150, 200, 150, 250, 250],
            [150, 200, 150, 250, 250],
            [150, 200, 150, 250, 250],
        ],
        "difference_x1": [
            [10, 20, 10, 10, 10],
            [10, 20, 10, 10, 10],
            [100, 200, 100, 100, 100],
            [100, 200, 100, 100, 100],
            [100, 200, 100, 100, 100],
        ],
        "maximum_x1": [
            [20, 30, 20, 30, 30],
            [20, 30, 20, 30, 30],
            [200, 300, 200, 300, 300],
            [200, 300, 200, 300, 300],
            [200, 300, 200, 300, 300],
        ],
        "minimum_x1": [
            [10, 10, 10, 20, 20],
            [10, 10, 10, 20, 20],
            [100, 100, 100, 200, 200],
            [100, 100, 100, 200, 200],
            [100, 100, 100, 200, 200],
        ],
        "same_x2": [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
        ],
    }

    assert result.names == ["baseline", *expected]
    baseline = _tensor(result, "baseline")
    np.testing.assert_array_equal(baseline, np.ones_like(baseline))
    for name, values in expected.items():
        np.testing.assert_array_equal(_tensor(result, name), np.asarray(values))


def test_signed_actor_difference_and_standardization_match_definitions():
    attributes = _attributes()
    history = _history()
    with pytest.warns(DeprecationWarning, match="attr_actors"):
        signed = remstats(
            history,
            tie_effects='~ difference(variable="x1", absolute=FALSE)',
            attr_actors=attributes,
            first=1,
        )
    expected_signed = np.array(
        [
            [-10, -20, 10, -10, 10],
            [-10, -20, 10, -10, 10],
            [-100, -200, 100, -100, 100],
            [-100, -200, 100, -100, 100],
            [-100, -200, 100, -100, 100],
        ],
        dtype=float,
    )
    np.testing.assert_array_equal(_tensor(signed, "difference_x1"), expected_signed)

    raw = _statistics()
    with pytest.warns(DeprecationWarning, match="attr_actors"):
        standardized = remstats(
            history,
            tie_effects=(
                '~ send(variable="x1", scaling="std") + '
                'receive(variable="x1", scaling="std") + '
                'average(variable="x1", scaling="std") + '
                'difference(variable="x1", scaling="std") + '
                'maximum(variable="x1", scaling="std") + '
                'minimum(variable="x1", scaling="std")'
            ),
            attr_actors=attributes,
            first=1,
        )
    for name in standardized.names[1:]:
        values = _tensor(raw, name)
        deviation = values.std(axis=1, ddof=1, keepdims=True)
        expected = np.divide(
            values - values.mean(axis=1, keepdims=True),
            deviation,
            out=np.zeros_like(values),
            where=deviation != 0,
        )
        np.testing.assert_allclose(_tensor(standardized, name), expected, atol=1e-14)


def test_time_point_grouping_reuses_actor_covariates_for_simultaneous_events():
    with pytest.warns(DeprecationWarning, match="attr_actors"):
        result = remstats(
            _history(simultaneous=True),
            tie_effects='~ send(variable="x1") + average(variable="x1")',
            attr_actors=_attributes(),
            first=1,
        )
    expected_send = np.array(
        [
            [10, 10, 20, 20, 30],
            [10, 10, 20, 20, 30],
            [100, 100, 200, 200, 300],
            [100, 100, 200, 200, 300],
        ],
        dtype=float,
    )
    expected_average = np.array(
        [
            [15, 20, 15, 25, 25],
            [15, 20, 15, 25, 25],
            [150, 200, 150, 250, 250],
            [150, 200, 150, 250, 250],
        ],
        dtype=float,
    )

    assert result.event_indices == [0, 1, 2, 4]
    np.testing.assert_array_equal(_tensor(result, "send_x1"), expected_send)
    np.testing.assert_array_equal(_tensor(result, "average_x1"), expected_average)
