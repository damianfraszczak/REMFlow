"""Type-expanded directed actor-covariate regression tests."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


def _history():
    return remify(
        pd.DataFrame(
            {
                "time": range(1, 6),
                "actor1": [1, 1, 2, 2, 3],
                "actor2": [2, 3, 1, 3, 2],
                "type": [1, 1, 2, 2, 1],
            }
        ),
        model="tie",
        riskset="active",
        extend_riskset_by_type=True,
    )


def _attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [1, 2, 3, 1, 2, 3],
            "time": [0, 0, 0, 3, 3, 3],
            "x1": [10, 20, 30, 100, 200, 300],
            "x2": [0, 1, 1, 1, 1, 0],
        }
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


def test_type_expanded_directed_actor_covariates_match_expected_matrices():
    result = _statistics()
    expected = {
        "send_x1": [[10, 10, 30, 20, 20]] * 2 + [[100, 100, 300, 200, 200]] * 3,
        "receive_x1": [[20, 30, 20, 10, 30]] * 2 + [[200, 300, 200, 100, 300]] * 3,
        "average_x1": [[15, 20, 25, 15, 25]] * 2 + [[150, 200, 250, 150, 250]] * 3,
        "difference_x1": [[10, 20, 10, 10, 10]] * 2 + [[100, 200, 100, 100, 100]] * 3,
        "maximum_x1": [[20, 30, 30, 20, 30]] * 2 + [[200, 300, 300, 200, 300]] * 3,
        "minimum_x1": [[10, 10, 20, 10, 20]] * 2 + [[100, 100, 200, 100, 200]] * 3,
        "same_x2": [[0, 0, 1, 0, 1]] * 2 + [[1, 0, 0, 1, 0]] * 3,
    }

    assert result.names == ["baseline", *expected]
    baseline = _tensor(result, "baseline")
    np.testing.assert_array_equal(baseline, np.ones_like(baseline))
    for name, values in expected.items():
        np.testing.assert_array_equal(_tensor(result, name), np.asarray(values))


def test_type_expanded_actor_covariate_standardization_matches_definition():
    raw = _statistics()
    with pytest.warns(DeprecationWarning, match="attr_actors"):
        standardized = remstats(
            _history(),
            tie_effects=(
                '~ send(variable="x1", scaling="std") + '
                'receive(variable="x1", scaling="std") + '
                'average(variable="x1", scaling="std") + '
                'difference(variable="x1", scaling="std") + '
                'maximum(variable="x1", scaling="std") + '
                'minimum(variable="x1", scaling="std")'
            ),
            attr_actors=_attributes(),
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
