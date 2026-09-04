"""Undirected endogenous-statistic regression tests."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


def _history(*, simultaneous: bool = False):
    times = [1, 2, 3, 4, 5, 5, 5, 6, 7, 8] if simultaneous else range(1, 11)
    return remify(
        pd.DataFrame(
            {
                "time": times,
                "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
                "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
            }
        ),
        model="tie",
        directed=False,
        riskset="active",
    )


def _tensor(result, name: str) -> np.ndarray:
    column = result.names.index(name)
    return np.stack([matrix[:, column] for matrix in result.stats])


def test_all_undirected_active_effects_match_hand_calculations():
    result = remstats(
        _history(),
        tie_effects=(
            "~ degreeDiff() + degreeMin() + degreeMax() + totaldegreeDyad() + "
            "inertia() + sp() + sp(unique=TRUE) + psABAB() + psABAY()"
        ),
        first=1,
    )

    expected_degree_minimum = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 2, 0, 1, 0, 0],
            [2, 3, 0, 2, 0, 0],
            [3, 3, 0, 3, 0, 0],
            [3, 3, 1, 3, 1, 1],
            [4, 4, 1, 4, 1, 1],
            [4, 4, 1, 5, 1, 1],
            [4, 4, 2, 6, 2, 2],
        ],
        dtype=float,
    )
    expected_degree_maximum = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [2, 2, 2, 1, 1, 1],
            [3, 3, 3, 2, 1, 2],
            [3, 3, 3, 3, 2, 3],
            [3, 4, 3, 4, 3, 4],
            [3, 5, 3, 5, 3, 5],
            [4, 5, 4, 5, 4, 5],
            [5, 6, 4, 6, 5, 6],
            [6, 6, 4, 6, 6, 6],
        ],
        dtype=float,
    )
    expected_total_degree = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 2, 1, 1, 0, 1],
            [3, 3, 2, 2, 1, 1],
            [4, 5, 3, 3, 1, 2],
            [5, 6, 3, 5, 2, 3],
            [6, 7, 3, 7, 3, 4],
            [6, 8, 4, 8, 4, 6],
            [8, 9, 5, 9, 5, 6],
            [9, 10, 5, 11, 6, 7],
            [10, 10, 6, 12, 8, 8],
        ],
        dtype=float,
    )
    expected_inertia = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0],
            [1, 2, 0, 1, 0, 0],
            [1, 2, 0, 2, 0, 0],
            [1, 2, 0, 2, 0, 1],
            [2, 2, 0, 2, 0, 1],
            [2, 2, 0, 3, 0, 1],
            [2, 2, 0, 3, 1, 1],
        ],
        dtype=float,
    )
    expected_shared_partners = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [2, 1, 0, 1, 0, 0],
            [2, 1, 1, 1, 1, 0],
            [2, 2, 1, 2, 1, 0],
            [2, 2, 1, 2, 1, 0],
            [2, 2, 2, 3, 1, 1],
        ],
        dtype=float,
    )
    expected_unique_shared_partners = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 2, 2, 1, 1],
        ],
        dtype=float,
    )
    expected_ps_abab = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=float,
    )
    expected_ps_abay = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1],
        ],
        dtype=float,
    )

    np.testing.assert_array_equal(_tensor(result, "baseline"), 1.0)
    np.testing.assert_array_equal(
        _tensor(result, "degreeMin"), expected_degree_minimum
    )
    np.testing.assert_array_equal(
        _tensor(result, "degreeMax"), expected_degree_maximum
    )
    np.testing.assert_array_equal(
        _tensor(result, "degreeDiff"),
        expected_degree_maximum - expected_degree_minimum,
    )
    np.testing.assert_array_equal(
        _tensor(result, "totaldegreeDyad"), expected_total_degree
    )
    np.testing.assert_array_equal(_tensor(result, "inertia"), expected_inertia)
    np.testing.assert_array_equal(
        _tensor(result, "sp"), expected_shared_partners
    )
    np.testing.assert_array_equal(
        _tensor(result, "sp.unique"), expected_unique_shared_partners
    )
    np.testing.assert_array_equal(_tensor(result, "psABAB"), expected_ps_abab)
    np.testing.assert_array_equal(_tensor(result, "psABAY"), expected_ps_abay)


def test_undirected_standard_and_proportional_scaling_match_definitions():
    history = _history()
    raw = remstats(
        history,
        tie_effects=(
            "~ degreeMin() + degreeMax() + degreeDiff() + totaldegreeDyad() + "
            "inertia() + sp() + sp(unique=TRUE)"
        ),
        first=1,
    )
    standardized = remstats(
        history,
        tie_effects=(
            '~ degreeMin(scaling="std") + degreeMax(scaling="std") + '
            'degreeDiff(scaling="std") + totaldegreeDyad(scaling="std") + '
            'inertia(scaling="std") + sp(scaling="std") + '
            'sp(scaling="std", unique=TRUE)'
        ),
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

    with pytest.raises(ValueError, match="not defined"):
        remstats(history, tie_effects='~ inertia(scaling="prop")', first=1)

    proportional = remstats(
        history,
        tie_effects=(
            '~ degreeMin(scaling="prop") + degreeMax(scaling="prop") + '
            'totaldegreeDyad(scaling="prop")'
        ),
        first=1,
    )
    prior_events = np.arange(10, dtype=float)[:, None]
    for name in ("degreeMin", "degreeMax"):
        expected = np.divide(
            _tensor(raw, name),
            prior_events,
            out=np.full_like(_tensor(raw, name), 0.25),
            where=prior_events != 0,
        )
        np.testing.assert_array_equal(_tensor(proportional, name), expected)
    expected_total = np.divide(
        _tensor(raw, "totaldegreeDyad"),
        2 * prior_events,
        out=np.full_like(_tensor(raw, "totaldegreeDyad"), 0.25),
        where=prior_events != 0,
    )
    np.testing.assert_array_equal(
        _tensor(proportional, "totaldegreeDyad"), expected_total
    )


def test_simultaneous_point_time_statistics_match_expected_matrices():
    result = remstats(
        _history(simultaneous=True),
        tie_effects="~ degreeMin() + sp()",
        first=1,
    )
    expected_degree_minimum = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 2, 0, 1, 0, 0],
            [2, 3, 0, 2, 0, 0],
            [4, 4, 1, 4, 1, 1],
            [4, 4, 1, 5, 1, 1],
            [4, 4, 2, 6, 2, 2],
        ],
        dtype=float,
    )
    expected_shared_partners = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [2, 2, 1, 2, 1, 0],
            [2, 2, 1, 2, 1, 0],
            [2, 2, 2, 3, 1, 1],
        ],
        dtype=float,
    )

    assert result.event_indices == [0, 1, 2, 3, 4, 7, 8, 9]
    np.testing.assert_array_equal(
        _tensor(result, "degreeMin"), expected_degree_minimum
    )
    np.testing.assert_array_equal(
        _tensor(result, "sp"), expected_shared_partners
    )
