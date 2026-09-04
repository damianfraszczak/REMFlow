"""Typed undirected endogenous-statistic regression tests."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


def _history():
    return remify(
        pd.DataFrame(
            {
                "time": range(1, 11),
                "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
                "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
                "type": [1, 1, 2, 2, 1, 2, 2, 1, 1, 1],
            }
        ),
        model="tie",
        directed=False,
        riskset="active",
        extend_riskset_by_type=True,
    )


def _tensor(result, name: str) -> np.ndarray:
    column = result.names.index(name)
    return np.stack([matrix[:, column] for matrix in result.stats])


def _ignored_statistics():
    return remstats(
        _history(),
        tie_effects=(
            "~ FEtype() + degreeDiff() + degreeMin() + degreeMax() + "
            "totaldegreeDyad() + inertia() + sp() + sp(unique=TRUE) + "
            "psABAB() + psABAY()"
        ),
        first=1,
    )


def test_typed_undirected_ignore_effects_match_expected_matrices():
    result = _ignored_statistics()
    expected = {
        "degreeMin": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 2, 0, 1, 0, 1, 2, 1, 0],
            [2, 3, 0, 2, 0, 2, 3, 2, 0],
            [3, 3, 0, 3, 0, 3, 3, 3, 0],
            [3, 3, 1, 3, 1, 3, 3, 3, 1],
            [4, 4, 1, 4, 1, 4, 4, 4, 1],
            [4, 4, 1, 5, 1, 4, 4, 5, 1],
            [4, 4, 2, 6, 2, 4, 4, 6, 2],
        ],
        "degreeMax": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [2, 2, 2, 1, 1, 2, 2, 1, 1],
            [3, 3, 3, 2, 1, 3, 3, 2, 2],
            [3, 3, 3, 3, 2, 3, 3, 3, 3],
            [3, 4, 3, 4, 3, 3, 4, 4, 4],
            [3, 5, 3, 5, 3, 3, 5, 5, 5],
            [4, 5, 4, 5, 4, 4, 5, 5, 5],
            [5, 6, 4, 6, 5, 5, 6, 6, 6],
            [6, 6, 4, 6, 6, 6, 6, 6, 6],
        ],
        "totaldegreeDyad": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 1, 1, 0, 1, 2, 1, 1],
            [3, 3, 2, 2, 1, 3, 3, 2, 1],
            [4, 5, 3, 3, 1, 4, 5, 3, 2],
            [5, 6, 3, 5, 2, 5, 6, 5, 3],
            [6, 7, 3, 7, 3, 6, 7, 7, 4],
            [6, 8, 4, 8, 4, 6, 8, 8, 6],
            [8, 9, 5, 9, 5, 8, 9, 9, 6],
            [9, 10, 5, 11, 6, 9, 10, 11, 7],
            [10, 10, 6, 12, 8, 10, 10, 12, 8],
        ],
        "inertia": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0],
            [1, 2, 0, 0, 0, 1, 2, 0, 0],
            [1, 2, 0, 1, 0, 1, 2, 1, 0],
            [1, 2, 0, 2, 0, 1, 2, 2, 0],
            [1, 2, 0, 2, 0, 1, 2, 2, 1],
            [2, 2, 0, 2, 0, 2, 2, 2, 1],
            [2, 2, 0, 3, 0, 2, 2, 3, 1],
            [2, 2, 0, 3, 1, 2, 2, 3, 1],
        ],
        "sp": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [2, 1, 0, 1, 0, 2, 1, 1, 0],
            [2, 1, 1, 1, 1, 2, 1, 1, 0],
            [2, 2, 1, 2, 1, 2, 2, 2, 0],
            [2, 2, 1, 2, 1, 2, 2, 2, 0],
            [2, 2, 2, 3, 1, 2, 2, 3, 1],
        ],
        "sp.unique": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 2, 2, 1, 1, 1, 2, 1],
        ],
        "psABAB": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        "psABAY": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 1],
        ],
    }

    assert result.names == [
        "baseline",
        "FEtype_2",
        "degreeDiff",
        "degreeMin",
        "degreeMax",
        "totaldegreeDyad",
        "inertia",
        "sp",
        "sp.unique",
        "psABAB",
        "psABAY",
    ]
    baseline = _tensor(result, "baseline")
    np.testing.assert_array_equal(baseline, np.ones_like(baseline))
    expected_type = np.zeros_like(baseline)
    expected_type[:, 5:] = 1.0
    np.testing.assert_array_equal(_tensor(result, "FEtype_2"), expected_type)
    for name, values in expected.items():
        np.testing.assert_array_equal(_tensor(result, name), np.asarray(values))
    np.testing.assert_array_equal(
        _tensor(result, "degreeDiff"),
        np.asarray(expected["degreeMax"]) - np.asarray(expected["degreeMin"]),
    )


def test_typed_separate_effects_match_expected_type_slices():
    ignored = _ignored_statistics()
    separate = remstats(
        _history(),
        tie_effects=(
            '~ inertia(consider_type="separate") + '
            'degreeMin(consider_type="separate")'
        ),
        first=1,
    )
    expected_degree_minimum_type_1 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [2, 2, 0, 2, 0, 2, 2, 2, 0],
            [2, 2, 0, 2, 0, 2, 2, 2, 0],
            [2, 2, 0, 2, 0, 2, 2, 2, 0],
            [2, 2, 0, 3, 0, 2, 2, 3, 0],
            [2, 2, 1, 3, 1, 2, 2, 3, 1],
        ],
        dtype=float,
    )
    expected_inertia_type_1 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0],
            [1, 1, 0, 2, 0, 1, 1, 2, 0],
            [1, 1, 0, 2, 1, 1, 1, 2, 0],
        ],
        dtype=float,
    )

    assert separate.names == [
        "baseline",
        "inertia.1",
        "inertia.2",
        "degreeMin.1",
        "degreeMin.2",
    ]
    np.testing.assert_array_equal(
        _tensor(separate, "inertia.1") + _tensor(separate, "inertia.2"),
        _tensor(ignored, "inertia"),
    )
    np.testing.assert_array_equal(
        _tensor(separate, "degreeMin.1"), expected_degree_minimum_type_1
    )
    np.testing.assert_array_equal(
        _tensor(separate, "inertia.1"), expected_inertia_type_1
    )


def test_typed_undirected_scaling_matches_documented_rules():
    raw = _ignored_statistics()
    history = _history()
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


def test_fetype_uses_one_contrast_per_nonbaseline_event_type():
    events = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "actor1": [1, 2, 3],
            "actor2": [2, 3, 1],
            "type": [1, 2, 3],
        }
    )
    history = remify(
        events,
        actors=[1, 2, 3],
        extend_riskset_by_type=True,
    )
    result = remstats(history, tie_effects="~ FEtype()", first=1)

    assert result.names == ["baseline", "FEtype_2", "FEtype_3"]
    for event, riskset in enumerate(history.risksets):
        np.testing.assert_array_equal(
            result.stats[event][:, 1],
            (riskset["event_type"] == 2).to_numpy(dtype=float),
        )
        np.testing.assert_array_equal(
            result.stats[event][:, 2],
            (riskset["event_type"] == 3).to_numpy(dtype=float),
        )
