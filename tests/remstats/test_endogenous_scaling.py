"""Endogenous-statistic scaling regression cases."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats, remstimate


def _undirected_history():
    return remify(
        pd.DataFrame(
            {
                "time": range(1, 11),
                "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
                "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
            }
        ),
        directed=False,
        riskset="active",
    )


def _tensor(result, name):
    index = result.names.index(name)
    return np.stack([matrix[:, index] for matrix in result.stats])


def test_unique_shared_partners_match_hand_calculation():
    result = remstats(
        _undirected_history(),
        tie_effects="~ sp() + sp(unique=TRUE)",
        first=1,
    )

    assert result.names == ["baseline", "sp", "sp.unique"]
    expected = np.array(
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
    np.testing.assert_array_equal(_tensor(result, "sp.unique"), expected)


def test_standard_scaling_is_applied_per_event_risk_set():
    history = _undirected_history()
    raw = remstats(history, tie_effects="~ degreeMin() + sp(unique=TRUE)", first=1)
    scaled = remstats(
        history,
        tie_effects='~ degreeMin(scaling="std") + sp(scaling="std", unique=TRUE)',
        first=1,
    )

    assert scaled.names == ["baseline", "degreeMin", "sp.unique"]
    for name in ("degreeMin", "sp.unique"):
        values = _tensor(raw, name)
        means = values.mean(axis=1, keepdims=True)
        standard_deviations = values.std(axis=1, ddof=1, keepdims=True)
        expected = np.divide(
            values - means,
            standard_deviations,
            out=np.zeros_like(values),
            where=standard_deviations != 0,
        )
        np.testing.assert_allclose(_tensor(scaled, name), expected, atol=1e-14)


def test_undirected_proportional_degree_scaling_and_inertia_error():
    history = _undirected_history()
    raw = remstats(history, tie_effects="~ degreeMin() + totaldegreeDyad()", first=1)
    scaled = remstats(
        history,
        tie_effects='~ degreeMin(scaling="prop") + totaldegreeDyad(scaling="prop")',
        first=1,
    )

    event_denominator = np.arange(10, dtype=float)[:, None]
    expected_minimum = np.divide(
        _tensor(raw, "degreeMin"),
        event_denominator,
        out=np.full_like(_tensor(raw, "degreeMin"), 0.25),
        where=event_denominator != 0,
    )
    expected_total = np.divide(
        _tensor(raw, "totaldegreeDyad"),
        2 * event_denominator,
        out=np.full_like(_tensor(raw, "totaldegreeDyad"), 0.25),
        where=event_denominator != 0,
    )
    np.testing.assert_allclose(_tensor(scaled, "degreeMin"), expected_minimum)
    np.testing.assert_allclose(_tensor(scaled, "totaldegreeDyad"), expected_total)

    with pytest.raises(ValueError, match="not defined"):
        remstats(history, tie_effects='~ inertia(scaling="prop")', first=1)


def test_directed_proportional_inertia_uses_sender_activity():
    history = remify(
        pd.DataFrame(
            {
                "time": range(1, 6),
                "actor1": [1, 1, 2, 1, 3],
                "actor2": [2, 3, 1, 2, 1],
            }
        ),
        actors=[1, 2, 3, 4],
        riskset="full",
    )
    raw = remstats(history, tie_effects="~ inertia() + outdegreeSender()", first=1)
    scaled = remstats(history, tie_effects='~ inertia(scaling="prop")', first=1)

    inertia_values = _tensor(raw, "inertia")
    sender_activity = _tensor(raw, "outdegreeSender")
    expected = np.divide(
        inertia_values,
        sender_activity,
        out=np.full_like(inertia_values, 1 / 3),
        where=sender_activity != 0,
    )
    np.testing.assert_allclose(_tensor(scaled, "inertia"), expected)


def test_unique_argument_is_restricted_to_shared_partner_effects():
    with pytest.raises(TypeError, match="unique is not defined"):
        remstats(_undirected_history(), tie_effects="~ inertia(unique=TRUE)", first=1)


def test_simultaneous_events_share_one_point_time_statistic_state():
    history = remify(
        pd.DataFrame(
            {
                "time": [1, 2, 3, 4, 5, 5, 5, 6, 7, 8],
                "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
                "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
            }
        ),
        directed=False,
        riskset="active",
    )
    result = remstats(history, tie_effects="~ degreeMin() + sp()", first=1)

    assert len(result.stats) == 8
    assert result.event_indices == [0, 1, 2, 3, 4, 7, 8, 9]
    assert [len(group) for group in result.observed_index_groups] == [1, 1, 1, 1, 3, 1, 1, 1]
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
    np.testing.assert_array_equal(_tensor(result, "degreeMin"), expected_degree_minimum)

    fitted = remstimate(history, result)
    assert np.isfinite(fitted.log_likelihood)
    assert fitted.gradient is not None
    assert fitted.hessian is not None
    assert len(fitted.event_probabilities) == history.E
