"""Typed-effect regression and partition tests."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


@pytest.fixture
def typed_events():
    return pd.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "actor1": [1, 1, 2, 1],
            "actor2": [2, 2, 1, 2],
            "type": ["work", "social", "work", "social"],
            "weight": [2.0, 3.0, 1.0, 4.0],
        }
    )


def test_separate_type_statistics_have_alphabetical_names_and_values(typed_events):
    history = remify(typed_events, actors=[1, 2, 3], extend_riskset_by_type=True)
    result = remstats(
        history,
        tie_effects='~ inertia(consider_type="separate")',
        first=1,
    )

    assert history.event_types == ["social", "work"]
    assert result.names == ["baseline", "inertia.social", "inertia.work"]
    riskset = history.risksets[2]
    rows = np.flatnonzero(
        ((riskset["sender"] == 1) & (riskset["receiver"] == 2)).to_numpy()
    )
    assert len(rows) == 2
    social_index = result.names.index("inertia.social")
    work_index = result.names.index("inertia.work")
    np.testing.assert_array_equal(result.stats[2][rows, social_index], [3.0, 3.0])
    np.testing.assert_array_equal(result.stats[2][rows, work_index], [2.0, 2.0])


def test_interact_type_statistics_mask_candidate_type(typed_events):
    history = remify(typed_events, actors=[1, 2, 3], extend_riskset_by_type=True)
    result = remstats(
        history,
        tie_effects='~ inertia(consider_type="interact")',
        first=1,
    )

    assert result.names == [
        "baseline",
        "inertia.social.social",
        "inertia.social.work",
        "inertia.work.social",
        "inertia.work.work",
    ]
    riskset = history.risksets[2]
    social_row = riskset.index[
        (riskset["sender"] == 1)
        & (riskset["receiver"] == 2)
        & (riskset["event_type"] == "social")
    ][0]
    work_row = riskset.index[
        (riskset["sender"] == 1)
        & (riskset["receiver"] == 2)
        & (riskset["event_type"] == "work")
    ][0]
    np.testing.assert_array_equal(result.stats[2][social_row], [1, 3, 0, 2, 0])
    np.testing.assert_array_equal(result.stats[2][work_row], [1, 0, 3, 0, 2])


def test_interact_without_type_expanded_riskset_reduces_to_separate(typed_events):
    history = remify(typed_events, actors=[1, 2, 3], extend_riskset_by_type=False)
    separate = remstats(history, tie_effects="~ inertia(consider_type=TRUE)", first=1)
    interact = remstats(
        history,
        tie_effects='~ inertia(consider_type="interact")',
        first=1,
    )

    assert interact.names == ["baseline", "inertia.social", "inertia.work"]
    for actual, expected in zip(interact.stats, separate.stats, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_ignore_type_aggregates_weighted_history(typed_events):
    history = remify(typed_events, actors=[1, 2, 3], extend_riskset_by_type=True)
    result = remstats(history, tie_effects="~ inertia(consider_type=FALSE)", first=1)
    riskset = history.risksets[2]
    rows = np.flatnonzero(
        ((riskset["sender"] == 1) & (riskset["receiver"] == 2)).to_numpy()
    )

    assert result.names == ["baseline", "inertia"]
    np.testing.assert_array_equal(result.stats[2][rows, 1], [5.0, 5.0])


def test_invalid_consider_type_fails_before_computation(typed_events):
    history = remify(typed_events)
    with pytest.raises(ValueError, match="consider_type"):
        remstats(history, tie_effects='~ inertia(consider_type="invalid")')
