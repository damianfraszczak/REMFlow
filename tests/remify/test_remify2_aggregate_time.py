"""Event-time aggregation behavior."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify


def _events(*, typed: bool = False) -> pd.DataFrame:
    events = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "actor1": ["a", "b", "a", "c", "b", "c"],
            "actor2": ["b", "a", "c", "a", "c", "b"],
        }
    )
    if typed:
        events["type"] = ["X", "Y", "X", "Y", "X", "Y"]
    return events


def test_aggregate_one_is_a_strict_time_noop():
    default = remify(_events())
    explicit = remify(_events(), aggregate_time=1)

    assert explicit.events["time"].to_list() == default.events["time"].to_list()
    assert len(explicit.events) == len(_events())


def test_aggregate_maps_to_the_next_kept_unique_time_and_clamps_tail():
    aggregated = remify(_events(), aggregate_time=2)

    assert aggregated.events["time"].to_list() == [2, 2, 4, 4, 6, 6]
    assert sorted(aggregated.events["time"].unique()) == [2, 4, 6]
    assert len(aggregated.events) == len(_events())

    tail = remify(_events(), aggregate_time=4)
    assert tail.events["time"].to_list() == [4, 4, 4, 4, 4, 4]


def test_aggregate_equal_to_unique_count_keeps_only_last_time():
    aggregated = remify(_events(), aggregate_time=6)

    assert aggregated.events["time"].to_list() == [6] * 6
    assert aggregated.events["time"].nunique() == 1


def test_aggregate_preserves_types_and_ordinal_time_is_dense():
    events = _events(typed=True)
    aggregated = remify(
        events,
        event_type="type",
        extend_riskset_by_type=True,
        aggregate_time=2,
        ordinal=True,
    )

    assert aggregated.events["time"].to_list() == [1, 1, 2, 2, 3, 3]
    assert aggregated.events["event_type"].to_list() == events["type"].to_list()


@pytest.mark.parametrize("value", [0, -1, np.nan, None, True])
def test_invalid_aggregate_time_is_rejected(value):
    with pytest.raises(ValueError, match=r"`aggregate_time` must be a single numeric value >= 1\."):
        remify(_events(), aggregate_time=value)
