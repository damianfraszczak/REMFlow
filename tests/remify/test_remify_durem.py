"""Duration event-history behavior."""

import numpy as np
import pandas as pd
import pytest

from remflow import DurationHistory, is_remify_durem, remify


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 3, 6, 9],
            "actor1": ["A", "B", "A", "C"],
            "actor2": ["B", "C", "C", "A"],
            "end": [2, 9, 8, 11],
        }
    )


def test_duration_class_metadata_and_end_column():
    history = remify(_events(), duration=True, model="tie")

    assert isinstance(history, DurationHistory)
    assert is_remify_durem(history)
    assert history.durem == {
        "n_complete": 4,
        "n_censored": 0,
        "has_censored": False,
        "dur_directed_end": False,
        "dur_type_exclusive": False,
        "has_who_ended": False,
    }
    assert history.events["end"].to_list() == [2.0, 9.0, 8.0, 11.0]


def test_legacy_and_duration_columns_are_normalized():
    legacy = pd.DataFrame(
        {
            "start_time": [1, 3, 6],
            "sender": ["A", "B", "A"],
            "receiver": ["B", "C", "C"],
            "end_time": [2, 5, 8],
        }
    )
    duration_column = pd.DataFrame(
        {
            "time": [1, 3, 6],
            "actor1": ["A", "B", "A"],
            "actor2": ["B", "C", "C"],
            "duration": [1, 2, 2],
        }
    )

    legacy_history = remify(legacy, duration=True, model="tie")
    duration_history = remify(duration_column, duration=True, model="tie")

    assert {"time", "sender", "receiver", "end"}.issubset(legacy_history.events)
    assert duration_history.events["end"].to_list() == [2.0, 5.0, 8.0]


def test_duration_validation_rejects_invalid_end_and_terminator():
    invalid_end = _events()
    invalid_end.loc[1, "end"] = invalid_end.loc[1, "time"] - 1
    with pytest.raises(ValueError, match="End time cannot be before start time"):
        remify(invalid_end, duration=True, model="tie")

    invalid_who = _events().assign(who_ended=["both", "actor1", "actor2", "actor1"])
    with pytest.raises(ValueError, match="who_ended"):
        remify(invalid_who, duration=True, dur_directed_end=True, model="tie")


def test_directed_end_with_and_without_terminator_column():
    with_who = _events().assign(who_ended=["actor1", "actor2", "actor1", "actor2"])
    history = remify(with_who, duration=True, dur_directed_end=True, model="tie")

    assert history.durem["dur_directed_end"] is True
    assert history.durem["has_who_ended"] is True
    assert history.events["who_ended"].to_list() == with_who["who_ended"].to_list()

    with pytest.warns(UserWarning, match="who_ended"):
        assumed = remify(_events(), duration=True, dur_directed_end=True, model="tie")
    assert assumed.events["who_ended"].to_list() == ["actor1"] * 4


def test_undirected_censoring_typing_and_exclusivity():
    undirected = remify(_events(), duration=True, directed=False, model="tie")
    assert undirected.directed is False

    censored_events = _events().iloc[:3].copy()
    censored_events.loc[1, "end"] = np.nan
    censored = remify(censored_events, duration=True, model="tie")
    assert censored.durem["has_censored"] is True
    assert censored.durem["n_censored"] == 1
    assert censored.durem["n_complete"] == 2

    with pytest.warns(UserWarning, match="dur_type_exclusive"):
        ignored = remify(_events(), duration=True, dur_type_exclusive=True, model="tie")
    assert ignored.durem["dur_type_exclusive"] is False

    typed_events = _events().assign(type=["X", "Y", "X", "Y"])
    exclusive = remify(
        typed_events,
        duration=True,
        dur_type_exclusive=True,
        extend_riskset_by_type=True,
        model="tie",
    )
    assert exclusive.C == 2
    assert exclusive.durem["dur_type_exclusive"] is True
    assert exclusive.extend_riskset_by_type is True


def test_default_model_warning_plain_predicate_and_duration_print_summary():
    with pytest.warns(UserWarning, match="model set to 'tie'"):
        duration = remify(_events(), duration=True)
    plain = remify(_events().drop(columns="end"))

    assert not is_remify_durem(plain)
    assert "duration events: complete=4, censored=0" in str(duration)
    assert duration.summary()["duration"]["n_complete"] == 4
