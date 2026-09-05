"""Duration-statistic stacking behavior."""

import numpy as np
import pandas as pd

from remflow import remify, remstats, stack_stats


def _events():
    return pd.DataFrame(
        {
            "time": [1, 2, 5],
            "actor1": ["A", "B", "A"],
            "actor2": ["B", "C", "C"],
            "end": [6, 7, 8],
        }
    )


def test_duration_stack_has_exact_dynamic_risksets_and_observations():
    history = remify(
        _events().assign(who_ended=["actor1"] * 3),
        duration=True,
        dur_directed_end=True,
        model="tie",
    )
    statistics = remstats(
        history,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        psi_start=1,
        psi_end=1,
    )
    stacked = stack_stats(statistics, history)
    frame = stacked.remstats_stack

    assert {
        "obs",
        "log_interevent",
        "inertia.start",
        "inertia.end",
        "time_index",
        "dyad",
        "process",
    }.issubset(frame.columns)
    assert frame["obs"].isin([0, 1]).all()
    assert stacked.E == 5
    assert not stacked.ordinal
    assert stacked.model == "durem"
    assert stacked.D_start == 6
    assert stacked.D_end == 6
    assert len(frame) == 30
    assert frame["obs"].sum() == 5
    assert frame.loc[frame["process"] == "start", "obs"].sum() == 2
    assert frame.loc[frame["process"] == "end", "obs"].sum() == 3
    assert np.isfinite(frame["log_interevent"]).all()
    assert stacked.stat_names == [
        "baseline.start",
        "inertia.start",
        "baseline.end",
        "inertia.end",
    ]
    assert stacked.stat_names_start[1] == "inertia.start"
    assert stacked.stat_names_end[1] == "inertia.end"


def test_duration_undirected_end_and_right_censoring_contracts():
    undirected_end = remify(_events(), duration=True, model="tie")
    statistics = remstats(
        undirected_end,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        psi_start=0,
        psi_end=0,
    )
    assert stack_stats(statistics).D_end == 3

    censored_events = _events()
    censored_events.loc[1, "end"] = np.nan
    censored = remify(censored_events, duration=True, model="tie")
    censored_stats = remstats(censored, start_effects="~ inertia()", end_effects="~ inertia()")
    frame = stack_stats(censored_stats).remstats_stack
    assert frame["obs"].sum() == 4
    assert frame.loc[frame["process"] == "end", "obs"].sum() == 2
