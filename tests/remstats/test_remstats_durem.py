"""Duration statistic dispatch, weighting, and structure."""

import numpy as np
import pandas as pd
import pytest

from remflow import RemEstimateDuration, is_remstats_durem, remify, remstats, remstimate


def _events():
    return pd.DataFrame(
        {
            "time": [1, 4, 7, 11, 15],
            "actor1": ["A", "B", "A", "C", "B"],
            "actor2": ["B", "C", "C", "A", "A"],
            "end": [3, 6, 9, 13, 18],
        }
    )


def test_duration_dispatch_suffixes_formulas_and_predicate():
    history = remify(_events(), model="tie", duration=True)
    statistics = remstats(
        history,
        start_effects="~ inertia() + reciprocity()",
        end_effects="~ inertia()",
    )

    assert is_remstats_durem(statistics)
    assert statistics.start_stats is None
    assert statistics.end_stats is None
    assert statistics.history is history
    assert statistics.stacked.remstats_stack.shape[0] > 0
    assert all(name.endswith(".start") for name in statistics.stacked.stat_names_start)
    assert all(name.endswith(".end") for name in statistics.stacked.stat_names_end)
    assert len(statistics.stacked.stat_names_start) == 3
    assert len(statistics.stacked.stat_names_end) == 2
    assert "Duration relational-event statistics" in str(statistics)
    fitted = remstimate(history, statistics)
    assert isinstance(fitted, RemEstimateDuration)
    assert fitted.names == statistics.stacked.stat_names
    assert np.isfinite(fitted.coef).all()

    plain_history = remify(_events().drop(columns="end"), model="tie")
    plain = remstats(plain_history, tie_effects="~ inertia()")
    assert not is_remstats_durem(plain)


def test_duration_psi_and_event_weights_change_statistics_multiplicatively():
    events = _events()
    weighted_events = events.assign(weight=[1, 2, 3, 4, 5])
    plain_history = remify(events, model="tie", duration=True)
    weighted_history = remify(weighted_events, model="tie", duration=True)

    weighted_p0 = remstats(
        weighted_history,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        psi_start=0,
        psi_end=0,
    )
    plain_p1 = remstats(
        plain_history,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        psi_start=1,
        psi_end=1,
    )
    weighted_p1 = remstats(
        weighted_history,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        psi_start=1,
        psi_end=1,
    )

    assert weighted_p0.psi_start == 0
    assert plain_p1.psi_end == 1
    assert not weighted_p0.stacked.remstats_stack.equals(weighted_p1.stacked.remstats_stack)
    assert not plain_p1.stacked.remstats_stack.equals(weighted_p1.stacked.remstats_stack)
    assert not weighted_p0.stacked.remstats_stack.equals(plain_p1.stacked.remstats_stack)


def test_duration_censoring_undirected_start_typing_and_sampling_warning():
    events = _events()
    directed = remify(events, model="tie", duration=True)
    undirected = remify(events, model="tie", duration=True, directed=False)
    directed_stats = remstats(directed, start_effects="~ inertia()", end_effects="~ inertia()")
    undirected_stats = remstats(undirected, start_effects="~ inertia()", end_effects="~ inertia()")
    assert undirected_stats.stacked.D_start < directed_stats.stacked.D_start

    censored_events = events.copy()
    censored_events.loc[2, "end"] = np.nan
    censored = remify(censored_events, model="tie", duration=True)
    censored_stats = remstats(censored, start_effects="~ inertia()", end_effects="~ inertia()")
    complete_ends = directed_stats.stacked.remstats_stack.query("process == 'end'")["obs"].sum()
    censored_ends = censored_stats.stacked.remstats_stack.query("process == 'end'")["obs"].sum()
    assert censored_ends < complete_ends

    typed_events = events.assign(type=["A", "B", "A", "B", "A"], weight=range(1, 6))
    typed = remify(
        typed_events,
        model="tie",
        duration=True,
        extend_riskset_by_type=True,
    )
    typed_stats = remstats(typed, start_effects="~ inertia()", end_effects="~ inertia()")
    assert typed.events["event_type"].to_list() == typed_events["type"].to_list()
    assert np.isfinite(
        typed_stats.stacked.remstats_stack[["inertia.start", "inertia.end"]].to_numpy()
    ).all()
    with pytest.warns(UserWarning, match="sampling"):
        sampled = remstats(
            directed,
            start_effects="~ inertia()",
            end_effects="~ inertia()",
            sampling=True,
        )
    assert is_remstats_durem(sampled)


def test_duration_history_uses_only_completed_events_and_inclusive_duration_weight():
    events = pd.DataFrame(
        {
            "time": [1, 2, 4],
            "actor1": ["A", "A", "A"],
            "actor2": ["B", "C", "B"],
            "end": [3, 6, 5],
            "weight": [2, 7, 3],
        }
    )
    history = remify(events, duration=True, model="tie")
    weighted = remstats(
        history,
        start_effects="~ inertia() + activeTie()",
        end_effects="~ inertia() + activeTie()",
        psi_start=1,
        psi_end=1,
        first=1,
    ).stacked.remstats_stack
    unscaled = remstats(
        history,
        start_effects="~ inertia()",
        psi_start=0,
        first=1,
    ).stacked.remstats_stack

    # At t=4 the first A->B event is complete. Its history weight is
    # event_weight * (end - start + 1)^psi = 2 * 3 = 6.
    observed_restart = weighted[
        (weighted["time_index"] == 4) & (weighted["process"] == "start") & (weighted["obs"] == 1)
    ]
    assert observed_restart["inertia.start"].item() == 6
    unscaled_restart = unscaled[
        (unscaled["time_index"] == 4) & (unscaled["process"] == "start") & (unscaled["obs"] == 1)
    ]
    assert unscaled_restart["inertia.start"].item() == 2

    # At its ending time the tie is active but not yet completed history.
    observed_end = weighted[
        (weighted["time_index"] == 3) & (weighted["process"] == "end") & (weighted["obs"] == 1)
    ]
    assert observed_end["activeTie.end"].item() == 1
    assert observed_end["inertia.end"].item() == 0
