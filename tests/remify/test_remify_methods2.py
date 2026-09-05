"""Event-history object methods."""

import pandas as pd
import pytest

from remflow import remify


def _events(*, typed: bool = True, simultaneous: bool = False) -> pd.DataFrame:
    times = [1, 2, 3, 4, 5, 6]
    if simultaneous:
        times[4] = times[5]
    data = {
        "time": times,
        "actor1": ["A", "B", "A", "C", "B", "C"],
        "actor2": ["B", "A", "C", "A", "C", "B"],
    }
    if typed:
        data["type"] = ["social", "social", "work", "work", "social", "work"]
    return pd.DataFrame(data)


def test_dim_contract_for_typed_untyped_and_simultaneous_histories():
    typed = remify(_events(), riskset="active", extend_riskset_by_type=True)
    untyped = remify(_events(typed=False))
    dated = _events(simultaneous=True)
    dated["time"] = pd.to_datetime(dated["time"], unit="D", origin="2020-01-01")
    simultaneous = remify(
        dated,
        origin=pd.Timestamp("2019-12-31"),
        riskset="active",
        extend_riskset_by_type=True,
    )
    actor = remify(dated, origin=pd.Timestamp("2019-12-31"), model="actor")

    assert typed.dim == (typed.M, typed.N, typed.C, typed.D, typed.activeD)
    assert untyped.dim == (untyped.M, untyped.N, untyped.D)
    assert simultaneous.dim == (
        simultaneous.E,
        simultaneous.M,
        simultaneous.N,
        simultaneous.C,
        simultaneous.D,
        simultaneous.activeD,
    )
    assert actor.dim == (actor.E, actor.M, actor.N, actor.C, actor.D)


def test_manual_and_full_riskset_sources_are_preserved():
    events = _events()
    manual = events[["actor1", "actor2"]]
    manual_history = remify(events, riskset="manual", manual_riskset=manual)
    full_history = remify(events, riskset="full")

    assert manual_history.riskset_mode == "manual"
    assert full_history.riskset_mode == "full"
    assert "riskset = manual" in str(manual_history)
    assert "riskset = full" in str(full_history)


def test_plot_returns_data_and_validates_actor_selection():
    history = remify(_events(), actors=["A", "B", "C", "D"])
    plot_data = history.plot()

    assert set(plot_data) == {"events", "actors", "dyads", "waiting_times"}
    assert len(plot_data["events"]) == history.E
    with pytest.raises(ValueError, match="not present"):
        history.plot(["missing"])
    with pytest.raises(ValueError, match="no events"):
        history.plot(["A", "D"])


def test_plot_warns_and_limits_more_than_fifty_actors():
    events = pd.DataFrame(
        {
            "time": range(1, 61),
            "actor1": range(1, 61),
            "actor2": range(2, 62),
        }
    )
    history = remify(events)

    with pytest.warns(UserWarning, match="50 most active"):
        plot_data = history.plot()
    assert len(plot_data["actors"]) == 50
