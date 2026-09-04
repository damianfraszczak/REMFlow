"""Event-history summary behavior."""

from contextlib import nullcontext

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


def test_untyped_and_typed_full_summary_contracts():
    untyped = remify(_events(typed=False))
    expanded = remify(_events(), event_type="type", extend_riskset_by_type=True)
    unexpanded = remify(_events(), event_type="type", extend_riskset_by_type=False)

    assert untyped.summary()["event_types"] == 1
    assert untyped.summary()["included_dyads"] == 6
    assert "extend_riskset_by_type" not in untyped.summary()

    assert expanded.summary()["event_types"] == 2
    assert expanded.summary()["included_dyads"] == 12
    assert expanded.summary()["extend_riskset_by_type"] is True
    assert unexpanded.summary()["included_dyads"] == 6
    assert unexpanded.summary()["extend_riskset_by_type"] is False


@pytest.mark.parametrize("mode", ["active", "manual"])
@pytest.mark.parametrize("extend", [True, False])
def test_active_and_manual_summary_reports_riskset_and_type_counts(mode, extend):
    kwargs = {}
    if mode == "manual":
        kwargs["manual_riskset"] = _events().loc[:1, ["actor1", "actor2"]]
    with pytest.warns(UserWarning) if mode == "manual" else nullcontext():
        history = remify(
            _events(),
            riskset=mode,
            event_type="type",
            extend_riskset_by_type=extend,
            **kwargs,
        )

    summary = history.summary()
    assert summary["riskset"] == mode
    assert summary["extend_riskset_by_type"] is extend
    if extend:
        assert sum(summary["dyads_per_type"].values()) == summary["included_dyads"]
        assert set(summary["dyads_per_type"]) == {"social", "work"}
    else:
        assert "dyads_per_type" not in summary


def test_undirected_and_simultaneous_dimensions_are_reported():
    undirected = remify(_events(), directed=False, event_type="type", extend_riskset_by_type=True)
    simultaneous = remify(_events(simultaneous=True))

    assert undirected.summary()["directed"] is False
    assert undirected.summary()["included_dyads"] == 6
    assert simultaneous.summary()["events"] == 6
    assert simultaneous.summary()["time_points"] == 5
    assert "time points = 5" in str(simultaneous)
