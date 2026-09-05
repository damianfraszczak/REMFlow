"""Typed event-history behavior."""

import pandas as pd
import pytest

from remflow import remify


def _typed_events(*, simultaneous: bool = False) -> pd.DataFrame:
    times = [1, 2, 3, 4, 5, 6]
    if simultaneous:
        times[4] = times[5]
    return pd.DataFrame(
        {
            "time": times,
            "actor1": ["A", "B", "A", "C", "B", "C"],
            "actor2": ["B", "A", "C", "A", "C", "B"],
            "type": ["social", "social", "work", "work", "social", "work"],
        }
    )


def _pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["sender"], frame["receiver"], strict=True))


def _typed_pairs(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(zip(frame["sender"], frame["receiver"], frame["event_type"], strict=True))


def test_full_typed_riskset_expansion_is_controlled_by_flag():
    events = _typed_events()
    expanded = remify(events, event_type="type", extend_riskset_by_type=True)
    unexpanded = remify(events, event_type="type", extend_riskset_by_type=False)

    assert expanded.C == unexpanded.C == 2
    assert expanded.D == 12
    assert unexpanded.D == 6
    assert set(expanded.risksets[0]["event_type"]) == {"social", "work"}
    assert "event_type" not in unexpanded.risksets[0]
    assert expanded.events["type_id"].notna().all()
    assert set(expanded.events["dyad_id"]).issubset(set(expanded.risksets[0]["dyad_id"]))


def test_active_risksets_equal_observed_typed_or_untyped_combinations():
    events = _typed_events()
    expanded = remify(events, riskset="active", event_type="type", extend_riskset_by_type=True)
    unexpanded = remify(events, riskset="active", event_type="type", extend_riskset_by_type=False)

    observed_typed = set(zip(events["actor1"], events["actor2"], events["type"], strict=True))
    observed_pairs = set(zip(events["actor1"], events["actor2"], strict=True))
    assert _typed_pairs(expanded.risksets[0]) == observed_typed
    assert _pairs(unexpanded.risksets[0]) == observed_pairs
    assert all(riskset.equals(expanded.risksets[0]) for riskset in expanded.risksets)
    assert all(riskset.equals(unexpanded.risksets[0]) for riskset in unexpanded.risksets)


def test_manual_risksets_expand_types_and_add_all_observed_combinations():
    events = _typed_events()
    manual = events.loc[:0, ["actor1", "actor2"]]

    with pytest.warns(UserWarning, match="observed dyads"):
        expanded = remify(
            events,
            riskset="manual",
            manual_riskset=manual,
            event_type="type",
            extend_riskset_by_type=True,
        )
    with pytest.warns(UserWarning, match="observed dyads"):
        unexpanded = remify(
            events,
            riskset="manual",
            manual_riskset=manual,
            event_type="type",
            extend_riskset_by_type=False,
        )

    observed_typed = set(zip(events["actor1"], events["actor2"], events["type"], strict=True))
    observed_pairs = set(zip(events["actor1"], events["actor2"], strict=True))
    assert observed_typed.issubset(_typed_pairs(expanded.risksets[0]))
    assert observed_pairs.issubset(_pairs(unexpanded.risksets[0]))
    assert {("A", "B", "social"), ("A", "B", "work")}.issubset(_typed_pairs(expanded.risksets[0]))
    assert "event_type" not in unexpanded.risksets[0]


def test_type_expansion_is_ignored_for_untyped_events():
    events = _typed_events().drop(columns="type")
    history = remify(events, extend_riskset_by_type=True)

    assert history.event_types == []
    assert history.C == 1
    assert history.D == 6
    assert "event_type" not in history.risksets[0]
    assert history.events["type_id"].isna().all()


@pytest.mark.parametrize("extend", [True, False])
def test_simultaneous_typed_events_keep_vector_ids_and_decode_contract(extend):
    history = remify(
        _typed_events(simultaneous=True),
        riskset="active",
        event_type="type",
        extend_riskset_by_type=extend,
        ordinal=True,
        riskset_decode="ids",
    )

    assert history.M < history.E
    assert history.events["type_id"].notna().all()
    assert history.riskset_info is not None
    included = history.riskset_info["included"]
    assert included is not None
    expected_columns = {"dyadID", "actor1ID", "actor2ID"}
    if extend:
        expected_columns.add("typeID")
    assert expected_columns.issubset(included.columns)
    assert set(history.events["dyad_id"]).issubset(set(included["dyadID"]))
