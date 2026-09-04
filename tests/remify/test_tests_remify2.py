"""Core event-history construction tests."""

import pandas as pd
import pytest

from remflow import EventHistory, remify


def _events(*, typed: bool = False, simultaneous: bool = False) -> pd.DataFrame:
    times = [10, 20, 30, 40, 50, 60]
    if simultaneous:
        times[3] = times[4]
    frame = pd.DataFrame(
        {
            "time": times,
            "actor1": ["A", "B", "A", "C", "B", "C"],
            "actor2": ["B", "A", "C", "A", "C", "B"],
            "weight_test": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        }
    )
    if typed:
        frame["type"] = ["social", "social", "work", "work", "social", "work"]
    return frame


def test_basic_structure_weights_and_dyad_ids():
    history = remify(_events(), event_weight="weight_test", riskset_decode="ids")

    assert isinstance(history, EventHistory)
    assert history.directed is True
    assert history.model == "tie"
    assert history.riskset_mode == "full"
    assert history.weighted is True
    assert history.events["event_weight"].to_list() == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert history.events["sender_id"].ne(history.events["receiver_id"]).all()
    assert history.events["dyad_id"].between(1, history.D).all()
    assert history.riskset_info is not None
    assert history.riskset_info["decode"] == "ids"

    unweighted = remify(_events().drop(columns="weight_test"))
    assert unweighted.weighted is False
    assert unweighted.summary()["weighted"] is False


def test_active_and_manual_riskset_contracts():
    events = _events()
    active = remify(events, riskset="active")
    active_ids = set(active.riskset_info["riskset_idx"])

    assert active.riskset_info["mode"] == "active"
    assert set(active.events["dyad_id"]).issubset(active_ids)

    manual = events.loc[:0, ["actor1", "actor2"]]
    with pytest.warns(UserWarning, match="observed dyads"):
        augmented = remify(events, riskset="manual", manual_riskset=manual)
    manual_ids = set(augmented.riskset_info["riskset_idx"])
    assert set(augmented.events["dyad_id"]).issubset(manual_ids)


def test_named_type_column_is_detected_and_type_ids_are_consistent():
    history = remify(_events(typed=True), riskset="active", extend_riskset_by_type=True)

    assert history.C == 2
    assert history.event_types == ["social", "work"]
    assert history.events["type_id"].notna().all()
    assert history.riskset_info["with_type"] is True
    included = history.riskset_info["included"]
    assert included is not None
    assert "type" in included
    assert set(history.events["dyad_id"]).issubset(set(included["dyadID"]))


def test_simultaneous_events_and_decode_threshold_fallback():
    simultaneous = remify(
        _events(typed=True, simultaneous=True),
        riskset="active",
        extend_riskset_by_type=True,
        ordinal=True,
        riskset_decode="ids",
    )
    assert simultaneous.M < simultaneous.E
    assert simultaneous.events["dyad_id"].notna().all()
    assert simultaneous.events["type_id"].notna().all()

    with pytest.warns(UserWarning, match="ID-only"):
        fallback = remify(
            _events(typed=True),
            riskset="active",
            extend_riskset_by_type=True,
            riskset_decode="labels",
            riskset_max_decode=2,
        )
    assert fallback.riskset_info["decode"] == "ids"
    assert set(fallback.riskset_info["included"]).issuperset(
        {"dyadID", "actor1ID", "actor2ID", "typeID"}
    )


@pytest.mark.parametrize("riskset", ["full", "active", "active_saturated"])
def test_parallel_riskset_construction_matches_serial_reference(riskset):
    events = _events(typed=True, simultaneous=True)

    serial = remify(
        events,
        riskset=riskset,
        extend_riskset_by_type=True,
        ordinal=True,
        ncores=1,
    )
    parallel = remify(
        events,
        riskset=riskset,
        extend_riskset_by_type=True,
        ordinal=True,
        ncores=3,
    )

    pd.testing.assert_frame_equal(parallel.events, serial.events)
    pd.testing.assert_frame_equal(parallel.actors, serial.actors)
    assert len(parallel.risksets) == len(serial.risksets)
    for actual, expected in zip(parallel.risksets, serial.risksets, strict=True):
        pd.testing.assert_frame_equal(actual, expected)


def test_parallel_manual_riskset_matches_serial_reference():
    events = _events(typed=True)
    manual = events.loc[:1, ["actor1", "actor2"]]

    with pytest.warns(UserWarning, match="observed dyads"):
        serial = remify(
            events,
            riskset="manual",
            manual_riskset=manual,
            extend_riskset_by_type=True,
            ncores=1,
        )
    with pytest.warns(UserWarning, match="observed dyads"):
        parallel = remify(
            events,
            riskset="manual",
            manual_riskset=manual,
            extend_riskset_by_type=True,
            ncores=2,
        )

    for actual, expected in zip(parallel.risksets, serial.risksets, strict=True):
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("ncores", [0, -1])
def test_ncores_must_be_positive(ncores):
    with pytest.raises(ValueError, match="positive integer"):
        remify(_events(), ncores=ncores)


@pytest.mark.parametrize("ncores", [True, 1.5, "2"])
def test_ncores_must_be_an_integer(ncores):
    with pytest.raises(TypeError, match="positive integer"):
        remify(_events(), ncores=ncores)
