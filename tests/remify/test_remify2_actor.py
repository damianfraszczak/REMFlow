"""Actor-oriented event-history behavior."""

import pandas as pd
import pytest

from remflow import EventHistory, remify


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [10, 20, 30, 40, 50, 60],
            "actor1": ["A", "A", "B", "C", "B", "C"],
            "actor2": ["B", "C", "A", "A", "C", "B"],
            "type": ["social", "work", "social", "work", "social", "work"],
        }
    )


def test_actor_model_rejects_undirected_and_type_expanded_risksets():
    with pytest.raises(ValueError, match="directed=True"):
        remify(_events(), model="actor", directed=False)
    with pytest.raises(ValueError, match="not supported for actor"):
        remify(_events(), model="actor", extend_riskset_by_type=True)


def test_actor_full_riskset_contains_every_sender_and_nonself_receiver():
    history = remify(_events(), model="actor", actors=["A", "B", "C", "D"], origin=0)

    assert isinstance(history, EventHistory)
    assert history.model == "actor"
    assert history.riskset_mode == "full"
    assert history.N == 4
    assert history.M == 6
    assert history.sender_riskset.tolist() == [1, 2, 3, 4]
    assert history.activeN == 4
    assert list(history.receiver_riskset) == ["A", "B", "C", "D"]
    for sender_id, sender in zip(history.sender_riskset, history.receiver_riskset, strict=True):
        assert len(history.receiver_riskset[sender]) == 3
        assert sender_id not in history.receiver_riskset[sender]
    assert history.sender_map.columns.to_list() == ["senderID", "actorName"]


def test_actor_active_riskset_uses_only_observed_sender_receiver_pairs():
    events = _events()
    history = remify(
        events,
        model="actor",
        actors=["A", "B", "C", "D"],
        riskset="active",
    )
    actor_ids = dict(zip(history.actors["actor"], history.actors["actor_id"], strict=True))

    assert history.activeN == 3
    assert set(history.receiver_riskset) == {"A", "B", "C"}
    for sender, receivers in history.receiver_riskset.items():
        expected = {
            actor_ids[receiver]
            for receiver in events.loc[events["actor1"] == sender, "actor2"].unique()
        }
        assert set(receivers) == expected
    assert len(history.sender_map) == history.activeN


def test_actor_manual_ordinal_and_typed_contracts():
    events = _events()
    manual = pd.concat(
        [events[["actor1", "actor2"]], pd.DataFrame({"actor1": ["D"], "actor2": ["A"]})],
        ignore_index=True,
    )
    history = remify(
        events,
        model="actor",
        actors=["A", "B", "C", "D"],
        riskset="manual",
        manual_riskset=manual,
        ordinal=True,
        origin=0,
    )

    assert history.ordinal is True
    assert history.events["time"].to_list() == [1, 2, 3, 4, 5, 6]
    assert history.C == 2
    assert history.activeN == 4
    assert set(history.receiver_riskset["D"]) == {1}


def test_actor_active_saturated_adds_reverse_pairs_and_prints():
    active = remify(_events(), model="actor", riskset="active")
    saturated = remify(_events(), model="actor", riskset="active_saturated")

    assert saturated.activeN >= active.activeN
    for row in _events().itertuples():
        sender_id = int(
            saturated.actors.loc[saturated.actors["actor"] == row.actor1, "actor_id"].iloc[0]
        )
        receiver_id = int(
            saturated.actors.loc[saturated.actors["actor"] == row.actor2, "actor_id"].iloc[0]
        )
        assert receiver_id in saturated.receiver_riskset[row.actor1]
        assert sender_id in saturated.receiver_riskset[row.actor2]
    assert "riskset = active_saturated" in str(saturated)
