import pandas as pd

from remflow import EventHistory, is_remify_durem, remify


def test_history_full_riskset_uses_public_one_based_ids():
    history = remify(
        pd.DataFrame(
            {
                "time": [1, 2, 3],
                "sender": ["A", "B", "A"],
                "receiver": ["B", "A", "C"],
                "kind": ["x", "x", "y"],
            }
        ),
        event_type="kind",
        actors=["A", "B", "C"],
        riskset="full",
    )

    assert isinstance(history, EventHistory)
    assert history.dim == (3, 3, 2, 6)
    assert history.actors["actor_id"].to_list() == [1, 2, 3]
    assert history.events["sender_id"].to_list() == [1, 2, 1]
    assert len(history.risksets[0]) == 6
    assert (
        (history.risksets[2]["sender_id"] == 1) & (history.risksets[2]["receiver_id"] == 3)
    ).any()
    assert not is_remify_durem(history)


def test_typed_riskset_expansion():
    history = remify(
        pd.DataFrame(
            {
                "sender": ["A", "B"],
                "receiver": ["B", "A"],
                "type": ["email", "call"],
            }
        ),
        event_type="type",
        extend_riskset_by_type=True,
    )

    assert set(history.risksets[1]["event_type"]) == {"email", "call"}
    assert len(history.risksets[1]) == 4


def test_duration_marker():
    history = remify(
        [{"time": 1, "actor1": "A", "actor2": "B", "end": 2}],
        duration=True,
        model="tie",
    )

    assert is_remify_durem(history)
