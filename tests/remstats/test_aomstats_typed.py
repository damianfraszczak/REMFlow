"""Typed actor-oriented statistic regression tests."""

import numpy as np
import pandas as pd

from remflow import aomstats, remify, tomstats


def _typed_histories():
    events = pd.DataFrame(
        {
            "time": [1, 2, 4, 7, 8, 11, 15, 16, 22, 25],
            "actor1": [1, 2, 1, 3, 4, 2, 3, 1, 4, 2],
            "actor2": [2, 1, 3, 1, 1, 4, 2, 4, 3, 3],
            "type": ["reply", "post", "reply", "mention", "post", "reply",
                     "mention", "post", "reply", "mention"],
        }
    )
    actor = remify(events, actors=[1, 2, 3, 4], model="actor")
    tie = remify(events, actors=[1, 2, 3, 4], model="tie")
    return actor, tie


def _tie_position(riskset, sender_id, receiver_id):
    matches = np.flatnonzero(
        (riskset["sender_id"].to_numpy() == sender_id)
        & (riskset["receiver_id"].to_numpy() == receiver_id)
    )
    assert len(matches) == 1
    return int(matches[0])


def _compare_receivers(actor_stats, tie_stats):
    history = actor_stats.history
    for output_index, event_index in enumerate(actor_stats.event_indices):
        event = history.events.iloc[event_index]
        for receiver_id in history.receiver_riskset[event["sender"]]:
            tie_index = _tie_position(
                tie_stats.history.risksets[event_index],
                int(event["sender_id"]),
                int(receiver_id),
            )
            np.testing.assert_allclose(
                actor_stats.receiver_stats[output_index][int(receiver_id) - 1],
                tie_stats.stats[output_index][tie_index, 1:],
                rtol=1e-12,
                atol=1e-12,
            )


def _compare_senders(actor_stats, tie_stats):
    history = actor_stats.history
    for output_index, event_index in enumerate(actor_stats.event_indices):
        riskset = tie_stats.history.risksets[event_index]
        for sender_position, sender_id in enumerate(history.sender_riskset):
            receiver_id = int(history.receiver_riskset[int(sender_id)][0])
            tie_index = _tie_position(riskset, int(sender_id), receiver_id)
            np.testing.assert_allclose(
                actor_stats.sender_stats[output_index][sender_position],
                tie_stats.stats[output_index][tie_index],
                rtol=1e-12,
                atol=1e-12,
            )


def test_typed_actor_dimensions_names_and_separate_sum_invariants():
    actor_history, _ = _typed_histories()
    receiver_effects = "inertia indegreeReceiver outdegreeReceiver reciprocity".split()
    sender_effects = ["outdegreeSender", "indegreeSender"]
    receiver_ignore = aomstats(
        reh=actor_history,
        receiver_effects="~ " + " + ".join(f"{name}()" for name in receiver_effects),
    )
    receiver_separate = aomstats(
        reh=actor_history,
        receiver_effects="~ "
        + " + ".join(
            f'{name}(consider_type="separate")' for name in receiver_effects
        ),
    )
    sender_ignore = aomstats(
        reh=actor_history,
        sender_effects="~ " + " + ".join(f"{name}()" for name in sender_effects),
    )
    sender_separate = aomstats(
        reh=actor_history,
        sender_effects="~ "
        + " + ".join(f'{name}(consider_type="separate")' for name in sender_effects),
    )

    event_count = len(actor_history.events) - 1
    assert receiver_ignore.receiver_names == receiver_effects
    assert len(receiver_ignore.receiver_stats) == event_count
    assert receiver_ignore.receiver_stats[0].shape == (4, 4)
    assert receiver_separate.receiver_stats[0].shape == (4, 12)
    assert sender_ignore.sender_names == ["baseline", *sender_effects]
    assert sender_ignore.sender_stats[0].shape == (4, 3)
    assert sender_separate.sender_stats[0].shape == (4, 7)

    types = ["mention", "post", "reply"]
    for event in range(event_count):
        for stat_index, stat in enumerate(receiver_effects):
            columns = [receiver_separate.receiver_names.index(f"{stat}.{kind}") for kind in types]
            np.testing.assert_allclose(
                receiver_separate.receiver_stats[event][:, columns].sum(axis=1),
                receiver_ignore.receiver_stats[event][:, stat_index],
            )
        for stat_index, stat in enumerate(sender_effects, start=1):
            columns = [sender_separate.sender_names.index(f"{stat}.{kind}") for kind in types]
            np.testing.assert_allclose(
                sender_separate.sender_stats[event][:, columns].sum(axis=1),
                sender_ignore.sender_stats[event][:, stat_index],
            )

    mention_column = receiver_separate.receiver_names.index("inertia.mention")
    # The first mention is event four, so output rows for events two and three
    # have no prior mention history.
    np.testing.assert_array_equal(
        np.stack(receiver_separate.receiver_stats[:2])[:, :, mention_column],
        0.0,
    )
    # A reply does not alter the mention-specific state between events four
    # and six (LOCF behavior for type-separated statistics).
    np.testing.assert_array_equal(
        receiver_separate.receiver_stats[3][:, mention_column],
        receiver_separate.receiver_stats[4][:, mention_column],
    )


def test_typed_actor_receiver_matches_tie_for_full_and_decay_memory():
    actor_history, tie_history = _typed_histories()
    names = ["inertia", "indegreeReceiver", "outdegreeReceiver", "reciprocity"]
    effects = "~ " + " + ".join(
        f'{name}(consider_type="separate")' for name in names
    )
    for memory, memory_value in [("full", None), ("decay", 5)]:
        actor_stats = aomstats(
            reh=actor_history,
            receiver_effects=effects,
            memory=memory,
            memory_value=memory_value,
        )
        tie_stats = tomstats(
            effects,
            reh=tie_history,
            memory=memory,
            memory_value=memory_value,
        )
        _compare_receivers(actor_stats, tie_stats)


def test_typed_actor_sender_matches_tie_for_full_and_decay_memory():
    actor_history, tie_history = _typed_histories()
    names = ["outdegreeSender", "indegreeSender"]
    effects = "~ " + " + ".join(
        f'{name}(consider_type="separate")' for name in names
    )
    for memory, memory_value in [("full", None), ("decay", 5)]:
        actor_stats = aomstats(
            reh=actor_history,
            sender_effects=effects,
            memory=memory,
            memory_value=memory_value,
        )
        tie_stats = tomstats(
            effects,
            reh=tie_history,
            memory=memory,
            memory_value=memory_value,
        )
        _compare_senders(actor_stats, tie_stats)
