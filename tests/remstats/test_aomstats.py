"""Actor-oriented statistic comparisons against tie-oriented kernels."""

import numpy as np
import pandas as pd

from remflow import AomStats, aomstats, remify, remstats


def _events():
    return pd.DataFrame(
        {
            "time": range(1, 7),
            "actor1": [1, 2, 1, 3, 2, 1],
            "actor2": [2, 1, 3, 1, 3, 2],
            "type": ["reply", "post", "reply", "post", "post", "reply"],
        }
    )


def _tie_row(riskset, sender_id, receiver_id):
    matches = riskset.index[
        (riskset["sender_id"] == sender_id) & (riskset["receiver_id"] == receiver_id)
    ]
    assert len(matches) == 1
    return int(matches[0])


def test_actor_sender_and_receiver_statistics_equal_tie_kernels():
    events = _events()
    actor_history = remify(events, actors=[1, 2, 3], model="actor", riskset="full")
    tie_history = remify(events, actors=[1, 2, 3], model="tie", riskset="full")
    actor = aomstats(
        reh=actor_history,
        sender_effects="~ outdegreeSender() + indegreeSender()",
        receiver_effects="~ inertia() + reciprocity() + indegreeReceiver()",
        first=1,
    )
    tie_sender = remstats(
        tie_history,
        tie_effects="~ outdegreeSender() + indegreeSender()",
        first=1,
    )
    tie_receiver = remstats(
        tie_history,
        tie_effects="~ inertia() + reciprocity() + indegreeReceiver()",
        first=1,
    )

    assert isinstance(actor, AomStats)
    assert actor.sender_names == ["baseline", "outdegreeSender", "indegreeSender"]
    assert actor.receiver_names == ["inertia", "reciprocity", "indegreeReceiver"]
    for event_index, event in enumerate(actor_history.events.itertuples()):
        tie_riskset = tie_history.risksets[event_index]
        for sender_position, sender_id in enumerate(actor_history.sender_riskset):
            receiver_id = int(actor_history.receiver_riskset[sender_id][0])
            tie_position = _tie_row(tie_riskset, int(sender_id), receiver_id)
            np.testing.assert_array_equal(
                actor.sender_stats[event_index][sender_position],
                tie_sender.stats[event_index][tie_position],
            )
        for receiver_id in actor_history.receiver_riskset[event.sender]:
            tie_position = _tie_row(tie_riskset, int(event.sender_id), int(receiver_id))
            np.testing.assert_array_equal(
                actor.receiver_stats[event_index][int(receiver_id) - 1],
                tie_receiver.stats[event_index][tie_position, 1:],
            )
        assert actor.observed_sender_indices[event_index] == int(event.sender_id) - 1
        assert actor.observed_receiver_indices[event_index] == int(event.receiver_id) - 1


def test_actor_receiver_typed_slices_match_tie_slices():
    events = _events()
    actor_history = remify(events, actors=[1, 2, 3], model="actor")
    tie_history = remify(events, actors=[1, 2, 3], model="tie")
    effects = '~ inertia(consider_type="separate")'
    actor = aomstats(reh=actor_history, receiver_effects=effects, first=1)
    tie = remstats(tie_history, tie_effects=effects, first=1)

    assert actor.receiver_names == ["inertia.post", "inertia.reply"]
    for event_index, event in enumerate(actor_history.events.itertuples()):
        riskset = tie_history.risksets[event_index]
        for receiver_id in actor_history.receiver_riskset[event.sender]:
            tie_position = _tie_row(riskset, int(event.sender_id), int(receiver_id))
            np.testing.assert_array_equal(
                actor.receiver_stats[event_index][int(receiver_id) - 1],
                tie.stats[event_index][tie_position, 1:],
            )


def test_actor_decay_memory_matches_tie_kernels():
    events = _events().assign(time=[1, 2, 4, 7, 11, 16])
    actor_history = remify(events, actors=[1, 2, 3], model="actor")
    tie_history = remify(events, actors=[1, 2, 3], model="tie")
    effects = "~ inertia() + indegreeReceiver() + outdegreeReceiver() + reciprocity()"
    actor = aomstats(
        reh=actor_history,
        receiver_effects=effects,
        memory="decay",
        memory_value=5,
        first=1,
    )
    tie = remstats(
        tie_history,
        tie_effects=effects,
        memory="decay",
        memory_value=5,
        first=1,
    )

    for event_index, event in enumerate(actor_history.events.itertuples()):
        riskset = tie_history.risksets[event_index]
        for receiver_id in actor_history.receiver_riskset[event.sender]:
            tie_position = _tie_row(riskset, int(event.sender_id), int(receiver_id))
            np.testing.assert_allclose(
                actor.receiver_stats[event_index][int(receiver_id) - 1],
                tie.stats[event_index][tie_position, 1:],
                rtol=1e-12,
                atol=1e-12,
            )
