"""Actor/tie statistic comparisons across all memory modes."""

import numpy as np
import pandas as pd
import pytest

from remflow import aomstats, remify, tomstats

EFFECT_GROUPS = {
    "base": ["inertia", "reciprocity"],
    "degrees": ["indegreeReceiver", "outdegreeReceiver", "totaldegreeReceiver"],
    "triads": ["otp", "itp", "osp", "isp"],
    "pshifts": ["psABBA", "psABBY", "psABXA", "psABXB", "psABXY", "psABAY", "psABAB"],
    "rrank": ["rrankSend", "rrankReceive"],
    "recency": ["recencySendReceiver", "recencyReceiveReceiver", "recencyContinue"],
}

CASES = [
    *(('full', None, name) for name in EFFECT_GROUPS),
    *(('decay', 5, name) for name in EFFECT_GROUPS),
    ("window", 10, "base"),
    ("window", 10, "degrees"),
    ("interval", (3, 15), "base"),
    ("interval", (3, 15), "degrees"),
]


def _histories():
    events = pd.DataFrame(
        {
            "time": [1, 2, 4, 7, 8, 11, 15, 16, 22, 25, 31, 34],
            "actor1": [1, 2, 1, 3, 4, 2, 3, 1, 4, 2, 3, 1],
            "actor2": [2, 1, 3, 1, 1, 4, 2, 4, 3, 3, 4, 2],
            "type": ["reply", "post", "reply", "mention", "post", "reply",
                     "mention", "post", "reply", "mention", "post", "reply"],
        }
    )
    return (
        remify(events, actors=[1, 2, 3, 4], model="actor"),
        remify(events, actors=[1, 2, 3, 4], model="tie"),
    )


def _riskset_position(riskset, sender_id, receiver_id):
    positions = np.flatnonzero(
        (riskset["sender_id"].to_numpy() == sender_id)
        & (riskset["receiver_id"].to_numpy() == receiver_id)
    )
    assert len(positions) == 1
    return int(positions[0])


@pytest.mark.parametrize(("memory", "memory_value", "group"), CASES)
def test_actor_receiver_statistics_match_tie_kernels_for_memory_modes(
    memory, memory_value, group
):
    actor_history, tie_history = _histories()
    effects = EFFECT_GROUPS[group]
    actor_formula = "~ " + " + ".join(
        f'{name}(consider_type="separate")' for name in effects
    )
    tie_formula = "~ " + " + ".join(
        f"{name}(consider_type=True)" for name in effects
    )
    actor = aomstats(
        reh=actor_history,
        receiver_effects=actor_formula,
        memory=memory,
        memory_value=memory_value,
        first=3,
    )
    tie = tomstats(
        tie_formula,
        reh=tie_history,
        memory=memory,
        memory_value=memory_value,
        first=3,
    )

    assert actor.receiver_names == tie.names[1:]
    for output_index, event_index in enumerate(actor.event_indices):
        event = actor_history.events.iloc[event_index]
        for receiver_id in actor_history.receiver_riskset[event["sender"]]:
            tie_position = _riskset_position(
                tie_history.risksets[event_index],
                int(event["sender_id"]),
                int(receiver_id),
            )
            np.testing.assert_allclose(
                actor.receiver_stats[output_index][int(receiver_id) - 1],
                tie.stats[output_index][tie_position, 1:],
                rtol=1e-12,
                atol=1e-12,
            )
