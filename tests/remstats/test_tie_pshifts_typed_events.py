"""Typed participation-shift point-time regressions."""

import numpy as np
import pandas as pd

from remflow import remify, tomstats

SEPARATE = (
    '~ psABBA(consider_type="separate") + '
    'psABAB(consider_type="separate") + psABAY(consider_type="separate")'
)
INTERACT = (
    '~ psABBA(consider_type="interact") + '
    'psABAB(consider_type="interact") + psABAY(consider_type="interact")'
)
IGNORE = (
    '~ psABBA(consider_type="ignore") + '
    'psABAB(consider_type="ignore") + psABAY(consider_type="ignore")'
)


def _position(riskset, sender, receiver, event_type=None):
    mask = (riskset["sender_id"] == sender) & (riskset["receiver_id"] == receiver)
    if event_type is not None:
        mask &= riskset["event_type"] == event_type
    positions = np.flatnonzero(mask.to_numpy())
    assert len(positions) == 1
    return int(positions[0])


def test_typed_pshifts_use_only_the_immediately_previous_point_in_time():
    events = pd.DataFrame(
        {
            "time": range(1, 6),
            "actor1": [1, 2, 1, 3, 2],
            "actor2": [2, 3, 3, 2, 1],
            "type": ["social", "work", "social", "social", "work"],
        }
    )
    plain = remify(events, actors=[1, 2, 3], extend_riskset_by_type=False)
    expanded = remify(events, actors=[1, 2, 3], extend_riskset_by_type=True)
    separate_plain = tomstats(SEPARATE, reh=plain, first=2)
    separate_expanded = tomstats(SEPARATE, reh=expanded, first=2)
    interaction = tomstats(INTERACT, reh=expanded, first=2)
    ignore_plain = tomstats(IGNORE, reh=plain, first=2)
    ignore_expanded = tomstats(IGNORE, reh=expanded, first=2)

    output = 3  # Event five; event four (3 -> 2, social) is the prior point.
    assert all(
        np.count_nonzero(separate_plain.stats[output][:, column]) == 0
        for column, name in enumerate(separate_plain.names)
        if name.endswith(".work")
    )
    reverse = _position(plain.risksets[4], 2, 3)
    repeat = _position(plain.risksets[4], 3, 2)
    assert separate_plain.stats[output][
        reverse, separate_plain.names.index("psABBA.social")
    ] == 1
    assert ignore_plain.stats[output][repeat, ignore_plain.names.index("psABAB")] == 1

    for name in ("psABBA", "psABAB", "psABAY"):
        social = separate_expanded.names.index(f"{name}.social")
        work = separate_expanded.names.index(f"{name}.work")
        np.testing.assert_array_equal(
            separate_expanded.stats[output][:, social],
            separate_expanded.stats[output][:, work],
        )
    for column, name in enumerate(interaction.names):
        if ".work." in name:
            np.testing.assert_array_equal(interaction.stats[output][:, column], 0.0)

    assert ignore_plain.names == ignore_expanded.names == [
        "baseline",
        "psABBA",
        "psABAB",
        "psABAY",
    ]
    for event in range(len(ignore_plain.stats)):
        expanded_riskset = expanded.risksets[ignore_expanded.event_indices[event]]
        for row, candidate in enumerate(expanded_riskset.itertuples()):
            plain_row = _position(plain.risksets[ignore_plain.event_indices[event]],
                                  int(candidate.sender_id), int(candidate.receiver_id))
            np.testing.assert_array_equal(
                ignore_expanded.stats[event][row], ignore_plain.stats[event][plain_row]
            )


def test_simultaneous_previous_events_contribute_to_their_own_type_slices():
    events = pd.DataFrame(
        {
            "time": [2, 2, 3, 4, 5],
            "actor1": [1, 2, 1, 3, 2],
            "actor2": [2, 3, 3, 2, 1],
            "type": ["social", "work", "social", "social", "work"],
        }
    )
    history = remify(events, actors=[1, 2, 3], extend_riskset_by_type=False)
    statistics = tomstats(SEPARATE, reh=history, first=2)
    first_output = 0  # time three, after both time-two events
    dyad_12 = _position(history.risksets[2], 1, 2)
    dyad_23 = _position(history.risksets[2], 2, 3)

    assert statistics.stats[first_output][
        dyad_12, statistics.names.index("psABAB.social")
    ] == 1
    assert statistics.stats[first_output][
        dyad_12, statistics.names.index("psABAB.work")
    ] == 0
    assert statistics.stats[first_output][
        dyad_23, statistics.names.index("psABAB.work")
    ] == 1
