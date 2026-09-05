"""Typed-event risk-set and effect orthogonality."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


def _events():
    return pd.DataFrame(
        {
            "time": range(1, 11),
            "actor1": [1, 2, 1, 3, 1, 4, 2, 3, 1, 4],
            "actor2": [2, 1, 2, 1, 3, 1, 4, 2, 2, 3],
            "type": ["work", "social", "social", "work", "social", "work",
                     "social", "work", "work", "social"],
            "weight": [2.0, 1.0, 3.0, 0.5, 4.0, 2.5, 1.5, 3.5, 5.0, 2.0],
        }
    )


def _history(*, riskset, directed, expanded):
    events = _events()
    kwargs = {}
    if riskset == "manual":
        kwargs["manual_riskset"] = events[["actor1", "actor2"]]
    return remify(
        events,
        actors=[1, 2, 3, 4],
        model="tie",
        riskset=riskset,
        directed=directed,
        extend_riskset_by_type=expanded,
        **kwargs,
    )


def _pair_position(riskset, row):
    positions = np.flatnonzero(
        (riskset["sender_id"].to_numpy() == int(row.sender_id))
        & (riskset["receiver_id"].to_numpy() == int(row.receiver_id))
    )
    assert len(positions) == 1
    return int(positions[0])


@pytest.mark.parametrize(
    ("riskset", "directed"),
    [("full", True), ("active", True), ("manual", True), ("full", False)],
)
def test_type_expansion_is_orthogonal_to_separate_and_ignore_history(riskset, directed):
    expanded_history = _history(riskset=riskset, directed=directed, expanded=True)
    plain_history = _history(riskset=riskset, directed=directed, expanded=False)
    expanded_separate = remstats(
        expanded_history,
        tie_effects='~ inertia(consider_type="separate")',
        first=1,
    )
    expanded_interact = remstats(
        expanded_history,
        tie_effects='~ inertia(consider_type="interact")',
        first=1,
    )
    expanded_ignore = remstats(
        expanded_history, tie_effects="~ inertia(consider_type=False)", first=1
    )
    plain_separate = remstats(
        plain_history,
        tie_effects='~ inertia(consider_type="separate")',
        first=1,
    )
    plain_ignore = remstats(
        plain_history, tie_effects="~ inertia(consider_type=False)", first=1
    )

    assert expanded_separate.names == plain_separate.names == [
        "baseline",
        "inertia.social",
        "inertia.work",
    ]
    assert expanded_ignore.names == plain_ignore.names == ["baseline", "inertia"]
    assert "event_type" in expanded_history.risksets[0]
    assert "event_type" not in plain_history.risksets[0]

    for output_index, event_index in enumerate(expanded_separate.event_indices):
        expanded_riskset = expanded_history.risksets[event_index]
        plain_riskset = plain_history.risksets[event_index]
        for expanded_position, row in enumerate(expanded_riskset.itertuples()):
            plain_position = _pair_position(plain_riskset, row)
            np.testing.assert_allclose(
                expanded_separate.stats[output_index][expanded_position],
                plain_separate.stats[output_index][plain_position],
            )
            np.testing.assert_allclose(
                expanded_ignore.stats[output_index][expanded_position],
                plain_ignore.stats[output_index][plain_position],
            )
            for history_type in ("social", "work"):
                separate_column = expanded_separate.names.index(
                    f"inertia.{history_type}"
                )
                for candidate_type in ("social", "work"):
                    interact_column = expanded_interact.names.index(
                        f"inertia.{history_type}.{candidate_type}"
                    )
                    expected = (
                        expanded_separate.stats[output_index][
                            expanded_position, separate_column
                        ]
                        if row.event_type == candidate_type
                        else 0.0
                    )
                    assert (
                        expanded_interact.stats[output_index][
                            expanded_position, interact_column
                        ]
                        == expected
                    )


def test_weighted_type_slices_have_hand_computed_inertia_values():
    history = _history(riskset="full", directed=True, expanded=False)
    statistics = remstats(
        history, tie_effects='~ inertia(consider_type="separate")', first=1
    )
    riskset = history.risksets[8]
    row = _pair_position(
        riskset,
        type("Pair", (), {"sender_id": 1, "receiver_id": 2})(),
    )

    # Before event nine, 1 -> 2 occurred with weight 2 (work) and 3 (social).
    np.testing.assert_allclose(
        statistics.stats[8][row],
        [1.0, 3.0, 2.0],
    )
    assert statistics.stats[8][row, 1:].sum() == statistics.stats[8][row, 1] + 2.0


def test_mixed_type_modes_preserve_term_order_on_unexpanded_riskset():
    history = _history(riskset="full", directed=True, expanded=False)
    mixed = remstats(
        history,
        tie_effects=(
            '~ inertia(consider_type="interact") + '
            "outdegreeSender(consider_type=False)"
        ),
        first=1,
    )
    separate = remstats(
        history, tie_effects='~ inertia(consider_type="separate")', first=1
    )

    assert mixed.names == [
        "baseline",
        "inertia.social",
        "inertia.work",
        "outdegreeSender",
    ]
    for mixed_values, separate_values in zip(mixed.stats, separate.stats, strict=True):
        np.testing.assert_array_equal(mixed_values[:, :3], separate_values)
