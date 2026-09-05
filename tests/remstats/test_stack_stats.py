"""Stacked-statistic representation regression tests."""

import numpy as np
import pandas as pd

from remflow import aomstats, remify, remstats, stack_stats


def _history(*, ordinal=False):
    return remify(
        pd.DataFrame(
            {
                "time": [1.0, 2.0, 4.0, 7.0],
                "actor1": [1, 1, 2, 3],
                "actor2": [2, 3, 1, 1],
            }
        ),
        actors=[1, 2, 3],
        ordinal=ordinal,
    )


def test_stack_stats_interval_structure_response_and_offset():
    history = _history()
    statistics = remstats(history, tie_effects="~ inertia()", first=2, last=4)
    result = stack_stats(statistics, history)
    frame = result["remstats_stack"]

    assert result.D == 6
    assert result.E == 3
    assert result.subset == (2, 4)
    assert result.ordinal is False
    assert len(frame) == result.D * result.E
    assert frame["obs"].sum() == result.E
    np.testing.assert_array_equal(frame["dyad"], np.tile(np.arange(1, 7), 3))
    np.testing.assert_array_equal(frame["time_index"], np.repeat([2, 3, 4], 6))
    np.testing.assert_allclose(
        frame.groupby("time_index")["log_interevent"].first(), np.log([1, 2, 3])
    )
    np.testing.assert_array_equal(frame["baseline"], np.ones(len(frame)))


def test_stack_stats_ordinal_has_no_interevent_offset():
    history = _history(ordinal=True)
    result = stack_stats(remstats(history, tie_effects="~ inertia()", first=1), history)

    assert result.ordinal is True
    assert "log_interevent" not in result.remstats_stack


def test_stack_stats_counts_simultaneous_observations_in_one_block():
    history = remify(
        pd.DataFrame(
            {
                "time": [1, 2, 2, 3],
                "actor1": [1, 1, 2, 3],
                "actor2": [2, 3, 1, 1],
            }
        ),
        actors=[1, 2, 3],
    )
    result = stack_stats(remstats(history, tie_effects="~ inertia()", first=1))
    middle = result.remstats_stack[result.remstats_stack["time_index"] == 2]

    assert result.E == 3
    assert middle["obs"].sum() == 2


def _typed_history(*, riskset="active", ordinal=False, expanded=False):
    return remify(
        pd.DataFrame(
            {
                "time": [1, 2, 4, 7, 11, 16, 22, 29, 37],
                "actor1": [1, 1, 2, 3, 4, 2, 3, 1, 4],
                "actor2": [2, 3, 1, 1, 1, 4, 2, 4, 3],
                "type": ["social", "work", "social", "work", "work", "social",
                         "work", "social", "work"],
            }
        ),
        actors=[1, 2, 3, 4],
        riskset=riskset,
        ordinal=ordinal,
        extend_riskset_by_type=expanded,
    )


def test_stack_stats_typed_active_and_full_subsets_preserve_contract():
    effects = (
        '~ inertia(consider_type="interact") + '
        "indegreeSender(consider_type=False) + outdegreeSender(consider_type=False)"
    )
    for riskset, expanded in [("active", False), ("full", True)]:
        history = _typed_history(riskset=riskset, expanded=expanded)
        statistics = remstats(history, tie_effects=effects, first=2, last=7)
        stacked = stack_stats(statistics, history)
        frame = stacked.remstats_stack

        assert stacked.subset == (2, 7)
        assert stacked.E == 6
        assert len(frame) == stacked.D * stacked.E
        assert frame["obs"].sum() == stacked.E
        assert {"time_index", "obs", "dyad", "log_interevent", *statistics.names}.issubset(
            frame.columns
        )
        assert frame["baseline"].eq(1.0).all()
        assert not stacked.ordinal


def test_stack_stats_sampled_interval_and_ordinal_weights():
    for ordinal in (False, True):
        history = _typed_history(ordinal=ordinal, expanded=True)
        statistics = remstats(
            history,
            tie_effects='~ inertia(consider_type="separate")',
            first=2,
            last=7,
            sampling=True,
            samp_num=5,
            seed=1,
        )
        stacked = stack_stats(statistics, history)
        frame = stacked.remstats_stack

        assert stacked.S == 5
        assert stacked.E == 6
        assert len(frame) == stacked.S * stacked.E
        assert frame["obs"].sum() == stacked.E
        assert frame.loc[frame["obs"] == 1, "weight"].eq(1.0).all()
        assert frame.loc[frame["obs"] == 0, "weight"].gt(0.0).all()
        assert ("log_interevent" in frame) is (not ordinal)
        assert stacked.ordinal is ordinal


def test_stack_actor_stats_separates_sender_rate_and_receiver_choice():
    history = remify(_typed_history().events, actors=[1, 2, 3, 4], model="actor")
    statistics = aomstats(
        reh=history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia(consider_type=False)",
        first=2,
    )
    stacked = stack_stats(statistics, history)

    assert stacked.sender_stack is not None
    assert stacked.receiver_stack is not None
    assert "log_interevent" in stacked.sender_stack
    assert "log_interevent" not in stacked.receiver_stack
    assert stacked.sender_stack.groupby("time_index")["obs"].sum().eq(1).all()
    assert stacked.receiver_stack.groupby("time_index")["obs"].sum().eq(1).all()
    assert stacked.sender_stack["baseline"].eq(1.0).all()


def test_stack_actor_stats_uses_none_for_unspecified_component():
    history = remify(_history().events, actors=[1, 2, 3], model="actor")
    receiver_only = aomstats(reh=history, receiver_effects="~ inertia()")
    sender_only = aomstats(reh=history, sender_effects="~ indegreeSender()")

    assert stack_stats(receiver_only).sender_stack is None
    assert stack_stats(receiver_only).receiver_stack is not None
    assert stack_stats(sender_only).sender_stack is not None
    assert stack_stats(sender_only).receiver_stack is None
