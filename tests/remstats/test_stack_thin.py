"""Stack thinning and selection regression tests."""

import numpy as np
import pandas as pd

from remflow import remify, remstats, stack_stats


def _events() -> pd.DataFrame:
    rng = np.random.default_rng(318)
    senders = rng.integers(1, 7, size=50)
    receivers = rng.integers(1, 6, size=50)
    receivers += receivers >= senders
    return pd.DataFrame(
        {
            "time": np.arange(1, 51),
            "actor1": senders,
            "actor2": receivers,
        }
    )


def _modeled_event_count(history) -> int:
    first_time = history.events.iloc[0]["time"]
    return int((history.events["time"] != first_time).sum())


def test_tie_stacks_preserve_aggregated_simultaneous_observation_counts():
    histories = [
        remify(_events(), model="tie", aggregate_time=1),
        remify(_events(), model="tie", aggregate_time=10),
    ]
    for history in histories:
        statistics = remstats(history, tie_effects="~ inertia()")
        stacked = stack_stats(statistics, history)
        frame = stacked.remstats_stack

        assert isinstance(frame, pd.DataFrame)
        assert pd.api.types.is_numeric_dtype(frame["obs"])
        assert set(frame["obs"].unique()).issubset({0, 1, 2, 3, 4, 5})
        assert int(frame["obs"].sum()) == _modeled_event_count(history)
        assert len(frame) == len(statistics.stats) * history.D


def test_actor_stacks_keep_numeric_sender_and_receiver_observations_after_aggregation():
    histories = [
        remify(_events(), model="actor", aggregate_time=1),
        remify(_events(), model="actor", aggregate_time=10),
    ]
    for history in histories:
        statistics = remstats(
            history,
            sender_effects="~ indegreeSender()",
            receiver_effects="~ inertia() + reciprocity()",
        )
        stacked = stack_stats(statistics, history, add_actors=True)

        assert isinstance(stacked.sender_stack, pd.DataFrame)
        assert isinstance(stacked.receiver_stack, pd.DataFrame)
        assert pd.api.types.is_numeric_dtype(stacked.sender_stack["obs"])
        assert pd.api.types.is_numeric_dtype(stacked.receiver_stack["obs"])
        assert int(stacked.sender_stack["obs"].sum()) == _modeled_event_count(history)
        assert int(stacked.receiver_stack["obs"].sum()) == _modeled_event_count(history)
        assert "actor" in stacked.sender_stack
        assert "actor" in stacked.receiver_stack

        without_actors = stack_stats(statistics, history)
        assert without_actors.sender_stack is not None
        assert without_actors.receiver_stack is not None
        assert "actor" not in without_actors.sender_stack
        assert "actor" not in without_actors.receiver_stack
