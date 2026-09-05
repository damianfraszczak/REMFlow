"""Tie-model memory modes."""

import numpy as np
import pandas as pd

from remflow import remify, remstats


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": range(1, 11),
            "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
            "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
        }
    )


def _expected_inertia(history, weight_for_age):
    riskset = history.risksets[0]
    positions = {
        (int(row.sender_id), int(row.receiver_id)): index
        for index, row in enumerate(riskset.itertuples())
    }
    expected = np.zeros((history.E, len(riskset)))
    for event_index in range(1, history.E):
        reference_time = float(history.events.iloc[event_index - 1]["time"])
        for previous_index in range(event_index):
            previous = history.events.iloc[previous_index]
            age = reference_time - float(previous["time"])
            weight = weight_for_age(age)
            if weight:
                position = positions[(int(previous["sender_id"]), int(previous["receiver_id"]))]
                expected[event_index, position] += weight
    return expected


def _stat_tensor(result):
    inertia_index = result.names.index("inertia")
    return np.stack([matrix[:, inertia_index] for matrix in result.stats])


def test_window_interval_and_decay_memory_match_documented_boundaries():
    history = remify(_events(), riskset="active")
    window = remstats(history, tie_effects="~ inertia()", memory="window", memory_value=5, first=1)
    interval = remstats(
        history,
        tie_effects="~ inertia()",
        memory="interval",
        memory_value=(2, 5),
        first=1,
    )
    decay = remstats(history, tie_effects="~ inertia()", memory="decay", memory_value=5, first=1)

    np.testing.assert_allclose(
        _stat_tensor(window), _expected_inertia(history, lambda age: float(0 <= age < 5))
    )
    np.testing.assert_allclose(
        _stat_tensor(interval), _expected_inertia(history, lambda age: float(2 <= age < 5))
    )
    np.testing.assert_allclose(
        _stat_tensor(decay),
        _expected_inertia(history, lambda age: float(np.exp(-age * np.log(2) / 5))),
        rtol=1e-12,
        atol=1e-12,
    )
