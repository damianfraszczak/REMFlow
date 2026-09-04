"""Tie-statistic binding behavior."""

import numpy as np
import pandas as pd
import pytest

from remflow import bind_remstats, remify, remstats


def test_bind_remstats_combines_unique_terms_and_preserves_contract():
    events = pd.DataFrame(
        {
            "time": range(1, 5),
            "actor1": [1, 2, 3, 1],
            "actor2": [4, 5, 6, 5],
        }
    )
    history = remify(events, actors=range(1, 7))
    first = remstats(history, tie_effects="~ inertia() + reciprocity() + otp()")
    second = remstats(history, tie_effects="~ outdegreeSender() + reciprocity()")
    third = remstats(history, tie_effects="~ reciprocity()")

    with pytest.warns(UserWarning, match="duplicate"):
        combined = bind_remstats(first, second, third)

    assert combined.history is history
    assert combined.names == [
        "baseline",
        "inertia",
        "reciprocity",
        "otp",
        "outdegreeSender",
    ]
    assert len(combined.stats) == 3
    assert combined.stats[0].shape == (30, 5)
    for event, matrix in enumerate(combined.stats):
        np.testing.assert_array_equal(matrix[:, :4], first.stats[event])
        np.testing.assert_array_equal(matrix[:, 4], second.stats[event][:, 1])


def test_bind_remstats_rejects_different_histories():
    one = remify(pd.DataFrame({"actor1": [1], "actor2": [2]}), ordinal=True)
    two = remify(pd.DataFrame({"actor1": [1], "actor2": [2]}), ordinal=True)
    with pytest.raises(ValueError, match="same EventHistory"):
        bind_remstats(
            remstats(one, tie_effects="~ inertia()", first=1),
            remstats(two, tie_effects="~ inertia()", first=1),
        )
