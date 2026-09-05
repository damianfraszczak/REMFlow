"""Tie-statistic slicing behavior."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


def test_first_and_last_select_the_same_point_time_tensor_slice():
    events = pd.DataFrame(
        {
            "time": range(1, 6),
            "actor1": [1, 1, 2, 2, 3],
            "actor2": [2, 3, 1, 3, 2],
        }
    )
    attributes = pd.DataFrame(
        {
            "name": [1, 2, 3, 1, 2, 3],
            "time": [0, 0, 0, 3, 3, 3],
            "x1": [10, 20, 30, 100, 200, 300],
            "x2": [0, 1, 1, 1, 1, 0],
        }
    )
    effects = """~
        outdegreeSender() + outdegreeReceiver() +
        indegreeSender() + indegreeReceiver() +
        totaldegreeSender() + totaldegreeReceiver() + totaldegreeDyad() +
        inertia() + reciprocity() +
        isp() + itp() + osp() + otp() +
        isp(unique=TRUE) + itp(unique=TRUE) +
        osp(unique=TRUE) + otp(unique=TRUE) +
        psABBA() + psABBY() + psABAB() + psABBY() +
        psABXA() + psABXB() + psABXY() +
        recencyContinue() + recencySendSender() + recencySendReceiver() +
        recencyReceiveSender() + recencyReceiveReceiver() +
        rrankSend() + rrankReceive() +
        send(variable="x1") + receive(variable="x1") +
        average(variable="x1") + difference(variable="x1") +
        maximum(variable="x1") + minimum(variable="x1") + same(variable="x2")
    """
    history = remify(events, riskset="active")
    with pytest.warns(DeprecationWarning):
        complete = remstats(
            history,
            tie_effects=effects,
            attr_actors=attributes,
            first=1,
        )
    with pytest.warns(DeprecationWarning):
        sliced = remstats(
            history,
            tie_effects=effects,
            attr_actors=attributes,
            first=2,
            last=4,
        )

    assert sliced.names == complete.names
    assert sliced.event_indices == [1, 2, 3]
    for actual, expected in zip(sliced.stats, complete.stats[1:4], strict=True):
        np.testing.assert_array_equal(actual, expected)
