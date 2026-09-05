"""Tie-statistic event weights."""

import numpy as np
import pandas as pd

from remflow import remify, remstats


def test_event_weights_accumulate_in_inertia_exactly():
    events = pd.DataFrame(
        {
            "time": range(1, 6),
            "actor1": [1, 1, 2, 2, 3],
            "actor2": [2, 3, 1, 3, 2],
            "weight": [0.15, 0.25, 0.35, 0.45, 0.55],
        }
    )
    history = remify(events)
    result = remstats(history, tie_effects="~ inertia()", first=1)
    inertia_index = result.names.index("inertia")
    tensor = np.stack([matrix[:, inertia_index] for matrix in result.stats])

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0.15, 0, 0, 0, 0, 0],
            [0.15, 0.25, 0, 0, 0, 0],
            [0.15, 0.25, 0.35, 0, 0, 0],
            [0.15, 0.25, 0.35, 0.45, 0, 0],
        ]
    )
    np.testing.assert_allclose(tensor, expected)
