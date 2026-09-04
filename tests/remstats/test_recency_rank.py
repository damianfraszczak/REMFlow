"""Hand-computed recency and relational-rank cases."""

import numpy as np
import pandas as pd

from remflow import remify, remstats


def _tensor(result, name):
    column = result.names.index(name)
    return np.stack([matrix[:, column] for matrix in result.stats])


def test_recency_continue_and_relational_ranks_match_hand_calculation():
    events = pd.DataFrame(
        {
            "time": range(1, 11),
            "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
            "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
            "type": [1, 1, 2, 2, 1, 2, 2, 1, 1, 1],
        }
    )
    history = remify(events, riskset="active", extend_riskset_by_type=True)
    result = remstats(
        history,
        tie_effects="~ recencyContinue() + rrankSend() + rrankReceive()",
        first=1,
    )

    expected_continue = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1 / 2, 0, 0, 0, 0, 0, 1 / 2, 0, 0, 0],
            [1 / 3, 1 / 2, 0, 0, 0, 0, 1 / 3, 1 / 2, 0, 0],
            [1 / 2, 1 / 3, 0, 0, 0, 0, 1 / 2, 1 / 3, 0, 0],
            [1 / 3, 1 / 4, 1 / 2, 0, 0, 0, 1 / 3, 1 / 4, 1 / 2, 0],
            [1 / 4, 1 / 5, 1 / 3, 0, 1 / 2, 0, 1 / 4, 1 / 5, 1 / 3, 0],
            [1 / 5, 1 / 6, 1 / 4, 0, 1 / 3, 0, 1 / 5, 1 / 6, 1 / 4, 1 / 2],
            [1 / 6, 1 / 2, 1 / 5, 0, 1 / 4, 0, 1 / 6, 1 / 2, 1 / 5, 1 / 3],
            [1 / 7, 1 / 3, 1 / 2, 0, 1 / 5, 0, 1 / 7, 1 / 3, 1 / 2, 1 / 4],
            [1 / 8, 1 / 4, 1 / 3, 1 / 2, 1 / 6, 0, 1 / 8, 1 / 4, 1 / 3, 1 / 5],
        ]
    )
    expected_send_rank = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1 / 2, 1, 0, 0, 0, 1, 1 / 2, 1, 0],
            [1, 1 / 2, 1, 0, 1, 0, 1, 1 / 2, 1, 0],
            [1, 1 / 2, 1, 0, 1, 0, 1, 1 / 2, 1, 1],
            [1, 1, 1 / 2, 0, 1, 0, 1, 1, 1 / 2, 1],
            [1, 1 / 2, 1, 0, 1, 0, 1, 1 / 2, 1, 1],
            [1, 1 / 3, 1 / 2, 1, 1, 0, 1, 1 / 3, 1 / 2, 1],
        ]
    )
    expected_receive_rank = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1 / 2, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1 / 2, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        ]
    )
    np.testing.assert_allclose(_tensor(result, "recencyContinue"), expected_continue)
    np.testing.assert_allclose(_tensor(result, "rrankSend"), expected_send_rank)
    np.testing.assert_allclose(_tensor(result, "rrankReceive"), expected_receive_rank)
