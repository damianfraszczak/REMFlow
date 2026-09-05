"""Statistic object method regression tests."""

import pandas as pd
import pytest

from remflow import remify, remstats


def test_tie_stats_print_and_summary_are_nonempty_and_preserve_interaction_name(capsys):
    history = remify(
        pd.DataFrame(
            {
                "time": range(1, 6),
                "actor1": [1, 1, 2, 2, 3],
                "actor2": [2, 3, 1, 3, 2],
            }
        ),
        model="tie",
    )
    attributes = pd.DataFrame(
        {
            "name": [1, 2, 3],
            "time": [0, 0, 0],
            "x1": [10, 20, 30],
            "x2": [0, 1, 1],
        }
    )
    with pytest.warns(DeprecationWarning, match="attr_actors"):
        statistics = remstats(
            history,
            tie_effects='~ send(variable="x1"):inertia()',
            attr_actors=attributes,
        )

    print(statistics)
    printed = capsys.readouterr().out
    assert printed.strip()
    assert statistics.names == [
        "baseline",
        "send_x1",
        "inertia",
        "send_x1:inertia",
    ]

    summary = statistics.summary()
    print(summary)
    summary_output = capsys.readouterr().out
    assert summary_output.strip()
    assert summary == {
        "events": 4,
        "terms": ["baseline", "send_x1", "inertia", "send_x1:inertia"],
        "riskset_sizes": [6, 6, 6, 6],
    }
