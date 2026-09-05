"""Progress-reporting regression tests."""

import pandas as pd

from remflow import remify, remstats


def test_display_progress_reports_calculation_for_full_effect_formula(capsys):
    history = remify(
        pd.DataFrame(
            {
                "time": range(1, 11),
                "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
                "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
            }
        ),
        model="tie",
        riskset="active",
    )
    effects = (
        "~ outdegreeSender() + outdegreeReceiver() + indegreeSender() + "
        "indegreeReceiver() + totaldegreeSender() + totaldegreeReceiver() + "
        "totaldegreeDyad() + inertia() + reciprocity() + isp() + itp() + "
        "osp() + otp() + isp(unique=TRUE) + itp(unique=TRUE) + "
        "osp(unique=TRUE) + otp(unique=TRUE) + psABBA() + psABBY() + "
        "psABAB() + psABBY() + psABXA() + psABXB() + psABXY() + "
        "recencyContinue() + recencySendSender() + recencySendReceiver() + "
        "recencyReceiveSender() + recencyReceiveReceiver() + rrankSend() + "
        "rrankReceive()"
    )

    statistics = remstats(
        history,
        tie_effects=effects,
        display_progress=True,
    )
    output = capsys.readouterr().out.splitlines()

    assert output
    assert all("Calculating" in line for line in output)
    assert statistics.stats
