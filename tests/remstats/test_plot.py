"""Statistic plot-data regression tests."""

import numpy as np
import pandas as pd

from remflow import remify, remstats


def test_tomstats_plot_and_boxplot_variants_return_filtered_plot_data():
    rng = np.random.default_rng(410)
    senders = rng.integers(1, 13, size=30)
    receivers = rng.integers(1, 12, size=30)
    receivers += receivers >= senders
    history = remify(
        pd.DataFrame(
            {
                "time": range(1, 31),
                "actor1": senders,
                "actor2": receivers,
            }
        ),
        actors=list(range(1, 13)),
        model="tie",
    )
    statistics = remstats(history, tie_effects="~ inertia()")

    by_time_name = statistics.boxplot(effect="inertia")
    by_time_index = statistics.boxplot(effect=1)
    by_dyad = statistics.boxplot(effect="inertia", by="dyads")
    dyad_subset = statistics.boxplot(
        effect="inertia",
        by="dyads",
        subset=[2, 3, 4, 5],
    )
    trajectories_name = statistics.plot(effect="inertia")
    trajectories_index = statistics.plot(effect=1)
    one_trajectory = statistics.plot(effect="inertia", subset=60)

    assert by_time_name["effect"] == by_time_index["effect"] == "inertia"
    assert by_time_name["data"].equals(by_time_index["data"])
    assert by_time_name["data"]["event_id"].nunique() == 20
    assert by_dyad["data"]["risk_id"].nunique() == 20
    assert dyad_subset["data"]["risk_id"].drop_duplicates().to_list() == [2, 3, 4, 5]
    assert trajectories_name["data"].equals(trajectories_index["data"])
    assert trajectories_name["data"]["risk_id"].nunique() == 5
    assert one_trajectory["data"]["risk_id"].unique().tolist() == [60]
