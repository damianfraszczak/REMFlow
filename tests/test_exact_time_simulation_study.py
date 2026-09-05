import numpy as np

from examples.exact_time_simulation_study import (
    result_table,
    run_study,
    simulate_study_process,
)


def test_exact_time_simulation_is_seeded_and_well_formed():
    first = simulate_study_process(actors=8, events=30, seed=1331)
    second = simulate_study_process(actors=8, events=30, seed=1331)

    assert first.equals(second)
    assert len(first) == 30
    assert np.all(np.diff(first["time"]) > 0)
    assert np.all(first["sender"] != first["receiver"])


def test_exact_time_simulation_can_be_reestimated():
    study = run_study(actors=10, events=80, seed=1331)
    table = result_table(study)

    assert study.fit.converged
    assert table["parameter"].to_list() == [
        "baseline",
        "psABBA",
        "recencySendSender",
    ]
    assert np.all(np.isfinite(table["estimate"]))
    assert np.isfinite(study.fit.BIC)
    assert study.diagnostics.observed_probabilities.shape == (80,)
