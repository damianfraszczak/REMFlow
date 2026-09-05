"""Case-control sampling invariants and reproducibility tests."""

import numpy as np
import pandas as pd

from remflow import remify, remstats, stack_stats
from remflow.estimate import _loglik_and_grad, _numpy_exact_objective


def _history():
    return remify(
        pd.DataFrame(
            {
                "time": range(1, 9),
                "actor1": [1, 1, 2, 3, 4, 2, 3, 1],
                "actor2": [2, 3, 1, 1, 1, 4, 2, 4],
            }
        ),
        actors=[1, 2, 3, 4],
        riskset="full",
    )


def test_sampled_statistics_are_reproducible_full_tensor_slices():
    history = _history()
    effects = "~ inertia() + reciprocity() + outdegreeSender() + otp()"
    full = remstats(history, tie_effects=effects, first=1)
    sampled = remstats(
        history,
        tie_effects=effects,
        first=1,
        sampling=True,
        samp_num=5,
        seed=11,
    )
    repeated = remstats(
        history,
        tie_effects=effects,
        first=1,
        sampling=True,
        samp_num=5,
        seed=11,
    )

    assert sampled.names == full.names
    assert len(sampled.sample_map) == len(full.stats)
    for event, indexes in enumerate(sampled.sample_map):
        zero_based = indexes - 1
        np.testing.assert_array_equal(sampled.stats[event], full.stats[event][zero_based])
        np.testing.assert_array_equal(indexes, repeated.sample_map[event])
        np.testing.assert_array_equal(sampled.stats[event], repeated.stats[event])
        assert full.observed_indices[event] in zero_based


def test_sampled_stack_has_case_control_weights_and_public_dyad_map():
    history = _history()
    sampled = remstats(
        history,
        tie_effects="~ inertia()",
        first=1,
        sampling=True,
        samp_num=5,
        seed=7,
    )
    stacked = stack_stats(sampled)
    frame = stacked.remstats_stack

    assert stacked.S == 5
    assert stacked.E == len(sampled.stats)
    assert len(frame) == stacked.S * stacked.E
    assert frame["obs"].sum() == stacked.E
    np.testing.assert_array_equal(
        frame["dyad"].to_numpy().reshape(stacked.E, stacked.S),
        np.stack(sampled.sample_map),
    )
    assert (frame.loc[frame["obs"] == 1, "weight"] == 1.0).all()
    np.testing.assert_allclose(
        frame.loc[frame["obs"] == 0, "weight"],
        (12 - 1) / (5 - 1),
    )


def test_sampling_weights_enter_ordinal_and_exact_likelihood_denominators():
    design = np.array([[1.0, 0.0], [1.0, 1.0]])
    weights = [np.array([1.0, 3.0])]
    loglik, gradient = _loglik_and_grad(
        np.zeros(2), [design], [0], sampling_weights=weights
    )

    np.testing.assert_allclose(loglik, -np.log(4.0))
    np.testing.assert_allclose(gradient, [0.0, -0.75])

    exact = _numpy_exact_objective(
        [design], [0], np.array([2.0]), sampling_weights=weights
    )
    negative_loglik, negative_gradient = exact(np.zeros(2))
    np.testing.assert_allclose(negative_loglik, 8.0)
    np.testing.assert_allclose(negative_gradient, [7.0, 6.0])
