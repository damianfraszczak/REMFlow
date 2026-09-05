"""Actor-oriented ordinal sender and receiver estimation tests."""

import numpy as np
import pandas as pd
import pytest

from remflow import AIC, WAIC, aomstats, available_backends, diagnostics, remify, remstimate
from remflow.estimate import ActorRemEstimate


def _actor_problem():
    senders = [1, 2, 3, 1, 3, 2] * 4
    receivers = [2, 1, 1, 3, 2, 3] * 4
    history = remify(
        pd.DataFrame(
            {
                "time": range(1, len(senders) + 1),
                "actor1": senders,
                "actor2": receivers,
            }
        ),
        actors=[1, 2, 3],
        model="actor",
        ordinal=True,
    )
    statistics = aomstats(
        reh=history,
        sender_effects="~ 0 + indegreeSender()",
        receiver_effects="~ 0 + inertia() + reciprocity()",
        first=2,
    )
    return history, statistics


def test_actor_ordinal_fit_has_separate_sender_and_receiver_models():
    history, statistics = _actor_problem()
    fitted = remstimate(history, statistics, method="MLE")

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model.names == ["indegreeSender"]
    assert fitted.receiver_model.names == ["inertia", "reciprocity"]
    assert np.isfinite(fitted.log_likelihood)
    assert fitted.metadata["model"] == "actor"
    assert len(fitted.sender_model.event_probabilities) == len(statistics.event_indices)
    assert len(fitted.receiver_model.event_probabilities) == len(statistics.event_indices)
    for probabilities in fitted.sender_model.event_probabilities:
        np.testing.assert_allclose(probabilities.sum(), 1.0)
    for probabilities in fitted.receiver_model.event_probabilities:
        np.testing.assert_allclose(probabilities.sum(), 1.0)
    assert np.isfinite(AIC(fitted))


@pytest.mark.skipif(
    not bool(available_backends().get("jax", {}).get("available")),
    reason="JAX is not installed",
)
def test_actor_ordinal_jax_cpu_matches_numpy():
    history, statistics = _actor_problem()
    reference = remstimate(history, statistics, backend="numpy")
    accelerated = remstimate(history, statistics, backend="jax:cpu")

    assert isinstance(reference, ActorRemEstimate)
    assert isinstance(accelerated, ActorRemEstimate)
    np.testing.assert_allclose(accelerated.coef, reference.coef, rtol=1e-6, atol=1e-8)
    assert accelerated.log_likelihood == pytest.approx(reference.log_likelihood, rel=1e-8)


def test_exact_time_actor_estimation_has_sender_rate_and_receiver_choice_models():
    history, statistics = _actor_problem()
    exact_history = remify(
        history.events[["time", "sender", "receiver"]],
        actors=[1, 2, 3],
        model="actor",
    )
    exact_statistics = aomstats(
        reh=exact_history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia()",
    )
    fitted = remstimate(exact_history, exact_statistics)

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model.names == ["baseline", "indegreeSender"]
    assert fitted.receiver_model.names == ["inertia"]
    assert fitted.sender_model.metadata["engine"] == "glm"
    assert fitted.receiver_model.metadata["engine"] == "clogit"
    assert fitted.metadata["timing"] == "exact"
    assert fitted.sender_model.gradient is not None
    assert fitted.receiver_model.gradient is not None
    assert np.isfinite(fitted.log_likelihood)

    result = diagnostics(fitted, exact_history, exact_statistics)
    assert result.sender_model.residuals.shape == (len(exact_statistics.event_indices),)
    assert result.receiver_model.residuals.shape == (len(exact_statistics.event_indices),)
    assert set(result.sender_model.recall) == {"per_event", "summary"}
    assert set(result.receiver_model.recall) == {"per_event", "summary"}
    assert result.sender_model.recall["per_event"]["rel_rank"].between(0, 1).all()
    assert result.receiver_model.recall["per_event"]["rel_rank"].between(0, 1).all()


@pytest.mark.skipif(
    not bool(available_backends().get("jax", {}).get("available")),
    reason="JAX is not installed",
)
def test_exact_time_actor_jax_cpu_matches_numpy():
    history, _ = _actor_problem()
    exact_history = remify(
        history.events[["time", "sender", "receiver"]],
        actors=[1, 2, 3],
        model="actor",
    )
    statistics = aomstats(
        reh=exact_history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia()",
    )
    reference = remstimate(exact_history, statistics, backend="numpy")
    accelerated = remstimate(exact_history, statistics, backend="jax:cpu")

    np.testing.assert_allclose(accelerated.coef, reference.coef, rtol=1e-6, atol=1e-8)
    assert accelerated.log_likelihood == pytest.approx(reference.log_likelihood, rel=1e-8)


def test_actor_mle_groups_simultaneous_sender_cases_and_expands_receiver_choices():
    events = pd.DataFrame(
        {
            "time": [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
            "actor1": [1, 2, 1, 3, 2, 1, 3, 2, 4, 1],
            "actor2": [2, 3, 4, 5, 1, 3, 2, 4, 1, 5],
        }
    )
    history = remify(events, model="actor")
    statistics = aomstats(
        reh=history,
        sender_effects="~ outdegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
    )
    fitted = remstimate(history, statistics)

    assert len(statistics.sender_stats) == events["time"].nunique() - 1
    assert len(statistics.receiver_stats) == events["time"].nunique() - 1
    assert any(len(group) == 2 for group in statistics.observed_sender_groups)
    assert len(statistics.receiver_choice_stats) == len(events) - 1
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.sender_model.converged
    assert fitted.receiver_model.converged
    assert len(fitted.sender_model.event_probabilities) == len(events) - 1
    assert len(fitted.receiver_model.event_probabilities) == len(events) - 1


def test_actor_frequentist_waic_is_computed_per_component():
    history, statistics = _actor_problem()
    fitted = remstimate(
        history,
        statistics,
        WAIC=True,
        nsimWAIC=30,
        seed=44,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert np.isfinite(fitted.sender_model.metadata["WAIC"])
    assert np.isfinite(fitted.receiver_model.metadata["WAIC"])
    assert WAIC(fitted) == pytest.approx(
        fitted.sender_model.metadata["WAIC"]
        + fitted.receiver_model.metadata["WAIC"]
    )


def test_actor_hmc_returns_reproducible_component_draws_and_waic():
    history, statistics = _actor_problem()
    controls = {
        "nsim": 7,
        "nchains": 1,
        "burnin": 3,
        "thin": 1,
        "L": 4,
        "epsilon": 0.002,
    }
    fitted = remstimate(
        history,
        statistics,
        method="HMC",
        bayes=controls,
        WAIC=True,
        seed=224,
    )
    repeated = remstimate(
        history,
        statistics,
        method="HMC",
        bayes=controls,
        WAIC=True,
        seed=224,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.metadata["method"] == "HMC"
    assert fitted.metadata["approach"] == "Bayesian"
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert repeated.sender_model is not None
    assert repeated.receiver_model is not None
    assert fitted.sender_model.draws is not None
    assert fitted.receiver_model.draws is not None
    assert fitted.sender_model.draws.shape == (7, 1)
    assert fitted.receiver_model.draws.shape == (7, 2)
    assert list(fitted.to_dict()) == ["sender_model", "receiver_model"]
    expected_component_fields = [
        "coefficients",
        "post.mean",
        "vcov",
        "sd",
        "loglik",
        "draws",
        "df.null",
        "df.model",
        "df.residual",
    ]
    assert list(fitted.sender_model.to_dict()) == expected_component_fields
    assert list(fitted.receiver_model.to_dict()) == expected_component_fields
    np.testing.assert_array_equal(
        fitted.sender_model.draws,
        repeated.sender_model.draws,
    )
    np.testing.assert_array_equal(
        fitted.receiver_model.draws,
        repeated.receiver_model.draws,
    )
    assert np.isfinite(WAIC(fitted))

    result = diagnostics(fitted, history, statistics)
    assert result.sender_model is not None
    assert result.receiver_model is not None
    assert np.isfinite(result.sender_model.residuals).all()
    assert np.isfinite(result.receiver_model.residuals).all()


def test_exact_time_actor_hmc_includes_sender_rate_baseline():
    ordinal_history, _ = _actor_problem()
    history = remify(
        ordinal_history.events[["time", "sender", "receiver"]],
        actors=[1, 2, 3],
        model="actor",
        ordinal=False,
    )
    statistics = aomstats(
        reh=history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia()",
        first=2,
    )
    fitted = remstimate(
        history,
        statistics,
        approach="Bayesian",
        bayes={
            "nsim": 5,
            "nchains": 1,
            "burnin": 2,
            "L": 3,
            "epsilon": 0.002,
        },
        seed=773,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.sender_model.names == ["baseline", "indegreeSender"]
    assert fitted.receiver_model.names == ["inertia"]
    assert fitted.sender_model.draws is not None
    assert fitted.receiver_model.draws is not None
    assert fitted.sender_model.draws.shape == (5, 2)
    assert fitted.receiver_model.draws.shape == (5, 1)
