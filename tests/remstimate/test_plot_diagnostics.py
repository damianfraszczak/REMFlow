"""Plot-data and diagnostics regression tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from remflow import diagnostics, remify, remstats, remstimate
from remflow.estimate import ActorDiagnostics, ActorRemEstimate, Diagnostics, RemEstimate


def _events() -> pd.DataFrame:
    rng = np.random.default_rng(912)
    senders = rng.integers(1, 6, size=36)
    receivers = rng.integers(1, 5, size=36)
    receivers += receivers >= senders
    times = np.cumsum(rng.integers(1, 4, size=36))
    for index in range(5, 36, 5):
        times[index] = times[index - 1]
    return pd.DataFrame({"time": times, "actor1": senders, "actor2": receivers})


@dataclass(frozen=True)
class _Problem:
    history: object
    statistics: object
    fit: RemEstimate | ActorRemEstimate
    diagnostic: Diagnostics | ActorDiagnostics


@pytest.fixture(scope="module")
def actor_problem() -> _Problem:
    history = remify(_events(), actors=[1, 2, 3, 4, 5], model="actor")
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
        first=6,
        last=30,
    )
    fit = remstimate(history, statistics, method="MLE", WAIC=True, nsimWAIC=20, seed=7)
    return _Problem(history, statistics, fit, diagnostics(fit, history, statistics))


@pytest.fixture(scope="module")
def tie_problem() -> _Problem:
    history = remify(_events(), actors=[1, 2, 3, 4, 5], model="tie")
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity()",
        first=6,
        last=30,
    )
    fit = remstimate(history, statistics, method="MLE", WAIC=True, nsimWAIC=20, seed=7)
    return _Problem(history, statistics, fit, diagnostics(fit, history, statistics))


def test_diagnostics_output_and_recall_structure(actor_problem: _Problem, tie_problem: _Problem):
    actor = actor_problem.diagnostic
    tie = tie_problem.diagnostic
    assert isinstance(actor, ActorDiagnostics)
    assert isinstance(tie, Diagnostics)
    assert {"sender_model", "receiver_model", ".reh.processed"}.issubset(
        actor.to_dict()
    )
    assert {"residuals", "rates", "recall", ".reh.processed"}.issubset(tie.to_dict())
    assert actor.sender_model is not None
    assert actor.receiver_model is not None
    for component in (actor.sender_model, actor.receiver_model):
        assert component.residuals.size > 0
        assert component.rates
        assert set(component.recall) == {"per_event", "summary"}
        assert {
            "mean_rel_rank",
            "median_rel_rank",
            "mean_cum_prob",
            "top_pct",
            "top_pct_prop",
        }.issubset(component.recall["summary"])
        assert {"event", "rel_rank", "cum_prob"}.issubset(
            component.recall["per_event"].columns
        )
        assert component.recall["per_event"]["rel_rank"].between(0, 1).all()


def test_actor_diagnostic_plot_data_selection_and_return_contract(actor_problem: _Problem):
    diagnostic = actor_problem.diagnostic
    assert isinstance(diagnostic, ActorDiagnostics)
    panels = diagnostic.plot_data(which=(1, 2))
    assert set(panels) == {
        "sender.panel1",
        "sender.panel2",
        "receiver.panel1",
        "receiver.panel2",
    }
    selected = diagnostic.plot_data(
        which=2,
        sender_effects="indegreeSender",
        receiver_effects="inertia",
    )
    assert set(selected["sender.panel2"]["effect"]) == {"indegreeSender"}
    assert set(selected["receiver.panel2"]["effect"]) == {"inertia"}
    receiver_only = diagnostic.plot_data(
        which=2,
        sender_effects=None,
        receiver_effects="reciprocity",
    )
    assert set(receiver_only) == {"receiver.panel2"}
    with pytest.raises(ValueError, match="not found"):
        diagnostic.plot(which=2, sender_effects="NONEXISTENT_STAT")
    with pytest.warns(UserWarning, match="require an HMC"):
        diagnostic.plot_data(which=(1, 2, 3, 4), object=actor_problem.fit)  # type: ignore[arg-type]
    assert diagnostic.plot(which=1) is diagnostic


def test_tie_diagnostic_plot_data_effect_selection(tie_problem: _Problem):
    diagnostic = tie_problem.diagnostic
    assert isinstance(diagnostic, Diagnostics)
    assert set(diagnostic.plot_data(which=(1, 2))) == {"panel1", "panel2"}
    inertia = diagnostic.plot_data(which=2, effects="inertia")["panel2"]
    assert set(inertia["effect"]) == {"inertia"}
    both = diagnostic.plot_data(which=2, effects=["inertia", "reciprocity"])
    assert set(both["panel2"]["effect"]) == {"inertia", "reciprocity"}
    with pytest.raises(ValueError, match="not found"):
        diagnostic.plot(which=2, effects="NONEXISTENT")
    assert set(
        diagnostic.plot_data(which=(1, 2, 3, 4), object=tie_problem.fit)  # type: ignore[arg-type]
    ) == {"panel1", "panel2"}


def test_fit_plot_backward_compatibility_and_validation(
    actor_problem: _Problem, tie_problem: _Problem
):
    actor_fit = actor_problem.fit
    tie_fit = tie_problem.fit
    assert isinstance(actor_fit, ActorRemEstimate)
    assert isinstance(tie_fit, RemEstimate)
    assert actor_fit.plot(
        reh=actor_problem.history,  # type: ignore[arg-type]
        stats=actor_problem.statistics,  # type: ignore[arg-type]
        which=1,
    ) is actor_problem.diagnostic or isinstance(
        actor_fit.plot(
            reh=actor_problem.history,  # type: ignore[arg-type]
            diagnostics=actor_problem.diagnostic,
            which=1,
        ),
        ActorDiagnostics,
    )
    assert isinstance(
        tie_fit.plot(
            reh=tie_problem.history,  # type: ignore[arg-type]
            stats=tie_problem.statistics,  # type: ignore[arg-type]
            which=(1, 2),
        ),
        Diagnostics,
    )
    assert tie_fit.plot(
        reh=tie_problem.history,  # type: ignore[arg-type]
        diagnostics=tie_problem.diagnostic,
        which=2,
        effects="inertia",
    ) is tie_problem.diagnostic
    with pytest.raises(ValueError, match="stats.*provided"):
        actor_fit.plot(reh=actor_problem.history, which=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="diagnostics"):
        actor_fit.plot(
            reh=actor_problem.history,  # type: ignore[arg-type]
            diagnostics={"foo": 1},  # type: ignore[arg-type]
            which=1,
        )


@pytest.mark.parametrize("chains", [1, 2])
def test_hmc_posterior_and_trace_plot_data(actor_problem: _Problem, chains: int):
    fitted = remstimate(
        actor_problem.history,  # type: ignore[arg-type]
        actor_problem.statistics,  # type: ignore[arg-type]
        method="HMC",
        nchains=chains,
        nsim=15,
        burnin=5,
        L=4,
        epsilon=0.002,
        seed=12345,
    )
    diagnostic = diagnostics(
        fitted,
        actor_problem.history,  # type: ignore[arg-type]
        actor_problem.statistics,  # type: ignore[arg-type]
    )
    assert isinstance(fitted, ActorRemEstimate)
    assert isinstance(diagnostic, ActorDiagnostics)
    panels = diagnostic.plot_data(
        which=(3, 4),
        sender_effects="indegreeSender",
        object=fitted,
    )
    assert {"sender.panel3", "sender.panel4", "receiver.panel3", "receiver.panel4"}.issubset(
        panels
    )
    assert panels["sender.panel3"]["chain"].nunique() == chains
    assert diagnostic.plot(which=4, object=fitted) is diagnostic
