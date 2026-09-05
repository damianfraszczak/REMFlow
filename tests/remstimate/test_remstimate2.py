"""Core MLE pipeline regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats, remstimate
from remflow.estimate import ActorRemEstimate


def _events(*, typed: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
            "actor1": [1, 2, 1, 3, 2, 1, 3, 2, 4, 1],
            "actor2": [2, 3, 4, 5, 1, 3, 2, 4, 1, 5],
        }
    )
    if typed:
        frame["type"] = np.resize(np.asarray(["a", "b"]), len(frame))
    return frame


def _actor_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": np.repeat(np.arange(1, 6), 2),
            "time": np.tile([0, 5], 5),
            "x": [0.2, 0.4, -0.3, -0.1, 0.7, 0.5, 1.1, 0.9, -0.8, -0.6],
        }
    )


def test_tie_mle_variants_cover_timing_risksets_types_and_actor_attributes():
    events = _events()
    history = remify(events, model="tie", ordinal=False)
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    fitted = remstimate(history, statistics, method="MLE")

    assert fitted.model == "tie"
    assert fitted.ordinal is False
    assert fitted.converged
    assert fitted.names == ["baseline", "inertia", "reciprocity"]

    ordinal_history = remify(events, model="tie", ordinal=True)
    ordinal_stats = remstats(
        ordinal_history, tie_effects="~ inertia() + reciprocity()"
    )
    ordinal_fit = remstimate(ordinal_history, ordinal_stats, method="MLE")
    assert ordinal_fit.ordinal is True
    assert ordinal_fit.names == ["inertia", "reciprocity"]

    active_history = remify(events, model="tie", riskset="active")
    active_stats = remstats(active_history, tie_effects="~ inertia()")
    assert "inertia" in remstimate(active_history, active_stats).names

    manual = pd.concat(
        [
            events[["actor1", "actor2"]],
            events[["actor2", "actor1"]].set_axis(["actor1", "actor2"], axis=1),
        ],
        ignore_index=True,
    ).drop_duplicates(ignore_index=True)
    manual_history = remify(
        events,
        model="tie",
        riskset="manual",
        manual_riskset=manual,
    )
    manual_stats = remstats(manual_history, tie_effects="~ inertia()")
    manual_fit = remstimate(manual_history, manual_stats)
    assert manual_fit.converged
    assert "inertia" in manual_fit.names

    typed_history = remify(_events(typed=True), model="tie")
    typed_default = remstats(typed_history, tie_effects="~ inertia()")
    typed_separate = remstats(
        typed_history,
        tie_effects='~ inertia(consider_type="separate")',
    )
    assert "inertia" in remstimate(typed_history, typed_default).names
    assert len(remstimate(typed_history, typed_separate).coef) > 2

    with pytest.warns(DeprecationWarning, match="attr_actors"):
        attribute_stats = remstats(
            history,
            tie_effects='~ inertia() + send(variable="x")',
            attr_actors=_actor_attributes(),
        )
    attribute_fit = remstimate(history, attribute_stats)
    assert {"inertia", "send_x"}.issubset(attribute_fit.names)


def test_actor_mle_components_timing_simultaneous_types_and_attributes():
    events = _events()
    exact_history = remify(events, model="actor", ordinal=False)
    sender_stats = remstats(
        exact_history,
        sender_effects="~ outdegreeSender()",
        receiver_effects=None,
    )
    sender_fit = remstimate(exact_history, sender_stats)
    assert isinstance(sender_fit, ActorRemEstimate)
    assert sender_fit.sender_model is not None
    assert sender_fit.receiver_model is None
    assert sender_fit.sender_model.converged
    assert "baseline" in sender_fit.sender_model.names

    ordinal_history = remify(events, model="actor", ordinal=True)
    ordinal_sender_stats = remstats(
        ordinal_history,
        sender_effects="~ outdegreeSender()",
        receiver_effects=None,
    )
    ordinal_sender_fit = remstimate(ordinal_history, ordinal_sender_stats)
    assert isinstance(ordinal_sender_fit, ActorRemEstimate)
    assert ordinal_sender_fit.sender_model is not None
    assert "baseline" not in ordinal_sender_fit.sender_model.names

    receiver_stats = remstats(
        exact_history,
        sender_effects=None,
        receiver_effects="~ inertia() + reciprocity()",
    )
    receiver_fit = remstimate(exact_history, receiver_stats)
    assert isinstance(receiver_fit, ActorRemEstimate)
    assert receiver_fit.sender_model is None
    assert receiver_fit.receiver_model is not None
    assert receiver_fit.receiver_model.converged
    assert "baseline" not in receiver_fit.receiver_model.names

    both_stats = remstats(
        exact_history,
        sender_effects="~ outdegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
    )
    both_fit = remstimate(exact_history, both_stats)
    assert isinstance(both_fit, ActorRemEstimate)
    assert both_fit.sender_model is not None
    assert both_fit.receiver_model is not None
    assert both_fit.sender_model.names == ["baseline", "outdegreeSender"]
    assert both_fit.receiver_model.names == ["inertia", "reciprocity"]
    assert len(both_stats.sender_stats) == events["time"].nunique() - 1
    assert len(both_stats.receiver_stats) == events["time"].nunique() - 1
    assert both_fit.receiver_model.converged

    typed_history = remify(_events(typed=True), model="actor")
    typed_stats = remstats(
        typed_history,
        sender_effects="~ outdegreeSender()",
        receiver_effects="~ inertia()",
    )
    typed_fit = remstimate(typed_history, typed_stats)
    assert isinstance(typed_fit, ActorRemEstimate)
    assert typed_fit.sender_model is not None
    assert typed_fit.receiver_model is not None

    with pytest.warns(DeprecationWarning, match="attr_actors"):
        attribute_stats = remstats(
            exact_history,
            sender_effects='~ outdegreeSender() + send(variable="x")',
            receiver_effects='~ inertia() + receive(variable="x")',
            attr_actors=_actor_attributes(),
        )
    attribute_fit = remstimate(exact_history, attribute_stats)
    assert isinstance(attribute_fit, ActorRemEstimate)
    assert attribute_fit.sender_model is not None
    assert attribute_fit.receiver_model is not None
    assert "send_x" in attribute_fit.sender_model.names
    assert "receive_x" in attribute_fit.receiver_model.names


def test_tie_and_actor_mle_results_expose_documented_output_fields():
    history = remify(_events(), model="tie")
    statistics = remstats(history, tie_effects="~ inertia()")
    fitted = remstimate(history, statistics)
    required = {
        "coefficients",
        "loglik",
        "AIC",
        "AICC",
        "BIC",
        "vcov",
        "se",
        "null.deviance",
        "residual.deviance",
        "model.deviance",
        "converged",
        "iterations",
        "df.null",
        "df.model",
        "df.residual",
    }
    assert required.issubset(fitted.to_dict())

    actor_history = remify(_events(), model="actor")
    actor_stats = remstats(
        actor_history,
        sender_effects="~ outdegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
    )
    actor_fit = remstimate(actor_history, actor_stats)
    assert isinstance(actor_fit, ActorRemEstimate)
    assert actor_fit.sender_model is not None
    assert actor_fit.receiver_model is not None
    assert required.issubset(actor_fit.sender_model.to_dict())
    assert required.issubset(actor_fit.receiver_model.to_dict())
