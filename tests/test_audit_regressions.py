import numpy as np
import pandas as pd
import pytest

from remflow import (
    diagnostics,
    dyad,
    event,
    formula,
    inertia,
    remify,
    remstats,
    remstimate,
    same,
    send,
    userStat,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0],
            "sender": ["A", "B", "A"],
            "receiver": ["B", "A", "B"],
        }
    )


def test_unknown_arguments_and_invalid_extension_controls_are_not_silently_ignored():
    with pytest.raises(TypeError, match="unexpected keyword"):
        remify(_events(), definitely_not_an_argument=True)

    exact = remify(_events(), actors=["A", "B"])
    exact_stats = remstats(exact, tie_effects="~ 1")
    assert remstimate(exact, exact_stats).metadata["timing"] == "exact"

    ordinal = remify(_events(), actors=["A", "B"], ordinal=True)
    ordinal_stats = remstats(ordinal, tie_effects="~ inertia()")
    with pytest.raises(TypeError, match="penalty must be a mapping"):
        remstimate(ordinal, ordinal_stats, penalty="lasso")
    sampled = remstats(
        ordinal, tie_effects="~ inertia()", sampling=True, samp_num=1, seed=1
    )
    assert all(len(indexes) == 1 for indexes in sampled.sample_map)
    assert all(group == [0] for group in sampled.observed_index_groups)


def test_type_expanded_riskset_matches_the_observed_event_type():
    events = _events().assign(kind=["post", "reply", "post"])
    history = remify(
        events,
        actors=["A", "B"],
        ordinal=True,
        event_type="kind",
        extend_riskset_by_type=True,
    )

    stats = remstats(history, tie_effects="~ inertia()", first=2)

    assert len(stats.observed_indices) == 2
    for event_index, observed_index in zip(
        stats.event_indices, stats.observed_indices, strict=True
    ):
        event = history.events.iloc[event_index]
        observed = history.risksets[event_index].iloc[observed_index]
        assert observed["sender_id"] == event["sender_id"]
        assert observed["receiver_id"] == event["receiver_id"]
        assert observed["event_type"] == event["event_type"]


def test_event_weights_affect_endogenous_counts():
    history = remify(
        _events(),
        actors=["A", "B"],
        ordinal=False,
        event_weight=[2.0, 1.0, 1.0],
    )

    stats = remstats(history, tie_effects="~ inertia()", first=3, last=3)

    np.testing.assert_allclose(stats.stats[0][:, stats.names.index("inertia")], [2.0, 1.0])


def test_decay_memory_uses_documented_half_life_weighting():
    history = remify(
        _events(),
        actors=["A", "B"],
        ordinal=False,
        event_weight=[2.0, 1.0, 1.0],
    )

    stats = remstats(
        history,
        tie_effects="~ inertia()",
        first=3,
        last=3,
        memory="decay",
        memory_value=10.0,
    )

    np.testing.assert_allclose(
        stats.stats[0][:, stats.names.index("inertia")], [1.0, 1.0], rtol=1e-14
    )


def test_actor_covariates_are_computed_instead_of_returning_zero():
    history = remify(_events(), actors=["A", "B"], ordinal=True)
    attributes = pd.DataFrame({"name": ["A", "B"], "age": [10.0, 20.0], "group": [1, 2]})

    stats = remstats(
        history,
        tie_effects=send("age", attr_actors=attributes) + same("group", attr_actors=attributes),
        first=2,
        last=2,
    )

    np.testing.assert_allclose(stats.stats[0], [[1.0, 10.0, 0.0], [1.0, 20.0, 0.0]])


def test_deprecated_global_actor_attributes_remain_a_working_fallback():
    history = remify(_events(), actors=["A", "B"], ordinal=True, riskset="active")
    attributes = pd.DataFrame(
        {
            "name": ["A", "B", "A", "B"],
            "time": [0, 0, 3, 3],
            "score": [10.0, 20.0, 100.0, 200.0],
        }
    )

    with pytest.warns(DeprecationWarning, match="effect omits attr_actors"):
        stats = remstats(
            history,
            tie_effects=('~ send(variable="score") + difference(variable="score", absolute=FALSE)'),
            attr_actors=attributes,
            first=1,
            last=3,
        )

    np.testing.assert_allclose(stats.stats[0], [[1.0, 10.0, -10.0], [1.0, 20.0, 10.0]])
    np.testing.assert_allclose(stats.stats[2], [[1.0, 100.0, -100.0], [1.0, 200.0, 100.0]])
    assert stats.names == ["baseline", "send_score", "difference_score"]


def test_dyad_event_and_user_statistics_are_computed():
    history = remify(_events(), actors=["A", "B"], ordinal=True)
    dyad_values = np.array([[0.0, 3.0], [4.0, 0.0]])
    event_values = pd.DataFrame({"magnitude": [10.0, 20.0, 30.0]})
    user_values = np.array([[1.0, 2.0], [5.0, 6.0], [8.0, 9.0]])

    stats = remstats(
        history,
        tie_effects=(
            dyad("strength", attr_dyads=dyad_values)
            + event("magnitude", event_attr=event_values)
            + userStat(user_values)
        ),
        first=2,
        last=2,
    )

    np.testing.assert_allclose(stats.stats[0], [[1.0, 3.0, 20.0, 5.0], [1.0, 4.0, 20.0, 6.0]])


def test_formula_parser_supports_safe_literals_keywords_and_namespaces():
    parsed = formula(
        '~ remstats::inertia(scaling="none") + FEtype() + reciprocity(consider_type=FALSE)'
    )

    assert parsed.canonical() == {
        "intercept": None,
        "terms": [
            "inertia(scaling='none')",
            "FEtype",
            "reciprocity(consider_type=False)",
        ],
    }
    with pytest.raises(ValueError, match="unsupported formula argument"):
        formula("~ inertia(scaling=__import__('os').system('echo unsafe'))")


def test_undirected_events_are_canonical_and_missing_risksets_fail_clearly():
    history = remify(
        pd.DataFrame({"time": [1.0], "sender": ["B"], "receiver": ["A"]}),
        actors=["A", "B"],
        directed=False,
        ordinal=True,
    )

    assert history.events.loc[0, ["sender", "receiver"]].to_list() == ["A", "B"]
    assert history.risksets[0].loc[0, ["sender", "receiver"]].to_list() == ["A", "B"]

    detached = remify(_events(), actors=["A", "B"], ordinal=True, attach_riskset=False)
    with pytest.raises(ValueError, match="no attached risk sets"):
        remstats(detached, tie_effects="~ inertia()")


def test_supported_fit_produces_nonempty_predictive_diagnostics():
    history = remify(_events(), actors=["A", "B"], ordinal=True)
    stats = remstats(history, tie_effects=inertia(), first=2)

    fit = remstimate(history, stats)
    result = diagnostics(fit)

    assert len(result.residuals) == len(stats.stats)
    assert len(result.ranks) == len(stats.stats)
    assert np.all((result.observed_probabilities >= 0) & (result.observed_probabilities <= 1))
    for probabilities in fit.event_probabilities:
        np.testing.assert_allclose(probabilities.sum(), 1.0)
