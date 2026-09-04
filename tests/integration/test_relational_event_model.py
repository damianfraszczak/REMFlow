import numpy as np
import pytest

from remflow import RelationalEventModel

EVENTS = [
    (1.0, "u1", "u2", "retweet", "support"),
    (2.0, "u2", "u3", "reply", "deny"),
    (3.0, "u1", "u3", "mention", "question"),
    (4.0, "u3", "u1", "retweet", "support"),
    (5.0, "u2", "u1", "reply", "deny"),
    (6.0, "u3", "u2", "mention", "question"),
]


def test_misinformation_facade_fits_predicts_and_reports_roles():
    model = RelationalEventModel(
        effects=(
            "reciprocity",
            "sender_activity",
            "receiver_popularity",
            "stance_similarity",
        ),
        ordinal=True,
    ).fit(EVENTS)

    summary = model.summary()
    prediction = model.predict_next_events(top_k=100)
    roles = model.actor_roles()

    assert summary["events"] == 6
    assert summary["actors"] == 3
    assert summary["timing"] == "ordinal"
    assert model.fit_result_ is not None
    assert model.fit_result_.names == [
        "reciprocity",
        "outdegreeSender",
        "indegreeReceiver",
        "same_stance",
    ]
    assert set(prediction).issuperset({"sender", "receiver", "event_type", "probability"})
    np.testing.assert_allclose(prediction["probability"].sum(), 1.0)
    assert set(roles["actor"]) == {"u1", "u2", "u3"}
    assert {
        "source_score",
        "amplifier_score",
        "intermediary_score",
    }.issubset(roles.columns)


def test_intervention_removes_probability_mass_and_renormalizes():
    model = RelationalEventModel(effects=("reciprocity",), ordinal=True).fit(EVENTS)

    result = model.simulate_intervention(blocked_actors=["u1"])

    assert 0 < result["probability_mass_removed"] < 1
    assert not result["next_events"]["sender"].eq("u1").any()
    assert not result["next_events"]["receiver"].eq("u1").any()
    np.testing.assert_allclose(result["next_events"]["probability"].sum(), 1.0)


def test_facade_validates_fit_state_effects_and_tuple_shape():
    model = RelationalEventModel()
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict_next_events()
    with pytest.raises(ValueError, match="unknown high-level effect"):
        RelationalEventModel(effects=("not_an_effect",)).fit(EVENTS)
    with pytest.raises(ValueError, match="3-5 fields"):
        RelationalEventModel().fit([(1, "a")])


def test_source_detection_and_echo_chamber_trajectory_are_transparent():
    model = RelationalEventModel(effects=("reciprocity",), ordinal=True).fit(EVENTS)

    sources = model.detect_sources(top_k=2)
    echo = model.echo_chamber_metrics()

    assert len(sources) == 2
    assert sources["source_score"].between(0, 1).all()
    assert sources["downstream_reach"].between(0, 2).all()
    assert echo["comparable_events"] == len(EVENTS)
    assert echo["within_stance_share"] == pytest.approx(1 / 3)
    assert echo["echo_chamber_score"] == pytest.approx(-1 / 3)
    assert len(echo["trajectory"]) == len(EVENTS)


def test_echo_chamber_metrics_require_stance_data():
    events_without_stance = [row[:4] for row in EVENTS]
    model = RelationalEventModel(effects=("reciprocity",), ordinal=True).fit(
        events_without_stance
    )

    with pytest.raises(ValueError, match="stance"):
        model.echo_chamber_metrics()
