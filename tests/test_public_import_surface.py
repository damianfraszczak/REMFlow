import remflow


def test_single_library_public_workflow():
    history = remflow.remify(
        [{"sender": "A", "receiver": "B"}, {"sender": "B", "receiver": "A"}],
        ordinal=True,
    )
    stats = remflow.remstats(history, tie_effects="~ inertia() + reciprocity()")
    fit = remflow.remstimate(history, stats)

    assert fit.metadata["backend"] == "numpy"
    assert remflow.is_remify_durem(history) is False


def test_public_exports_are_unique_and_core_workflow_is_present():
    assert len(remflow.__all__) == len(set(remflow.__all__))
    assert {
        "remify",
        "remstats",
        "remstimate",
        "diagnostics",
        "RelationalEventModel",
        "inertia",
        "reciprocity",
        "otp",
    }.issubset(remflow.__all__)


def test_named_effects_are_constructible_inside_remflow():
    for name in ["otp", "psABBA", "recencyContinue", "activeTie"]:
        effect = getattr(remflow, name)()
        assert effect.name == name


def test_named_effects_compute_in_single_library():
    history = remflow.remify(
        [
            {"sender": "A", "receiver": "B"},
            {"sender": "B", "receiver": "C"},
            {"sender": "A", "receiver": "C"},
        ],
        actors=["A", "B", "C"],
        ordinal=True,
    )

    stats = remflow.remstats(
        history,
        tie_effects="~ otp() + psABBA() + recencyContinue() + activeSharedPartners_otp()",
        first=3,
    )

    assert stats.names == [
        "baseline",
        "otp",
        "psABBA",
        "recencyContinue",
        "activeSharedPartners_otp",
    ]
    assert stats.stats[0].shape == (6, 5)
