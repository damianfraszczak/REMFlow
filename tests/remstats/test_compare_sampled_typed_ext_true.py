"""Sampled and full tensors for type-expanded risk sets."""

import pytest

from tests.remstats._typed_sampling_support import (
    DIRECTED_FORMULAS,
    UNDIRECTED_FORMULAS,
    assert_sample_matches_full,
    make_history,
)

SCENARIOS = [
    ("active", True, False, "decay", 5),
    ("full", False, False, "decay", 5),
    ("manual", True, False, "decay", 5),
    ("full", True, True, "window", 3),
]


@pytest.mark.parametrize(
    ("riskset", "directed", "ordinal", "memory", "memory_value"), SCENARIOS
)
def test_sampled_expanded_typed_statistics_are_exact_full_tensor_slices(
    riskset, directed, ordinal, memory, memory_value
):
    history = make_history(
        riskset=riskset,
        directed=directed,
        ordinal=ordinal,
        extend_riskset_by_type=True,
    )
    formulas = DIRECTED_FORMULAS if directed else UNDIRECTED_FORMULAS
    formulas = [
        *formulas,
        '~ inertia(consider_type="interact")',
        '~ inertia(consider_type="separate") + outdegreeSender(consider_type=False)'
        if directed
        else '~ inertia(consider_type="separate") + degreeMin(consider_type=False)',
    ]
    for formula in formulas:
        assert_sample_matches_full(
            history, formula, memory=memory, memory_value=memory_value
        )

    assert "event_type" in history.risksets[0]
    from remflow import tomstats

    interaction = tomstats(
        '~ inertia(consider_type="interact")', reh=history, first=1
    )
    assert {
        "inertia.social.social",
        "inertia.social.work",
        "inertia.work.social",
        "inertia.work.work",
    }.issubset(interaction.names)
