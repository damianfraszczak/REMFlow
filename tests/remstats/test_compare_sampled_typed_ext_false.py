"""Sampled and full tensors for unexpanded typed risk sets."""

import pytest

from remflow import tomstats
from tests.remstats._typed_sampling_support import (
    DIRECTED_FORMULAS,
    UNDIRECTED_FORMULAS,
    assert_sample_matches_full,
    make_history,
)

SCENARIOS = [
    ("active", True, False, "decay", 5),
    ("full", False, False, "decay", 5),
    ("manual", False, False, "decay", 5),
    ("full", False, True, "window", 3),
]


@pytest.mark.parametrize(
    ("riskset", "directed", "ordinal", "memory", "memory_value"), SCENARIOS
)
def test_sampled_unexpanded_typed_statistics_are_exact_full_tensor_slices(
    riskset, directed, ordinal, memory, memory_value
):
    history = make_history(
        riskset=riskset,
        directed=directed,
        ordinal=ordinal,
        extend_riskset_by_type=False,
    )
    formulas = DIRECTED_FORMULAS if directed else UNDIRECTED_FORMULAS
    for formula in formulas:
        assert_sample_matches_full(
            history, formula, memory=memory, memory_value=memory_value
        )

    assert "event_type" not in history.risksets[0]
    shape = tomstats(
        "~ inertia(consider_type=True)",
        reh=history,
        first=1,
        sampling=True,
        samp_num=5,
        seed=1,
    )
    assert shape.names == ["baseline", "inertia.social", "inertia.work"]
