"""Type-expanded undirected ordinal-window sampling."""

import pytest

from tests.remstats._typed_sampling_support import (
    UNDIRECTED_FORMULAS,
    assert_sample_matches_full,
    make_history,
)


@pytest.mark.parametrize("effect_formula", UNDIRECTED_FORMULAS)
def test_type_expanded_ordinal_window_sample_is_exact_full_slice(effect_formula):
    history = make_history(
        riskset="full",
        directed=False,
        ordinal=True,
        extend_riskset_by_type=True,
    )
    assert_sample_matches_full(
        history,
        effect_formula,
        memory="window",
        memory_value=3.0,
    )
