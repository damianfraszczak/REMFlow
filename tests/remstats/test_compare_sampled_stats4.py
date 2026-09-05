"""Directed typed active decay-memory sampling."""

import pytest

from tests.remstats._typed_sampling_support import (
    DIRECTED_FORMULAS,
    assert_sample_matches_full,
    make_history,
)


@pytest.mark.parametrize("effect_formula", DIRECTED_FORMULAS)
def test_typed_active_directed_decay_sample_is_exact_full_slice(effect_formula):
    history = make_history(
        riskset="active",
        directed=True,
        ordinal=False,
        extend_riskset_by_type=False,
    )
    assert_sample_matches_full(
        history,
        effect_formula,
        memory="decay",
        memory_value=1000.0,
    )
