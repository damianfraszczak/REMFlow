"""Typed undirected manual decay-memory sampling."""

import pytest

from tests.remstats._typed_sampling_support import (
    UNDIRECTED_FORMULAS,
    assert_sample_matches_full,
    make_history,
)


@pytest.mark.parametrize("effect_formula", UNDIRECTED_FORMULAS)
def test_typed_manual_undirected_decay_sample_is_exact_full_slice(effect_formula):
    history = make_history(
        riskset="manual",
        directed=False,
        ordinal=False,
        extend_riskset_by_type=False,
    )
    assert_sample_matches_full(
        history,
        effect_formula,
        memory="decay",
        memory_value=1000.0,
    )
