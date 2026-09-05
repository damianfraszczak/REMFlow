"""Directed manual decay-memory sampling consistency tests."""

import pytest

from tests.remstats._typed_sampling_support import (
    DIRECTED_BASE_FORMULAS,
    DIRECTED_PROPORTIONAL_FORMULAS,
    assert_sample_matches_full,
    make_history,
)


@pytest.mark.parametrize(
    "effect_formula",
    [*DIRECTED_BASE_FORMULAS, *DIRECTED_PROPORTIONAL_FORMULAS],
)
def test_manual_directed_decay_sample_is_reproducible_full_slice(effect_formula):
    history = make_history(
        riskset="manual",
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
