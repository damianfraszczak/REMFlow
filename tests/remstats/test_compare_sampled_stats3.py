"""Undirected ordinal decay-memory sampling consistency tests."""

import numpy as np
import pytest

from remflow import userStat
from tests.remstats._typed_sampling_support import (
    UNDIRECTED_BASE_FORMULAS,
    assert_sample_matches_full,
    make_history,
)


@pytest.mark.parametrize("effect_formula", UNDIRECTED_BASE_FORMULAS)
def test_undirected_ordinal_decay_sample_is_reproducible_full_slice(effect_formula):
    history = make_history(
        riskset="full",
        directed=False,
        ordinal=True,
        extend_riskset_by_type=False,
    )
    assert_sample_matches_full(
        history,
        effect_formula,
        memory="decay",
        memory_value=3.0,
    )


def test_undirected_ordinal_user_stat_sample_is_exact_full_slice():
    history = make_history(
        riskset="full",
        directed=False,
        ordinal=True,
        extend_riskset_by_type=False,
    )
    values = np.arange(history.E * history.D, dtype=float).reshape(
        history.E,
        history.D,
    )
    assert_sample_matches_full(
        history,
        userStat(x=values, variableName="actor_event"),
        memory="decay",
        memory_value=3.0,
    )
