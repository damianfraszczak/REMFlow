"""Active directed interval-sampling consistency tests."""

import numpy as np
import pandas as pd
import pytest

from tests.remstats._typed_sampling_support import (
    DIRECTED_BASE_FORMULAS,
    assert_sample_matches_full,
    make_history,
)


@pytest.mark.parametrize("effect_formula", DIRECTED_BASE_FORMULAS)
def test_active_directed_sample_is_reproducible_full_tensor_slice(effect_formula):
    history = make_history(
        riskset="active",
        directed=True,
        ordinal=False,
        extend_riskset_by_type=False,
    )
    assert_sample_matches_full(
        history,
        effect_formula,
        memory="full",
        memory_value=None,
    )


@pytest.mark.parametrize("representation", ["wide", "long"])
def test_sampled_dyad_covariates_match_full_for_wide_and_long_inputs(representation):
    history = make_history(
        riskset="active",
        directed=True,
        ordinal=False,
        extend_riskset_by_type=False,
    )
    matrix = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=float,
    )
    if representation == "wide":
        attributes = matrix
    else:
        attributes = pd.DataFrame(
            [
                (sender, receiver, matrix[sender - 1, receiver - 1])
                for sender in range(1, 5)
                for receiver in range(1, 5)
                if sender != receiver
            ],
            columns=["actor1", "actor2", "both_male"],
        )
    assert_sample_matches_full(
        history,
        '~ tie(variable="both_male")',
        memory="full",
        memory_value=None,
        attr_dyads=attributes,
    )
