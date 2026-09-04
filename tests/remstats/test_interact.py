"""Tie-model interaction effects."""

import numpy as np
import pandas as pd
import pytest

from remflow import remify, remstats


def test_colon_interaction_preserves_effect_arguments_and_main_effects():
    events = pd.DataFrame(
        {
            "time": range(1, 6),
            "actor1": [1, 1, 2, 2, 3],
            "actor2": [2, 3, 1, 3, 2],
        }
    )
    actors = pd.DataFrame(
        {
            "name": [1, 2, 3],
            "time": [0, 0, 0],
            "x1": [10, 20, 30],
        }
    )
    history = remify(events, riskset="active")

    with pytest.warns(DeprecationWarning, match="attr_actors"):
        result = remstats(
            history,
            tie_effects='~ send(variable="x1"):inertia()',
            attr_actors=actors,
        )

    assert result.names == ["baseline", "send_x1", "inertia", "send_x1:inertia"]
    send_index = result.names.index("send_x1")
    inertia_index = result.names.index("inertia")
    interaction_index = result.names.index("send_x1:inertia")
    for matrix in result.stats:
        np.testing.assert_array_equal(
            matrix[:, interaction_index],
            matrix[:, send_index] * matrix[:, inertia_index],
        )


def test_star_builder_and_string_formula_have_equivalent_interactions():
    from remflow import formula, inertia, reciprocity

    built = formula(inertia() * reciprocity())
    parsed = formula("~ inertia() * reciprocity()")

    assert parsed.canonical() == built.canonical()
