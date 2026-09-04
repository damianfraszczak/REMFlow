import numpy as np
import pandas as pd

from remflow import formula, inertia, reciprocity, remify, remstats


def test_string_and_builder_formula_have_same_terms():
    string_formula = formula("~ remflow::inertia() + reciprocity()")
    builder_formula = formula(inertia() + reciprocity())

    assert string_formula.canonical() == builder_formula.canonical()


def test_inertia_and_reciprocity_reference_kernel():
    history = remify(
        pd.DataFrame(
            {
                "sender": ["A", "B", "A"],
                "receiver": ["B", "A", "B"],
            }
        ),
        actors=["A", "B"],
        riskset="full",
        ordinal=True,
    )

    stats = remstats(history, tie_effects="~ inertia() + reciprocity()", first=2)

    assert stats.names == ["baseline", "inertia", "reciprocity"]
    np.testing.assert_array_equal(stats.stats[0], np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
    np.testing.assert_array_equal(stats.stats[1], np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    assert stats.observed_indices == [1, 0]
