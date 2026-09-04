"""Validation and warning regression cases for statistic construction."""

import numpy as np
import pandas as pd
import pytest

from remflow import (
    average,
    degreeDiff,
    degreeMax,
    degreeMin,
    difference,
    formula,
    indegreeReceiver,
    indegreeSender,
    inertia,
    isp,
    itp,
    maximum,
    minimum,
    osp,
    otp,
    outdegreeReceiver,
    outdegreeSender,
    receive,
    reciprocity,
    remify,
    remstats,
    send,
    sp,
    spUnique,
    tie,
    tomstats,
    totaldegreeDyad,
    totaldegreeReceiver,
    totaldegreeSender,
    userStat,
)


def _events():
    return pd.DataFrame(
        {
            "time": range(1, 6),
            "actor1": [1, 1, 2, 2, 3],
            "actor2": [2, 3, 1, 3, 2],
        }
    )


def test_deprecated_effect_arguments_emit_actionable_warnings():
    with pytest.warns(DeprecationWarning, match="scaling.*none"):
        inertia(scaling="as.is")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        tie(variable="test")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        spUnique()
    values = np.arange(15, dtype=float).reshape(5, 3)
    values[0, 0] = np.nan
    with pytest.warns(UserWarning, match="missing values"):
        userStat(values)


@pytest.mark.parametrize(
    "factory",
    [
        indegreeSender,
        outdegreeSender,
        totaldegreeSender,
        indegreeReceiver,
        outdegreeReceiver,
        totaldegreeReceiver,
        degreeDiff,
        degreeMax,
        degreeMin,
        totaldegreeDyad,
        inertia,
        reciprocity,
        otp,
        itp,
        osp,
        isp,
        sp,
    ],
)
def test_every_endogenous_as_is_alias_warns(factory):
    with pytest.warns(DeprecationWarning, match="scaling.*none"):
        factory(scaling="as.is")


@pytest.mark.parametrize(
    "factory",
    [send, receive, average, difference, maximum, minimum],
)
def test_every_actor_covariate_as_is_alias_warns(factory):
    with pytest.warns(DeprecationWarning, match="scaling.*none"):
        factory(variable="x1", scaling="as.is")


def test_covariate_constructor_validates_variable_and_missing_data():
    attributes = pd.DataFrame(
        {"name": [1, 2, 3], "time": [0.0, 0.0, 0.0], "x1": [10.0, 20.0, 30.0]}
    )
    with pytest.raises(TypeError, match="string"):
        send(variable=1, attr_actors=attributes)
    with pytest.raises(ValueError, match="not in attr_actors"):
        send(variable="x3", attr_actors=attributes)
    missing_time = attributes.copy()
    missing_time.loc[0, "time"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        send(variable="x1", attr_actors=missing_time)
    missing_value = attributes.copy()
    missing_value.loc[0, "x1"] = np.nan
    with pytest.warns(UserWarning, match="Missing values"):
        send(variable="x1", attr_actors=missing_value)


def test_same_accepts_categorical_values_but_arithmetic_effects_reject_them():
    history = remify(_events(), actors=[1, 2, 3])
    attributes = pd.DataFrame(
        {"name": [1, 2, 3], "group": ["x", "y", "x"]}
    )

    with pytest.warns(DeprecationWarning):
        categorical = remstats(
            history,
            tie_effects='~ same("group")',
            attr_actors=attributes,
            first=1,
        )
    assert categorical.names == ["baseline", "same_group"]
    assert set(np.unique(np.concatenate(categorical.stats)[:, 1])) == {0.0, 1.0}
    with pytest.warns(DeprecationWarning), pytest.raises(
        TypeError, match="numeric actor attribute"
    ):
        remstats(
            history,
            tie_effects=formula('~ difference("group")'),
            attr_actors=attributes,
            first=1,
        )


def test_remstats_validates_history_slice_and_direction_specific_effects():
    directed = remify(_events())
    with pytest.raises(TypeError, match="EventHistory"):
        remstats(_events(), tie_effects="~ 1")
    with pytest.raises(ValueError, match="1 or a larger"):
        remstats(directed, tie_effects="~ 1", first=0)
    with pytest.raises(ValueError, match="cannot be smaller"):
        remstats(directed, tie_effects="~ 1", first=5, last=3)
    with pytest.raises(ValueError, match="directed events"):
        remstats(directed, tie_effects="~ sp()")

    undirected = remify(_events(), directed=False)
    with pytest.raises(ValueError, match="undirected events"):
        remstats(undirected, tie_effects="~ outdegreeReceiver()")

    with pytest.raises(TypeError, match="EventHistory"):
        tomstats("~ 1", reh=_events())


def test_deprecated_actor_attribute_fallback_validates_schema_and_actor_coverage():
    history = remify(_events())
    attributes = pd.DataFrame(
        {
            "name": [1, 2, 3],
            "time": [0.0, 0.0, 0.0],
            "x1": [10.0, 20.0, 30.0],
        }
    )
    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError, match="not in attr_actors"
    ):
        remstats(
            history,
            tie_effects='~ send(variable="x3")',
            attr_actors=attributes,
        )

    missing_time = attributes.copy()
    missing_time.loc[0, "time"] = np.nan
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Missing"):
        remstats(
            history,
            tie_effects='~ send(variable="x1")',
            attr_actors=missing_time,
        )

    missing_value = attributes.copy()
    missing_value.loc[0, "x1"] = np.nan
    with pytest.warns(DeprecationWarning), pytest.warns(UserWarning, match="Missing"):
        remstats(
            history,
            tie_effects='~ send(variable="x1")',
            attr_actors=missing_value,
        )

    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError, match="Missing actors|missing actor"
    ):
        remstats(
            history,
            tie_effects='~ send(variable="x1")',
            attr_actors=attributes.iloc[:2],
        )

    extra_actor = pd.concat(
        [
            attributes,
            pd.DataFrame({"name": [4], "time": [0.0], "x1": [40.0]}),
        ],
        ignore_index=True,
    )
    with pytest.warns(DeprecationWarning), pytest.warns(UserWarning, match="risk set"):
        remstats(
            history,
            tie_effects='~ send(variable="x1")',
            attr_actors=extra_actor,
        )


def test_matrix_tie_covariate_validates_shape_symmetry_and_missing_values():
    matrix = np.arange(1, 10, dtype=float).reshape(3, 3)
    np.fill_diagonal(matrix, 0.0)
    directed = remify(_events())

    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError, match="actor count"
    ):
        remstats(directed, tie_effects=formula(tie(x=matrix[:2, :])))
    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError, match="actor count"
    ):
        remstats(directed, tie_effects=formula(tie(x=matrix[:, :2])))

    undirected = remify(_events(), directed=False)
    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError, match="symmetric"
    ):
        remstats(undirected, tie_effects=formula(tie(x=matrix)))

    with pytest.warns(DeprecationWarning):
        valid = remstats(directed, tie_effects=formula(tie(x=matrix)), first=1)
    first_riskset = directed.risksets[0]
    expected = matrix[
        first_riskset["sender_id"].to_numpy(dtype=int) - 1,
        first_riskset["receiver_id"].to_numpy(dtype=int) - 1,
    ]
    np.testing.assert_array_equal(valid.stats[0][:, 1], expected)

    matrix[0, 0] = np.nan
    with pytest.warns(DeprecationWarning), pytest.raises(
        ValueError, match="missing values"
    ):
        remstats(directed, tie_effects=formula(tie(x=matrix)))


def test_memory_validation_matches_public_error_contract():
    history = remify(_events())
    for memory in ("window", "decay"):
        with pytest.raises(ValueError, match="memory_value"):
            remstats(history, tie_effects="~ inertia()", memory=memory)
    with pytest.raises(ValueError, match="memory_value"):
        remstats(
            history,
            tie_effects="~ inertia()",
            memory="interval",
            memory_value=1,
        )
    with pytest.raises(ValueError, match="first interval"):
        remstats(
            history,
            tie_effects="~ inertia()",
            memory="interval",
            memory_value=(5, 2),
        )
