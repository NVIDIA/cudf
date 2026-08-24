# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

from polars import polars  # type: ignore[attr-defined]

from cudf_polars.dsl.expressions.boolean import BooleanFunction
from cudf_polars.dsl.expressions.datetime import TemporalFunction
from cudf_polars.dsl.expressions.string import StringFunction
from cudf_polars.dsl.expressions.struct import StructFunction
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_138,
    POLARS_VERSION_LT_141,
    POLARS_VERSION_LT_142,
)


@pytest.fixture(
    params=[BooleanFunction, StringFunction, TemporalFunction, StructFunction]
)
def function(request):
    return request.param


def test_function_name_serialization_all_values(function):
    # Test serialization and deserialization for all values of function.Name
    for name in function.Name:
        serialized_name = pickle.dumps(name)
        deserialized_name = pickle.loads(serialized_name)
        assert deserialized_name is name


def test_function_name_invalid(function):
    # Test invalid attribute name
    with pytest.raises(AttributeError, match="InvalidAttribute"):
        assert function.Name.InvalidAttribute is function.Name.InvalidAttribute


def test_from_polars_all_names(function):
    # Test that all valid names of polars expressions are correctly converted
    polars_function = getattr(polars._expr_nodes, function.__name__)
    polars_names = [name for name in dir(polars_function) if not name.startswith("_")]
    # Check names advertised by polars are the same as we advertise
    polars_names_set = set(polars_names)
    cudf_polars_names_set = set(function.Name.__members__)
    if function == StructFunction:
        cudf_polars_names_set = cudf_polars_names_set - {
            "FieldByIndex",
            "MultipleFields",
        }
    if POLARS_VERSION_LT_138 and function == StringFunction:
        cudf_polars_names_set = cudf_polars_names_set - {"SplitRegex"}
    if POLARS_VERSION_LT_141 and function == BooleanFunction:
        # 'HasNulls' and 'IsEmpty' were added to polars' BooleanFunction in 1.41.
        cudf_polars_names_set = cudf_polars_names_set - {"HasNulls", "IsEmpty"}
    if POLARS_VERSION_LT_142 and function == BooleanFunction:
        # 'IsSorted' was added to polars' BooleanFunction in 1.42.
        cudf_polars_names_set = cudf_polars_names_set - {"IsSorted"}
    assert polars_names_set == cudf_polars_names_set
    names = function.Name
    if function == StructFunction:
        names = set(names) - {
            StructFunction.Name.FieldByIndex,
            StructFunction.Name.MultipleFields,
        }
    if POLARS_VERSION_LT_138 and function == StringFunction:
        names = set(names) - {StringFunction.Name.SplitRegex}
    if POLARS_VERSION_LT_141 and function == BooleanFunction:
        names = set(names) - {
            BooleanFunction.Name.HasNulls,
            BooleanFunction.Name.IsEmpty,
        }
    if POLARS_VERSION_LT_142 and function == BooleanFunction:
        names = set(names) - {BooleanFunction.Name.IsSorted}
    for name in names:
        attr = getattr(polars_function, name.name)
        assert function.Name.from_polars(attr) == name


def test_from_polars_invalid_attribute(function):
    # Test converting from invalid attribute name
    with pytest.raises(ValueError, match=f"{function.__name__} required"):
        function.Name.from_polars("InvalidAttribute")


def test_from_polars_invalid_polars_attribute(function):
    # Test converting from polars function with invalid attribute name
    with pytest.raises(AttributeError, match="InvalidAttribute"):
        function.Name.from_polars(f"{function.__name__}.InvalidAttribute")
