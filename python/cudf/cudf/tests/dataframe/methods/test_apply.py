# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import decimal
import math
import operator

import numpy as np
import pytest
from numba import cuda
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.cuda.cudaimpl import lower as cuda_lower

import cudf
from cudf.core.missing import NA
from cudf.core.udf._ops import (
    comparison_ops,
)
from cudf.core.udf.strings_lowering import (
    cast_string_view_to_managed_udf_string,
)
from cudf.core.udf.strings_typing import (
    StringView,
    managed_udf_string,
    string_view,
)
from cudf.testing import assert_eq


def sv_to_managed_udf_str(sv):
    """
    Cast a string_view object to a managed_udf_string object

    This placeholder function never runs in python
    It exists only for numba to have something to replace
    with the typing and lowering code below

    This is similar conceptually to needing a translation
    engine to emit an expression in target language "B" when
    there is no equivalent in the source language "A" to
    translate from. This function effectively defines the
    expression in language "A" and the associated typing
    and lowering describe the translation process, despite
    the expression having no meaning in language "A"
    """
    pass


@cuda_decl_registry.register_global(sv_to_managed_udf_str)
class StringViewToUDFStringDecl(AbstractTemplate):
    def generic(args, kws):
        if isinstance(args[0], StringView) and len(args) == 1:
            return nb_signature(managed_udf_string, string_view)


@cuda_lower(sv_to_managed_udf_str, string_view)
def sv_to_udf_str_testing_lowering(context, builder, sig, args):
    return cast_string_view_to_managed_udf_string(
        context, builder, sig.args[0], sig.return_type, args[0]
    )


def run_masked_udf_test(func, data, args=(), nullable=True, **kwargs):
    gdf = data
    pdf = data.to_pandas(nullable=nullable)

    expect = pdf.apply(func, args=args, axis=1)
    obtain = gdf.apply(func, args=args, axis=1).to_pandas(nullable=nullable)
    if "check_dtype" in kwargs and not kwargs.get("check_dtype", True):
        expect = expect.astype(obtain.dtype)
    assert_eq(expect, obtain, **kwargs)


@pytest.fixture
def str_udf_data():
    return cudf.DataFrame(
        {
            "str_col": [
                "abc",
                "ABC",
                "AbC",
                "123",
                "123aBc",
                "123@.!",
                "",
                "rapids ai",
                "gpu",
                "True",
                "False",
                "1.234",
                ".123a",
                "0.013",
                "1.0",
                "01",
                "20010101",
                "cudf",
                "cuda",
                "gpu",
                "This Is A Title",
                "This is Not a Title",
                "Neither is This a Title",
                "NoT a TiTlE",
                "123 Title Works",
            ]
        }
    )


@pytest.fixture(params=["a", "2", "gpu", "", " "])
def substr(request):
    return request.param


def test_string_udf_len(str_udf_data):
    def func(row):
        return len(row["str_col"])

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_startswith(str_udf_data, substr):
    def func(row):
        return row["str_col"].startswith(substr)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_endswith(str_udf_data, substr):
    def func(row):
        return row["str_col"].endswith(substr)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_find(str_udf_data, substr):
    def func(row):
        return row["str_col"].find(substr)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_rfind(str_udf_data, substr):
    def func(row):
        return row["str_col"].rfind(substr)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_contains(str_udf_data, substr):
    def func(row):
        return substr in row["str_col"]

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


@pytest.mark.parametrize("other", ["cudf", "123", "", " "])
@pytest.mark.parametrize("cmpop", comparison_ops)
def test_string_udf_cmpops(str_udf_data, other, cmpop):
    def func(row):
        return cmpop(row["str_col"], other)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_isalnum(str_udf_data):
    def func(row):
        return row["str_col"].isalnum()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_isalpha(str_udf_data):
    def func(row):
        return row["str_col"].isalpha()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_isdigit(str_udf_data):
    def func(row):
        return row["str_col"].isdigit()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_isdecimal(str_udf_data):
    def func(row):
        return row["str_col"].isdecimal()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_isupper(str_udf_data):
    def func(row):
        return row["str_col"].isupper()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_islower(str_udf_data):
    def func(row):
        return row["str_col"].islower()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_isspace(str_udf_data):
    def func(row):
        return row["str_col"].isspace()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_istitle(str_udf_data):
    def func(row):
        return row["str_col"].istitle()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_count(str_udf_data, substr):
    def func(row):
        return row["str_col"].count(substr)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


@pytest.mark.xfail(reason="Identity function not supported.")
def test_string_udf_return_string(str_udf_data):
    def func(row):
        return row["str_col"]

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


@pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_strip(str_udf_data, strip_char):
    def func(row):
        return row["str_col"].strip(strip_char)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


@pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_lstrip(str_udf_data, strip_char):
    def func(row):
        return row["str_col"].lstrip(strip_char)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


@pytest.mark.parametrize("strip_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_rstrip(str_udf_data, strip_char):
    def func(row):
        return row["str_col"].rstrip(strip_char)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_upper(str_udf_data):
    def func(row):
        return row["str_col"].upper()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_string_udf_lower(str_udf_data):
    def func(row):
        return row["str_col"].lower()

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


@pytest.mark.parametrize("concat_char", ["1", "a", "12", " ", "", ".", "@"])
def test_string_udf_concat(str_udf_data, concat_char):
    def func(row):
        return row["str_col"] + concat_char

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


@pytest.mark.parametrize("to_replace", ["a", "1", "", "@"])
@pytest.mark.parametrize("replacement", ["a", "1", "", "@"])
def test_string_udf_replace(str_udf_data, to_replace, replacement):
    def func(row):
        return row["str_col"].replace(to_replace, replacement)

    run_masked_udf_test(func, str_udf_data, check_dtype=False)


def test_arith_masked_vs_masked(arithmetic_op):
    # This test should test all the typing
    # and lowering for arithmetic ops between
    # two columns
    def func(row):
        x = row["a"]
        y = row["b"]
        return arithmetic_op(x, y)

    gdf = cudf.DataFrame({"a": [1, None, 3, None], "b": [4, 5, None, None]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_bitwise_masked_vs_masked(bitwise_op):
    # This test should test all the typing
    # and lowering for bitwise ops between
    # two columns
    def func(row):
        x = row["a"]
        y = row["b"]
        return bitwise_op(x, y)

    gdf = cudf.DataFrame(
        {
            "a": [1, 0, 1, 0, 0b1011, 42, None],
            "b": [1, 1, 0, 0, 0b1100, -42, 5],
        }
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_arith_masked_vs_masked_datelike(
    op, datetime_types_as_str, temporal_types_as_str
):
    # Datetime version of the above
    # does not test all dtype combinations for now
    if temporal_types_as_str.startswith("datetime") and op is operator.add:
        # don't try adding datetimes to datetimes.
        pytest.skip("Adding datetime to datetime is not valid")

    def func(row):
        x = row["a"]
        y = row["b"]
        return op(x, y)

    gdf = cudf.DataFrame(
        {
            "a": ["2011-01-01", cudf.NA, "2011-03-01", cudf.NA],
            "b": [4, 5, cudf.NA, cudf.NA],
        }
    )
    gdf["a"] = gdf["a"].astype(datetime_types_as_str)
    gdf["b"] = gdf["b"].astype(temporal_types_as_str)

    pdf = gdf.to_pandas()
    expect = op(pdf["a"], pdf["b"])
    obtain = gdf.apply(func, axis=1)
    assert_eq(expect, obtain, check_dtype=False)
    # TODO: After the following pandas issue is
    # fixed, uncomment the following line and delete
    # through `to_pandas()` statement.
    # https://github.com/pandas-dev/pandas/issues/52411

    # run_masked_udf_test(func, gdf, nullable=False, check_dtype=False)


def test_compare_masked_vs_masked(comparison_op):
    # this test should test all the
    # typing and lowering for comparisons
    # between columns

    def func(row):
        x = row["a"]
        y = row["b"]
        return comparison_op(x, y)

    # we should get:
    # [?, ?, <NA>, <NA>, <NA>]
    gdf = cudf.DataFrame(
        {"a": [1, 0, None, 1, None], "b": [0, 1, 0, None, None]}
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("constant", [1, 1.5, True, False])
def test_arith_masked_vs_constant(arithmetic_op, constant):
    if constant is False and arithmetic_op in {
        operator.mod,
        operator.pow,
        operator.truediv,
        operator.floordiv,
        operator.imod,
        operator.ipow,
        operator.itruediv,
        operator.ifloordiv,
    }:
        # The following tests cases yield undefined behavior:
        # - truediv(x, False) because its dividing by zero
        # - floordiv(x, False) because its dividing by zero
        # - mod(x, False) because its mod by zero,
        # - pow(x, False) because we have an NA in the series and pandas
        #   insists that (NA**0 == 1) where we do not
        pytest.skip(
            f"{constant=} yields undefined behavior for {arithmetic_op=}"
        )

    def func(row):
        x = row["data"]
        return arithmetic_op(x, constant)

    gdf = cudf.DataFrame({"data": [1, 2, cudf.NA]})
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("constant", [1, 1.5, True, False])
@pytest.mark.parametrize("data", [[2, 3, cudf.NA], [1, cudf.NA, 1]])
def test_arith_masked_vs_constant_reflected(
    request, arithmetic_op, constant, data
):
    def func(row):
        x = row["data"]
        return arithmetic_op(constant, x)

    # Just a single column -> result will be all NA
    gdf = cudf.DataFrame({"data": data})

    # cudf differs from pandas for 1**NA
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                constant == 1
                and arithmetic_op in {operator.pow, operator.ipow}
            ),
            reason="https://github.com/NVIDIA/cudf/issues/7478",
        )
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("data", [[1, cudf.NA, 3], [2, 3, cudf.NA]])
def test_arith_masked_vs_null(request, arithmetic_op, data):
    def func(row):
        x = row["data"]
        return arithmetic_op(x, NA)

    gdf = cudf.DataFrame({"data": data})

    # In pandas, 1**NA == 1.
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                (gdf["data"] == 1).any()
                and arithmetic_op in {operator.pow, operator.ipow}
            ),
            reason="https://github.com/NVIDIA/cudf/issues/7478",
        )
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_arith_masked_vs_null_reflected(arithmetic_op):
    def func(row):
        x = row["data"]
        return arithmetic_op(NA, x)

    gdf = cudf.DataFrame({"data": [1, None, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_unary_masked(unary_op):
    # This test should test all the typing
    # and lowering for unary ops

    def func(row):
        x = row["a"]
        return unary_op(x) if x is not NA else NA

    if "log" in unary_op.__name__:
        gdf = cudf.DataFrame({"a": [0.1, 1.0, None, 3.5, 1e8]})
    elif unary_op.__name__ in {"asin", "acos"}:
        gdf = cudf.DataFrame({"a": [0.0, 0.5, None, 1.0]})
    elif unary_op.__name__ in {"atanh"}:
        gdf = cudf.DataFrame({"a": [0.0, -0.5, None, 0.8]})
    elif unary_op.__name__ in {"acosh", "sqrt", "lgamma"}:
        gdf = cudf.DataFrame({"a": [1.0, 2.0, None, 11.0]})
    elif unary_op.__name__ in {"gamma"}:
        gdf = cudf.DataFrame({"a": [0.1, 2, None, 4]})
    elif unary_op.__name__ in {"invert"}:
        gdf = cudf.DataFrame({"a": [-100, 128, None, 0]}, dtype="int64")
    else:
        gdf = cudf.DataFrame({"a": [-125.60, 395.2, 0.0, None]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_masked_is_null_conditional():
    def func(row):
        x = row["a"]
        y = row["b"]
        if x is NA:
            return y
        else:
            return x + y

    gdf = cudf.DataFrame({"a": [1, None, 3, None], "b": [4, 5, None, None]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_contains():
    def func(row):
        x = row["a"]
        return x in [1, 2]

    gdf = cudf.DataFrame({"a": [1, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("op", [operator.add, operator.and_, operator.eq])
def test_apply_mixed_dtypes(numeric_types_as_str, numeric_types_as_str2, op):
    """
    Test that operations can be performed between columns
    of different dtypes and return a column with the correct
    values and nulls
    """

    # First perform the op on two dummy data on host, if numpy can
    # safely type cast, we should expect it to work in udf too.
    try:
        op(
            np.dtype(numeric_types_as_str).type(0),
            np.dtype(numeric_types_as_str2).type(42),
        )
    except TypeError:
        pytest.skip("Operation is unsupported for corresponding dtype.")

    def func(row):
        x = row["a"]
        y = row["b"]
        return op(x, y)

    gdf = cudf.DataFrame({"a": [1.5, None, 3, None], "b": [4, 5, None, None]})
    gdf["a"] = gdf["a"].astype(numeric_types_as_str)
    gdf["b"] = gdf["b"].astype(numeric_types_as_str2)
    run_masked_udf_test(func, gdf, check_dtype=False)


@pytest.mark.parametrize("val", [5, 5.5])
def test_apply_return_literal(val):
    """
    Test unification codepath for scalars and MaskedType
    makes sure that numba knows how to cast a scalar value
    to a MaskedType
    """

    def func(row):
        x = row["a"]
        y = row["b"]
        if x is not NA and x < 2:
            return val
        else:
            return x + y

    gdf = cudf.DataFrame({"a": [1, None, 3, None], "b": [4, 5, None, None]})

    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_return_null():
    """
    Tests casting / unification of Masked and NA
    """

    def func(row):
        x = row["a"]
        if x is NA:
            return NA
        else:
            return x

    gdf = cudf.DataFrame({"a": [1, None, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_return_either_null_or_literal():
    def func(row):
        x = row["a"]
        if x > 5:
            return 2
        else:
            return NA

    gdf = cudf.DataFrame({"a": [1, 3, 6]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_return_literal_only():
    def func(x):
        return 5

    gdf = cudf.DataFrame({"a": [1, None, 3]})
    run_masked_udf_test(func, gdf, check_dtype=False)


def test_apply_everything():
    def func(row):
        w = row["a"]
        x = row["b"]
        y = row["c"]
        z = row["d"]
        if x is NA:
            return w + y - z
        elif ((z > y) is not NA) and z > y:
            return x
        elif ((x + y) is not NA) and x + y == 0:
            return z / x
        elif x + y is NA:
            return 2.5
        elif w > 100:
            return (
                math.sin(x)
                + math.sqrt(y)
                - (-z)
                + math.lgamma(x) * math.fabs(-0.8) / math.radians(3.14)
            )
        else:
            return y > 2

    gdf = cudf.DataFrame(
        {
            "a": [1, 3, 6, 0, None, 5, None, 101],
            "b": [3.0, 2.5, None, 5.0, 1.0, 5.0, 11.0, 1.0],
            "c": [2, 3, 6, 0, None, 5, None, 6],
            "d": [4, None, 6, 0, None, 5, None, 7.5],
        }
    )
    run_masked_udf_test(func, gdf, check_dtype=False)


###


###


def test_masked_udf_lambda_support(binary_op):
    func = lambda row: binary_op(row["a"], row["b"])  # noqa: E731

    data = cudf.DataFrame(
        {"a": [1, cudf.NA, 3, cudf.NA], "b": [1, 2, cudf.NA, cudf.NA]}
    )

    run_masked_udf_test(func, data, check_dtype=False)


def test_masked_udf_nested_function_support(binary_op):
    """
    Nested functions need to be explicitly jitted by the user
    for numba to recognize them. Unfortunately the object
    representing the jitted function can not itself be used in
    pandas udfs.
    """

    def inner(x, y):
        return binary_op(x, y)

    def outer(row):
        x = row["a"]
        y = row["b"]
        return inner(x, y)

    gdf = cudf.DataFrame(
        {"a": [1, cudf.NA, 3, cudf.NA], "b": [1, 2, cudf.NA, cudf.NA]}
    )

    with pytest.raises(ValueError):
        gdf.apply(outer, axis=1)

    pdf = gdf.to_pandas(nullable=True)
    inner_gpu = cuda.jit(device=True)(inner)

    def outer_gpu(row):
        x = row["a"]
        y = row["b"]
        return inner_gpu(x, y)

    got = gdf.apply(outer_gpu, axis=1).to_pandas(nullable=True)
    expect = pdf.apply(outer, axis=1).astype(got.dtype)
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
        {"a": [1, 2, 3], "c": [4, 5, 6], "b": [7, 8, 9]},
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": ["a", "b", "c"]},
    ],
)
def test_masked_udf_subset_selection(data):
    def func(row):
        return row["a"] + row["b"]

    data = cudf.DataFrame(data)
    run_masked_udf_test(func, data, check_dtype=False)


@pytest.mark.parametrize(
    "unsupported_col",
    [
        lambda: cudf.Series(
            [
                decimal.Decimal("1.0"),
                decimal.Decimal("2.0"),
                decimal.Decimal("3.0"),
            ],
            dtype=cudf.Decimal64Dtype(2, 1),
        ),
        lambda: cudf.Series([1, 2, 3], dtype="category"),
        lambda: cudf.interval_range(start=0, end=3),
        lambda: [[1, 2], [3, 4], [5, 6]],
        lambda: [{"a": 1}, {"a": 2}, {"a": 3}],
    ],
)
def test_masked_udf_unsupported_dtype(unsupported_col):
    data = cudf.DataFrame({"unsupported_col": unsupported_col()})

    def func(row):
        return row["unsupported_col"]

    # check that we fail when an unsupported type is used within a function
    with pytest.raises(ValueError):
        data.apply(func, axis=1)

    # also check that a DF containing unsupported dtypes can still run a
    # function that does NOT involve any of the unsupported dtype columns
    data["supported_col"] = 1

    def other_func(row):
        return row["supported_col"]

    expect = cudf.Series(np.ones(len(data)))
    got = data.apply(other_func, axis=1)

    assert_eq(expect, got, check_dtype=False)


# tests for `DataFrame.apply(f, args=(x,y,z))`
# testing the whole space of possibilities is intractable
# these test the most rudimentary guaranteed functionality
@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, cudf.NA, 3]},
        {"a": [0.5, 2.0, cudf.NA, cudf.NA, 5.0]},
        {"a": [True, False, cudf.NA]},
    ],
)
def test_masked_udf_scalar_args_binops(data, binary_op):
    data = cudf.DataFrame(data)

    def func(row, c):
        return binary_op(row["a"], c)

    run_masked_udf_test(func, data, args=(1,), check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, cudf.NA, 3]},
        {"a": [0.5, 2.0, cudf.NA, cudf.NA, 5.0]},
        {"a": [True, False, cudf.NA]},
    ],
)
def test_masked_udf_scalar_args_binops_multiple(data, binary_op):
    data = cudf.DataFrame(data)

    def func(row, c, k):
        x = binary_op(row["a"], c)
        y = binary_op(x, k)
        return y

    run_masked_udf_test(
        func, data, args=(1, 2), check_dtype=False, nullable=True
    )
