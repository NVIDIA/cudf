# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running the narwhals test suite with cudf.pandas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping


TESTS_TO_SKIP: Mapping[str, str] = {
    "tests/expr_and_series/is_in_test.py::test_expr_is_in_with_null_in_other[pandas]": "Fails or passes depending on test ordering under cudf.pandas (is_in with nulls in the other operand).",
    "tests/expr_and_series/arithmetic_test.py::test_floordiv_floats[pandas-1.0-0.1-10.0]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/concat_str_test.py::test_concat_str_nullable_boolean[pandas]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/list/len_test.py::test_pandas_object_series": "cudf.pandas does not raise on object-dtype Series construction (NVIDIA/cudf#18248).",
    "tests/expr_and_series/quantile_test.py::test_quantile_inf[pandas]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/str/replace_test.py::test_str_replace_value_edge_expr_scalar[pandas-backslash_literal]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/str/replace_test.py::test_str_replace_value_edge_series_scalar[pandas-backslash_literal]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/struct_/field_test.py::test_pandas_object_series": "cudf.pandas does not raise on object-dtype Series construction (NVIDIA/cudf#18248).",
    "tests/frame/to_arrow_test.py::test_to_arrow[pandas]": "cudf.pandas cannot convert to a host Arrow object (NVIDIA/cudf#18248).",
    "tests/series_only/from_iterable_test.py::test_series_from_iterable[pandas-polars.series.series.Series-String]": "Flaky under cudf.pandas: a pandas DeprecationWarning (turned into an error by Narwhals' filterwarnings) fires non-deterministically, and Narwhals otherwise marks this param xfail(strict) so an XPASS also fails.",
}


EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/expr_and_series/all_horizontal_test.py::test_all_ignore_nulls[pandas]": "NVIDIA/cudf#19417: Kleene any/all horizontal with nulls.",
    "tests/expr_and_series/all_horizontal_test.py::test_allh_kleene[pandas]": "NVIDIA/cudf#19417: Kleene any/all horizontal with nulls.",
    "tests/expr_and_series/any_horizontal_test.py::test_anyh_kleene[pandas]": "NVIDIA/cudf#19417: Kleene any/all horizontal with nulls.",
    "tests/expr_and_series/cast_test.py::test_pandas_pyarrow_dtypes": "cudf.pandas preserves pyarrow extension dtypes, causing PyArrow dtype handling differences.",
    "tests/expr_and_series/concat_str_test.py::test_concat_str_dask_nullable_boolean[False-expected1-2-bool[pyarrow]]": "_pickle.PicklingError: Can't pickle <class 'pandas.arrays.ArrowStringArray'>: it's not the same object as pandas.arrays.ArrowStringArray",
    "tests/expr_and_series/concat_str_test.py::test_concat_str_dask_nullable_boolean[False-expected1-2-boolean]": "_pickle.PicklingError: Can't pickle <class 'pandas.arrays.ArrowStringArray'>: it's not the same object as pandas.arrays.ArrowStringArray",
    "tests/expr_and_series/concat_str_test.py::test_concat_str_dask_nullable_boolean[True-expected0-2-bool[pyarrow]]": "_pickle.PicklingError: Can't pickle <class 'pandas.arrays.ArrowStringArray'>: it's not the same object as pandas.arrays.ArrowStringArray",
    "tests/expr_and_series/concat_str_test.py::test_concat_str_dask_nullable_boolean[True-expected0-2-boolean]": "_pickle.PicklingError: Can't pickle <class 'pandas.arrays.ArrowStringArray'>: it's not the same object as pandas.arrays.ArrowStringArray",
    "tests/expr_and_series/dt/offset_by_test.py::test_offset_by_date_pandas": "NVIDIA/cudf#19418: dt.offset_by on date columns.",
    "tests/expr_and_series/fill_null_test.py::test_fill_null_pandas_downcast": "cudf.pandas represents nullable bool natively, so fill_null keeps a bool (not object) dtype.",
    "tests/expr_and_series/is_finite_test.py::test_is_finite_column_with_null[pandas-data1]": "AssertionError: Mismatch at index 2, key a: None != False",
    "tests/expr_and_series/is_finite_test.py::test_is_finite_expr[pandas]": "AssertionError: Mismatch at index 0, key a: True != False",
    "tests/expr_and_series/is_finite_test.py::test_is_finite_series[pandas]": "AssertionError: Mismatch at index 0, key a: True != False",
    "tests/expr_and_series/list/get_test.py::test_get_series[pandas-0-expected0]": "cudf.pandas List.get does not raise on out-of-bounds indices.",
    "tests/expr_and_series/lit_test.py::test_null_list_lit[pandas]": "AssertionError: assert Schema({'literal': Unknown}) == {'literal': List(Int64)}",
    "tests/expr_and_series/log_test.py::test_log_dtype_pandas": "cudf.pandas promotes the result of log to float64.",
    "tests/expr_and_series/log_test.py::test_log_dtype_pandas_nullable": "cudf.pandas promotes the result of log to float64.",
    "tests/expr_and_series/over_test.py::test_over_when_then_aggregation_partition_by[pandas-expr0-expected_c0]": "cudf.pandas window aggregation with when/then returns wrong values.",
    "tests/expr_and_series/over_test.py::test_over_when_then_aggregation_partition_by[pandas-expr1-expected_c1]": "cudf.pandas window aggregation with when/then returns wrong values.",
    "tests/expr_and_series/over_test.py::test_over_when_then_aggregation_partition_by[pandas-expr2-expected_c2]": "cudf.pandas window aggregation with when/then returns wrong values.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype12-right_dtype12-result_dtype12]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype13-right_dtype13-result_dtype13]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype15-right_dtype15-result_dtype15]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype21-right_dtype21-result_dtype21]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype22-right_dtype22-result_dtype22]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype23-right_dtype23-result_dtype23]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype24-right_dtype24-result_dtype24]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype25-right_dtype25-result_dtype25]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype29-right_dtype29-result_dtype29]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype30-right_dtype30-result_dtype30]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/pandas_str_dtypes_test.py::test_pandas_str_types[left_dtype31-right_dtype31-result_dtype31]": "cudf.pandas returns a different pandas string dtype than the test expects.",
    "tests/expr_and_series/std_test.py::test_std_constant_group_by[pandas-0]": "AssertionError: Mismatch at index 0, key a: nan != 0.0",
    "tests/expr_and_series/std_test.py::test_std_constant_group_by[pandas-1]": "AssertionError: Mismatch at index 0, key a: nan != 0.0",
    "tests/expr_and_series/std_test.py::test_std_constant_group_by[pandas-2]": "AssertionError: Mismatch at index 0, key a: nan != 0.0",
    "tests/expr_and_series/str/strip_chars_test.py::test_str_strip_chars[pandas--expected2]": "AssertionError: Mismatch at index 1, key a: bar != bar",
    "tests/expr_and_series/str/strip_chars_test.py::test_str_strip_chars_series[pandas--expected2]": "AssertionError: Mismatch at index 1, key a: bar != bar",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_invalid_raises[pandas-trailing]": "Failed: DID NOT RAISE <class 'Exception'>",
    "tests/frame/select_test.py::test_select_boolean_cols": "NVIDIA/cudf#19421: selecting boolean columns raises a length-mismatch error.",
    "tests/frame/select_test.py::test_select_boolean_cols_multi_group_by": "NVIDIA/cudf#19421: selecting boolean columns raises a length-mismatch error.",
    "tests/frame/top_k_test.py::test_top_k[pandas]": "cudf.pandas top_k returns rows in a different order.",
    "tests/frame/with_columns_test.py::test_with_columns_numpy_array_pandas_no_copy": "AssertionError: assert False",
    "tests/series_only/arg_min_max_test.py::test_arg_extremum_preserves_comparison_errors[arg_max]": "Failed: DID NOT RAISE any of (<class 'TypeError'>, <class 'ValueError'>)",
    "tests/series_only/arg_min_max_test.py::test_arg_extremum_preserves_comparison_errors[arg_min]": "Failed: DID NOT RAISE any of (<class 'TypeError'>, <class 'ValueError'>)",
    "tests/series_only/from_iterable_test.py::test_series_from_iterable[polars-cudf.pandas.fast_slow_proxy._FunctionProxy-Float64]": "cudf.pandas Series constructor rejects nullable/pyarrow-backed arrays produced via the fast-slow proxy.",
    "tests/series_only/from_iterable_test.py::test_series_from_iterable[polars-cudf.pandas.fast_slow_proxy._FunctionProxy-Int32]": "cudf.pandas Series constructor rejects nullable/pyarrow-backed arrays produced via the fast-slow proxy.",
    "tests/series_only/from_iterable_test.py::test_series_from_iterable[polars-cudf.pandas.fast_slow_proxy._FunctionProxy-String]": "cudf.pandas Series constructor rejects nullable/pyarrow-backed arrays produced via the fast-slow proxy.",
    "tests/series_only/from_iterable_test.py::test_series_from_iterable[polars-cudf.pandas.fast_slow_proxy._FunctionProxy-no-dtype]": "cudf.pandas Series constructor rejects nullable/pyarrow-backed arrays produced via the fast-slow proxy.",
    "tests/series_only/from_numpy_test.py::test_series_from_numpy_pandas_no_copy": "assert False",
    "tests/testing/assert_frame_equal_test.py::test_self_equal[pandas]": "Narwhals maps cudf.pandas nested dtypes to Object, so assert_series_equal falls back to an undefined element-wise comparison.",
}


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Mark known failing tests."""
    for item in items:
        if (reason := TESTS_TO_SKIP.get(item.nodeid)) is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
        elif (reason := EXPECTED_FAILURES.get(item.nodeid)) is not None:
            item.add_marker(pytest.mark.xfail(reason=reason))
