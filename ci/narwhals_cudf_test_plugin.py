# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running the narwhals test suite with the cuDF constructor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping


TESTS_TO_SKIP: Mapping[str, str] = {
    "tests/expr_and_series/cast_test.py::test_cast_datetime_tz_aware[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/cast_test.py::test_cast_to_float16[cudf]": "cudf does not support the float16 dtype.",
    "tests/expr_and_series/division_by_zero_test.py::test_expr_rfloordiv_by_zero[cudf-0]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/division_by_zero_test.py::test_series_rfloordiv_by_zero[cudf-0]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/dt/convert_time_zone_test.py::test_convert_time_zone[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/dt/convert_time_zone_test.py::test_convert_time_zone_from_none[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/dt/convert_time_zone_test.py::test_convert_time_zone_series[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/dt/replace_time_zone_test.py::test_replace_time_zone[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/dt/replace_time_zone_test.py::test_replace_time_zone_series[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/first_last_test.py::test_first_last_expr_over_order_by[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value0-dtype0]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value1-dtype1]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value2-dtype2]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value3-None]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value4-None]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value5-None]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value6-None]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value7-None]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/lit_test.py::test_nested_structures[cudf-value8-dtype8]": "Nested-structure literals pass under this constructor but Narwhals marks them xfail(strict).",
    "tests/expr_and_series/over_test.py::test_len_over_2369[cudf]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/str/to_time_test.py::test_to_time[cudf]": "cudf does not support a native Time dtype.",
    "tests/expr_and_series/str/to_time_test.py::test_to_time_infer_fmt[cudf-data0-12:34:56]": "cudf does not support a native Time dtype.",
    "tests/expr_and_series/str/to_time_test.py::test_to_time_infer_fmt[cudf-data1-12:34:00]": "cudf does not support a native Time dtype.",
    "tests/expr_and_series/str/to_time_test.py::test_to_time_series[cudf]": "cudf does not support a native Time dtype.",
    "tests/expr_and_series/str/to_time_test.py::test_to_time_series_infer_fmt[cudf-data0-12:34:56]": "cudf does not support a native Time dtype.",
    "tests/expr_and_series/str/to_time_test.py::test_to_time_series_infer_fmt[cudf-data1-12:34:00]": "cudf does not support a native Time dtype.",
    "tests/expr_and_series/when_test.py::test_when_then_otherwise_aggregate_select[cudf-condition0-100-None-expected0]": "Passes under this constructor but Narwhals marks it xfail(strict) citing lack of mixed-type support.",
    "tests/expr_and_series/when_test.py::test_when_then_otherwise_aggregate_select[cudf-condition5-100-None-expected5]": "Passes under this constructor but Narwhals marks it xfail(strict) citing lack of mixed-type support.",
    "tests/expr_and_series/when_test.py::test_when_then_otherwise_aggregate_with_columns[cudf-condition0-100-None-expected0]": "Passes under this constructor but Narwhals marks it xfail(strict) citing lack of mixed-type support.",
    "tests/expr_and_series/when_test.py::test_when_then_otherwise_aggregate_with_columns[cudf-condition5-100-None-expected5]": "Passes under this constructor but Narwhals marks it xfail(strict) citing lack of mixed-type support.",
    "tests/frame/group_by_test.py::test_group_by_depth_1_agg_bool_ops[cudf-not-nullable]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/frame/group_by_test.py::test_group_by_depth_1_agg_bool_ops[cudf-nullable]": "Passes under this constructor but Narwhals marks it xfail(strict).",
}


EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/dtypes/dtypes_test.py::test_cast_decimal_to_native[cudf-10-1]": "cudf does not support casting to this Decimal precision/scale.",
    "tests/dtypes/dtypes_test.py::test_cast_decimal_to_native[cudf-10-8]": "cudf does not support casting to this Decimal precision/scale.",
    "tests/dtypes/dtypes_test.py::test_cast_decimal_to_native[cudf-2-1]": "cudf does not support casting to this Decimal precision/scale.",
    "tests/dtypes/dtypes_test.py::test_cast_decimal_to_native[cudf-None-1]": "cudf does not support casting to this Decimal precision/scale.",
    "tests/dtypes/dtypes_test.py::test_cast_decimal_to_native[cudf-None-20]": "cudf does not support casting to this Decimal precision/scale.",
    "tests/expr_and_series/cast_test.py::test_cast[cudf]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/expr_and_series/cast_test.py::test_cast_series[cudf]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/expr_and_series/corr_test.py::test_corr_expr[cudf-a-a-c-None]": "cudf's cupy-based corr does not support nulls.",
    "tests/expr_and_series/corr_test.py::test_corr_expr_spearman[cudf-a-a-c-None]": "cudf's cupy-based corr does not support nulls.",
    "tests/expr_and_series/corr_test.py::test_corr_pairwise_nulls[cudf]": "cudf's cupy-based corr does not support nulls.",
    "tests/expr_and_series/corr_test.py::test_corr_series[cudf-a-a-c-None]": "cudf's cupy-based corr does not support nulls.",
    "tests/expr_and_series/corr_test.py::test_corr_series_spearman[cudf-a-a-c-None]": "cudf's cupy-based corr does not support nulls.",
    "tests/expr_and_series/cov_test.py::test_cov_expr[cudf]": "cudf's cupy-based cov does not support nulls.",
    "tests/expr_and_series/cov_test.py::test_cov_series[cudf]": "cudf's cupy-based cov does not support nulls.",
    "tests/expr_and_series/first_last_test.py::test_first_last_different_orders[cudf]": "Narwhals' pandas-like backend supports only one order_by in group_by.",
    "tests/expr_and_series/is_close_test.py::test_issue_3474_expr_decimal[cudf]": "cudf does not support casting to this Decimal precision/scale.",
    "tests/expr_and_series/is_close_test.py::test_issue_3474_series_decimal[cudf]": "cudf does not support casting to this Decimal precision/scale.",
    "tests/expr_and_series/list_test.py::test_list_of_lists[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/list_test.py::test_list_positional_exprs[cudf-exprs0]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/list_test.py::test_list_positional_exprs[cudf-exprs1]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/list_test.py::test_list_positional_exprs[cudf-exprs2]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/list_test.py::test_list_single_column[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/list_test.py::test_list_with_expressions[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/rolling_mean_test.py::test_scrambled_groups_over[cudf]": "cudf does not support selecting duplicate column labels.",
    "tests/expr_and_series/str/contains_test.py::test_expr_contains_literal_vs_regex[cudf]": "cudf's str.contains fills nulls with False (pandas-3 str semantics); Narwhals expects None.",
    "tests/expr_and_series/str/contains_test.py::test_expr_contains_str_pattern[cudf-Parrot|dove-True-expected_with_null2-expected_without_null2]": "cudf's str.contains fills nulls with False (pandas-3 str semantics); Narwhals expects None.",
    "tests/expr_and_series/str/contains_test.py::test_expr_contains_str_pattern[cudf-parrot|Dove-False-expected_with_null1-expected_without_null1]": "cudf's str.contains fills nulls with False (pandas-3 str semantics); Narwhals expects None.",
    "tests/expr_and_series/str/contains_test.py::test_series_contains_literal_vs_regex[cudf]": "cudf's str.contains fills nulls with False (pandas-3 str semantics); Narwhals expects None.",
    "tests/expr_and_series/str/contains_test.py::test_series_contains_str_pattern[cudf-Parrot|dove-True-expected_with_null2-expected_without_null2]": "cudf's str.contains fills nulls with False (pandas-3 str semantics); Narwhals expects None.",
    "tests/expr_and_series/str/contains_test.py::test_series_contains_str_pattern[cudf-parrot|Dove-False-expected_with_null1-expected_without_null1]": "cudf's str.contains fills nulls with False (pandas-3 str semantics); Narwhals expects None.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime[cudf]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_infer_fmt[cudf-data0-2020-01-01 12:34:56-2020-01-01T12:34:56.000000000-2020-01-01 12:34:56+00:00]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_infer_fmt[cudf-data1-2020-01-01 12:34:00-2020-01-01T12:34:00.000000000-2020-01-01 12:34:00+00:00]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_infer_fmt[cudf-data2-2024-01-01 12:34:56-2024-01-01T12:34:56.000000000-2024-01-01 12:34:56+00:00]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_series[cudf]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_series_infer_fmt[cudf-data0-2020-01-01 12:34:56-2020-01-01T12:34:56.000000000]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_series_infer_fmt[cudf-data1-2020-01-01 12:34:00-2020-01-01T12:34:00.000000000]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/str/to_datetime_test.py::test_to_datetime_series_infer_fmt[cudf-data2-2024-01-01 12:34:56-2024-01-01T12:34:56.000000000]": "cudf's string-to-datetime cast produces a different string representation than expected.",
    "tests/expr_and_series/struct_test.py::test_struct_mixed_series_and_exprs[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_named_exprs[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_named_with_series[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_positional_and_named[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_positional_exprs[cudf-exprs0]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_positional_exprs[cudf-exprs1]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_positional_exprs[cudf-exprs2]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_positional_exprs[cudf-exprs3]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_with_expressions[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_with_literals[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_with_schema[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/expr_and_series/struct_test.py::test_struct_with_series[cudf]": "Constructing nested list/struct columns triggers a disallowed implicit host PyArrow conversion in cudf.",
    "tests/frame/columns_test.py::test_iter_columns[cudf]": "cudf does not support iterating over a Series/DataFrame/Index.",
    "tests/frame/getitem_test.py::test_getitem_boolean_columns[cudf]": "cudf does not support iterating over a Series/DataFrame/Index.",
    "tests/frame/with_columns_test.py::test_with_columns_dtypes_single_row[cudf]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_categorical[cudf]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_enum[cudf-selector0]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_enum[cudf-selector1]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_enum[cudf-selector2]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_enum_distinct_from_categorical[cudf-selector0-expected0]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_enum_distinct_from_categorical[cudf-selector1-expected1]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_enum_distinct_from_categorical[cudf-selector2-expected2]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/selectors_test.py::test_enum_distinct_from_categorical[cudf-selector3-expected3]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/series_only/cast_test.py::test_cast_to_enum_vmain[cudf]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/testing/assert_frame_equal_test.py::test_check_row_order[cudf-False]": "cudf does not support iterating over a Series/DataFrame/Index.",
    "tests/testing/assert_frame_equal_test.py::test_self_equal[cudf]": "cudf cannot round-trip this categorical dtype through Arrow.",
    "tests/testing/assert_series_equal_test.py::test_categorical_as_str[cudf-False-context1]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
    "tests/testing/assert_series_equal_test.py::test_categorical_as_str[cudf-True-context0]": "Narwhals hashes cudf's CategoricalDtype, which is unhashable.",
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
