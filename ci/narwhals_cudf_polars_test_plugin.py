# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running the narwhals test suite with the cuDF Polars constructor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping


TESTS_TO_SKIP: Mapping[str, str] = {
    "tests/expr_and_series/cum_sum_test.py::test_cum_sum_does_not_overflow[polars[lazy]-False-Int32-2147483647]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/cum_sum_test.py::test_cum_sum_does_not_overflow[polars[lazy]-False-UInt32-4294967295]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/cum_sum_test.py::test_cum_sum_does_not_overflow[polars[lazy]-True-Int32-2147483647]": "Passes under this constructor but Narwhals marks it xfail(strict).",
    "tests/expr_and_series/cum_sum_test.py::test_cum_sum_does_not_overflow[polars[lazy]-True-UInt32-4294967295]": "Passes under this constructor but Narwhals marks it xfail(strict).",
}


EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/expr_and_series/dt/convert_time_zone_test.py::test_convert_time_zone[polars[lazy]]": "https://github.com/NVIDIA/cudf/issues/23935",
    "tests/expr_and_series/dt/convert_time_zone_test.py::test_convert_time_zone_from_none[polars[lazy]]": "https://github.com/NVIDIA/cudf/issues/23935",
    "tests/expr_and_series/dt/replace_time_zone_test.py::test_replace_time_zone[polars[lazy]]": "https://github.com/NVIDIA/cudf/issues/23935",
    "tests/expr_and_series/fill_null_test.py::test_fill_null_scalar_receiver[polars[lazy]]": "https://github.com/NVIDIA/cudf/issues/24864",
    "tests/expr_and_series/is_in_test.py::test_expr_is_in_with_only_null_in_other[polars[lazy]]": "https://github.com/NVIDIA/cudf/issues/24863",
    "tests/expr_and_series/str/zfill_test.py::test_str_zfill[polars[lazy]-non_ascii]": "cudf-polars zfill only supports ascii strings.",
    "tests/expr_and_series/str/zfill_test.py::test_str_zfill[polars[lazy]-non_ascii_plus_w3]": "cudf-polars zfill only supports ascii strings.",
    "tests/expr_and_series/str/zfill_test.py::test_str_zfill[polars[lazy]-non_ascii_plus_w4]": "cudf-polars zfill only supports ascii strings.",
    "tests/expr_and_series/str/zfill_test.py::test_str_zfill[polars[lazy]-non_ascii_sign_w5]": "cudf-polars zfill only supports ascii strings.",
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
