# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin for running the narwhals test suite with the cuDF Polars constructor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping


TESTS_TO_SKIP: Mapping[str, str] = {}


EXPECTED_FAILURES: Mapping[str, str] = {
    "tests/expr_and_series/lit_test.py::test_nested_structures[polars[lazy]-value0-dtype0]": "cudf-polars mishandles list/tuple literals without nested lists.",
    "tests/expr_and_series/lit_test.py::test_nested_structures[polars[lazy]-value1-dtype1]": "cudf-polars mishandles list/tuple literals without nested lists.",
    "tests/expr_and_series/lit_test.py::test_nested_structures[polars[lazy]-value3-None]": "cudf-polars mishandles list/tuple literals without nested lists.",
    "tests/expr_and_series/lit_test.py::test_nested_structures[polars[lazy]-value4-None]": "cudf-polars mishandles list/tuple literals without nested lists.",
    "tests/expr_and_series/lit_test.py::test_nested_structures[polars[lazy]-value6-None]": "cudf-polars mishandles list/tuple literals without nested lists.",
    "tests/expr_and_series/lit_test.py::test_nested_structures[polars[lazy]-value7-None]": "cudf-polars mishandles list/tuple literals without nested lists.",
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
