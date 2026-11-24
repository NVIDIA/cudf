# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Static type checks, run by mypy and never by pytest.

The mypy configuration sets ``ignore_missing_imports = true``, so a package
whose types mypy cannot find becomes ``Any`` without any error. Code using it
then passes the mypy hook unchecked. The checks in this file fail the hook
when that happens to a package that cudf-polars relies on.

Two kinds of checks work for this:

- ``assert_type(expr, T)``, where ``T`` does not come from the package under
  test, such as a builtin or a type from another package. If the package
  becomes ``Any``, ``expr`` is ``Any`` and the check fails. Comparing against
  a type from the same package, as in ``assert_type(chunk, TableChunk)``,
  does not work, because both sides become ``Any``.
- A misuse silenced with ``# type: ignore[<code>]``. If the package becomes
  ``Any``, the error disappears, and ``warn_unused_ignores`` fails the hook.
  This form also checks that the types reject a misuse.

pytest does not collect this file, because its name does not start with
``test_``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_type

import pylibcudf as plc

if TYPE_CHECKING:
    from cudf_streaming.table_chunk import TableChunk


def _check_table_chunk(chunk: TableChunk) -> None:
    assert_type(chunk.table_view(), plc.Table)
    assert_type(chunk.is_available(), bool)
    assert_type(chunk.shape, tuple[int, int])
    chunk.no_such_method()  # type: ignore[attr-defined]
