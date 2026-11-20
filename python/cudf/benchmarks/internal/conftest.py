# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Defines pytest fixtures for internal benchmarks."""

from config import NUM_ROWS_FIXTURES, cudf
from utils import (
    OrderedSet,
    collapse_fixtures,
    column_generators,
    make_fixture,
)

fixtures = OrderedSet()
for dtype, column_generator in column_generators.items():
    for nr in NUM_ROWS_FIXTURES:

        def column_nulls_false(request, nr=nr):
            return cudf.core.column.as_column(column_generator(nr))

        make_fixture(
            f"column_dtype_{dtype}_nulls_false_rows_{nr}",
            column_nulls_false,
            globals(),
            fixtures,
        )

        def column_nulls_true(request, nr=nr):
            c = cudf.core.column.as_column(column_generator(nr))
            c[::2] = None
            return c

        make_fixture(
            f"column_dtype_{dtype}_nulls_true_rows_{nr}",
            column_nulls_true,
            globals(),
            fixtures,
        )

num_new_fixtures = len(fixtures)

# Keep trying to merge existing fixtures until no new fixtures are added.
while num_new_fixtures > 0:
    num_fixtures = len(fixtures)

    # Note: If we start also introducing unions across dtypes, most likely
    # those will take the form `*int_and_float*` or similar since we won't want
    # to union _all_ dtypes. In that case, the regexes will need to use
    # suitable lookaheads etc to avoid infinite loops here.
    for pat, repl in [
        ("_nulls_(true|false)", ""),
        (r"_rows_\d+", ""),
    ]:
        collapse_fixtures(fixtures, pat, repl, globals())

    num_new_fixtures = len(fixtures) - num_fixtures
