#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import numbers
from pathlib import Path

import duckdb

QUERIES = (
    "q01",
    "q02",
    "q03",
    "q04",
    "q05",
    "q06",
    "q07",
    "q08",
    "q09",
    "q10",
    "q11",
    "q12",
    "q13",
    "q14",
    "q15",
    "q16",
    "q17",
    "q18",
    "q19",
    "q20",
    "q21",
    "q22",
)
EXPECTED_NAMES = {
    "q18": [
        "c_name",
        "c_custkey",
        "o_orderkey",
        "o_orderdate",
        "o_totalprice",
        "sum(l_quantity)",
    ]
}


def values_equal(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, numbers.Number) and isinstance(
        expected, numbers.Number
    ):
        return math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=0.01
        )
    return actual == expected


def validate_query(query_name, sql_dir, output_dir, scale_factor=0.01):
    connection = duckdb.connect()
    for path in (output_dir / query_name / "input").glob("*.parquet"):
        table_name = path.stem.replace('"', '""')
        parquet_path = str(path).replace("'", "''")
        connection.execute(
            f'CREATE VIEW "{table_name}" AS '
            f"SELECT * FROM read_parquet('{parquet_path}')"
        )

    parameters = (
        {"scale_factor": scale_factor} if query_name == "q11" else None
    )
    expected = connection.execute(
        (sql_dir / f"{query_name}.sql").read_text(), parameters
    )
    expected_names = [column[0] for column in expected.description]
    expected_rows = expected.fetchall()

    result_path = output_dir / query_name / "results" / f"{query_name}.parquet"
    actual = connection.execute(
        "SELECT * FROM read_parquet(?)", [str(result_path)]
    )
    actual_names = [column[0] for column in actual.description]
    actual_rows = actual.fetchall()

    expected_names = EXPECTED_NAMES.get(query_name, expected_names)
    if actual_names != expected_names:
        return f"column names differ: {actual_names} != {expected_names}"
    if len(actual_rows) != len(expected_rows):
        return f"row counts differ: {len(actual_rows)} != {len(expected_rows)}"

    for row_index, (actual_row, expected_row) in enumerate(
        zip(actual_rows, expected_rows, strict=True)
    ):
        for column_name, actual_value, expected_value in zip(
            actual_names, actual_row, expected_row, strict=True
        ):
            if not values_equal(actual_value, expected_value):
                return (
                    f"row {row_index}, column {column_name} differs: "
                    f"{actual_value!r} != {expected_value!r}"
                )
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate NDS-H benchmark Parquet results against DuckDB"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sql-dir", type=Path, required=True)
    parser.add_argument("--scale-factor", type=float, default=0.01)
    args = parser.parse_args()

    failed = False
    for query_name in QUERIES:
        error = validate_query(
            query_name, args.sql_dir, args.output_dir, args.scale_factor
        )
        if error is None:
            print(f"{query_name}: PASSED")
        else:
            failed = True
            print(f"{query_name}: FAILED: {error}")

    raise SystemExit(failed)


if __name__ == "__main__":
    main()
