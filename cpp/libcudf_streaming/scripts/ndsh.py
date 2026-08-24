#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "duckdb",
#     "numpy",
#     "pyarrow",
#     "tpchgen-cli",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""
Validation script for NDSH benchmarks.

This script validates the correctness of NDSH benchmark outputs by:
1. Running SQL queries via DuckDB to generate expected results
2. Running the C++ benchmark binaries
3. Comparing the benchmark output against the DuckDB result

Usage:
    # Run benchmarks and generate expected results
    python validate_ndsh.py run \\
        --benchmark-dir /path/to/build/benchmarks/ndsh \\
        --sql-dir /path/to/sql/queries \\
        --input-dir /raid/rapidsmpf/data/tpch/scale-1.0 \\
        --output-dir /tmp/validation

    # Validate results against expected
    python validate_ndsh.py validate \\
        --results-path /tmp/validation/output \\
        --expected-path /tmp/validation/expected
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TPCH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]

# simple precaution to ensure that the SQL hasn't changed
# from what we expect.
QUERY_HASHES = {
    "q01": "cccf0c3d9302ee4b56a2bac2f2aa05ab",
    "q03": "1fdb8b3d8044f7d72c7bb021d025ea70",
    "q04": "ed586bdb2b1495d3b1b19d1217dd6750",
    "q09": "de7c61303a841c512857ee50412b54b9",
    "q17": "7be3e180995f841ece86ffb2de9cf2b0",
    "q18": "38b13cbbaeea09c224cc53b532507f76",
    "q21": "61bcc0a1239c0feefc54b683027df014",
}


def discover_benchmarks(
    benchmark_dir: Path, sql_dir: Path
) -> list[tuple[str, Path, Path]]:
    """
    Discover benchmark binaries and their corresponding SQL files.

    Parameters
    ----------
    benchmark_dir
        Directory containing benchmark binaries (q04, q09, etc.)
    sql_dir
        Directory containing SQL query files (q04.sql, q09.sql, etc.)

    Returns
    -------
    List of tuples with the following elements:

    - query_name
    - binary_path
    - sql_path

    These are sorted by query_name.
    """
    benchmarks = []
    pattern = re.compile(r"^q(\d+)$")

    for binary in benchmark_dir.iterdir():
        if not binary.is_file():
            continue
        match = pattern.match(binary.name)
        if not match:
            continue

        query_name = binary.name
        sql_path = sql_dir / f"{query_name}.sql"

        if sql_path.exists():
            benchmarks.append((query_name, binary, sql_path))
        else:
            print(f"Warning: No SQL file found for {query_name} at {sql_path}")

    return sorted(benchmarks)


def discover_parquet_files(directory: Path) -> dict[str, Path]:
    """
    Discover parquet files matching the qDD.parquet pattern.

    Parameters
    ----------
    directory
        Directory containing parquet files (q03.parquet, q09.parquet, etc.)

    Returns
    -------
    Dictionary mapping query_name (e.g., 'q03') to file path.
    """
    pattern = re.compile(r"^q_?(\d+)\.parquet$")
    files = {}

    for file in directory.iterdir():
        if not file.is_file():
            continue
        match = pattern.match(file.name)
        if match:
            query_name = f"q{match.group(1)}"
            files[query_name] = file

    return files


def generate_expected(
    sql_path: Path, input_dir: Path, output_path: Path
) -> None:
    """
    Generate expected results by running a SQL query via DuckDB.

    Parameters
    ----------
    sql_path
        Path to the SQL query file
    input_dir:
        Directory containing TPC-H parquet files
    output_path
        Path to write the expected parquet result
    """
    con = duckdb.connect()

    # Register TPC-H tables as views from parquet files
    for table in TPCH_TABLES:
        # Try both single file and directory patterns
        single_file = input_dir / f"{table}.parquet"
        directory = input_dir / table

        if single_file.exists():
            parquet_path = single_file
        elif directory.exists() and directory.is_dir():
            parquet_path = directory / "*.parquet"
        else:
            raise FileNotFoundError(f"Table {table} not found in {input_dir}")

        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')"
        )

    # Read and execute the query
    query = sql_path.read_text()
    query_hash = hashlib.md5(query.encode()).hexdigest()
    query_id = sql_path.stem

    if query_id not in QUERY_HASHES:
        raise ValueError(
            f"Query {query_id} from file {sql_path} not found in QUERY_HASHES. Please update scripts/ndsh.py with the new hash."
        )
    if query_hash != QUERY_HASHES[query_id]:
        raise ValueError(
            f"Query {query_id} from file {sql_path} has changed. Please update scripts/ndsh.py with the new hash using 'md5sum {sql_path}'."
        )

    result = con.sql(query).arrow().read_all()

    # Write result to parquet
    pq.write_table(result, output_path)
    print(f"  Generated expected: {output_path} ({result.num_rows} rows)")


def generate_data(input_dir: Path) -> None:
    """
    Generate data for the benchmarks.

    This uses tpchgen-cli to generate the data and casts some columns
    to the types expected by the benchmarks.
    """
    print(f"Generating data for {input_dir}...")
    subprocess.check_output(
        [
            "tpchgen-cli",
            "--scale-factor",
            "1",
            "--format",
            "parquet",
            "--output-dir",
            str(input_dir),
        ]
    )

    # Some of our queries are written expecting float (Double)
    casts = {
        ("customer", "c_nationkey"): pa.int32(),
        ("customer", "c_acctbal"): pa.float64(),
        ("lineitem", "l_linenumber"): pa.int64(),
        ("lineitem", "l_quantity"): pa.float64(),
        ("lineitem", "l_extendedprice"): pa.float64(),
        ("lineitem", "l_discount"): pa.float64(),
        ("lineitem", "l_tax"): pa.float64(),
        ("lineitem", "l_shipdate"): pa.date32(),
        ("lineitem", "l_commitdate"): pa.date32(),
        ("lineitem", "l_receiptdate"): pa.date32(),
        ("nation", "n_nationkey"): pa.int32(),
        ("nation", "n_regionkey"): pa.int32(),
        ("orders", "o_totalprice"): pa.float64(),
        ("orders", "o_orderdate"): pa.date32(),
        ("part", "p_retailprice"): pa.float64(),
        ("partsupp", "ps_availqty"): pa.int64(),
        ("partsupp", "ps_supplycost"): pa.float64(),
        ("region", "r_regionkey"): pa.int32(),
        ("supplier", "s_nationkey"): pa.int32(),
        ("supplier", "s_acctbal"): pa.float64(),
    }

    for table_name in TPCH_TABLES:
        file = (input_dir / table_name).with_suffix(".parquet")
        table = pq.read_table(file)
        schema = table.schema
        for i, field in enumerate(schema):
            if cast := casts.get((table_name, field.name)):
                schema = schema.set(i, field.with_type(cast))

        pq.write_table(table.cast(schema), file)


def run_benchmark(
    binary_path: Path,
    input_dir: Path,
    output_path: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a benchmark binary.

    Parameters
    ----------
    binary_path
        Path to the benchmark binary
    input_dir
        Directory containing TPC-H parquet files
    output_path
        Path for benchmark output
    extra_args
        Additional arguments to pass to the benchmark

    Returns
    -------
    CompletedProcess result
    """
    cmd = [
        "mpirun",
        "-np",
        "1",
        "--allow-run-as-root",
        str(binary_path),
        "--input-directory",
        str(input_dir),
        "--output-file",
        str(output_path),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"  Running: {' '.join(cmd)}")

    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )


def _types_compatible(
    o_type: pa.DataType,
    e_type: pa.DataType,
    *,
    ignore_timezone: bool = False,
    ignore_string_type: bool = False,
    ignore_integer_sign: bool = False,
    ignore_integer_size: bool = False,
    ignore_decimal_int: bool = False,
) -> bool:
    """
    Check if two Arrow types are compatible given the ignore flags.

    Returns True if the types should be considered equal.
    """
    if o_type.equals(e_type):
        return True

    # Ignore differences in timezone and precision for timestamps
    if (
        ignore_timezone
        and pa.types.is_timestamp(o_type)
        and pa.types.is_timestamp(e_type)
    ):
        return True

    # Ignore large_string vs string differences
    if ignore_string_type:
        string_types = {pa.string(), pa.large_string()}
        if o_type in string_types and e_type in string_types:
            return True

    # Check integer compatibility
    if pa.types.is_integer(o_type) and pa.types.is_integer(e_type):
        o_signed = pa.types.is_signed_integer(o_type)
        e_signed = pa.types.is_signed_integer(e_type)
        o_width = o_type.bit_width
        e_width = e_type.bit_width

        sign_matches = o_signed == e_signed or ignore_integer_sign
        size_matches = o_width == e_width or ignore_integer_size

        if sign_matches and size_matches:
            return True

    # Ignore decimal vs integer differences
    if ignore_decimal_int:
        o_is_numeric = pa.types.is_integer(o_type) or pa.types.is_decimal(
            o_type
        )
        e_is_numeric = pa.types.is_integer(e_type) or pa.types.is_decimal(
            e_type
        )
        if o_is_numeric and e_is_numeric:
            return True

    return False


def compare_parquet(
    output_path: Path,
    expected_path: Path,
    decimal: int = 2,
    *,
    ignore_timezone: bool = False,
    ignore_string_type: bool = False,
    ignore_integer_sign: bool = False,
    ignore_integer_size: bool = False,
    ignore_decimal_int: bool = False,
) -> tuple[bool, str | None]:
    """
    Compare two parquet files for exact equality.

    Parameters
    ----------
    output_path
        Path to the benchmark output parquet
    expected_path
        Path to the expected parquet
    decimal
        Number of decimal places to compare for floating point values
    ignore_timezone
        Ignore differences in timezone and precision for timestamp types
    ignore_string_type
        Ignore differences between string and large_string types.
        Note that the values will still be compared.
    ignore_integer_sign
        Ignore differences between signed and unsigned integer types
        Note that the values will still be compared.
    ignore_integer_size
        Ignore differences in integer bit width (e.g., int32 vs int64)
        Note that the values will still be compared.
    ignore_decimal_int
        Ignore differences between decimal and integer types
        Note that the values will still be compared.

    Returns
    -------
    Tuple of boolean indicating success and list of error messages. A non-empty list indicates failure.
    """
    try:
        output = pq.read_table(output_path)
        expected = pq.read_table(expected_path)
    except Exception as e:
        return False, f"Failed to read parquet files: {e}"

    # Check the schema and data by validating...
    # 1. names...
    if output.schema.names != expected.schema.names:
        return (
            False,
            f"Schema name mismatch: {output.schema.names} != {expected.schema.names}",
        )

    # 2. types...
    errors = []
    for name in output.schema.names:
        o_field = output.schema.field(name)
        e_field = expected.schema.field(name)
        # We only care about the type, not the metadata or nullability
        if not _types_compatible(
            o_field.type,
            e_field.type,
            ignore_timezone=ignore_timezone,
            ignore_string_type=ignore_string_type,
            ignore_integer_sign=ignore_integer_sign,
            ignore_integer_size=ignore_integer_size,
            ignore_decimal_int=ignore_decimal_int,
        ):
            errors.append(f"\t{name}: {o_field.type} != {e_field.type}")
    if errors:
        return False, "\n".join(
            ["Field type mismatch (output != expected)", *errors]
        )

    # 3. row count...
    if output.num_rows != expected.num_rows:
        return False, (
            f"Row count mismatch: output={output.num_rows}, expected={expected.num_rows}"
        )

    # 4. and values.
    for name, out_col, expected_col in zip(
        output.column_names, output.columns, expected.columns, strict=False
    ):
        if pa.types.is_floating(out_col.type):
            # We don't promise exact equality
            try:
                np.testing.assert_array_almost_equal(
                    out_col.to_numpy(),
                    expected_col.to_numpy(),
                    decimal=decimal,
                )
            except AssertionError as e:
                errors.append(f"{name} differs. {e}")
        else:
            try:
                np.testing.assert_array_equal(
                    out_col.to_numpy(), expected_col.to_numpy()
                )
            except AssertionError as e:
                errors.append(f"{name} differs. {e}")

    if errors:
        return False, "\n".join(errors)

    return True, None


def run_single_benchmark(
    query_name: str,
    binary_path: Path,
    sql_path: Path,
    input_dir: Path,
    output_dir: Path,
    expected_dir: Path,
    extra_args: list[str] | None = None,
    *,
    reuse_expected: bool = False,
    reuse_output: bool = False,
) -> bool:
    """
    Run a single benchmark and generate expected results.

    Parameters
    ----------
    query_name
        Name of the query to run (e.g., 'q03')
    binary_path
        Path to the benchmark binary
    sql_path
        Path to the SQL query file
    input_dir
        Directory containing TPC-H parquet files
    output_dir
        Directory for benchmark output
    expected_dir
        Directory for expected results
    extra_args
        Additional arguments to pass to the benchmark
    reuse_expected
        Skip generating expected results if the expected file already exists
    reuse_output
        Skip running the benchmark if the output file already exists

    Returns
    -------
    True if both operations succeed, False otherwise.
    """
    print(f"\nRunning {query_name}...")

    expected_path = expected_dir / f"{query_name}.parquet"
    benchmark_output = output_dir / f"{query_name}.parquet"

    # Generate expected
    if reuse_expected and expected_path.exists():
        print(f"  Reusing existing expected: {expected_path}")
    else:
        print("  Generating expected via DuckDB...")
        try:
            generate_expected(sql_path, input_dir, expected_path)
        except Exception as e:
            print(f"  FAILED: Expected generation error: {e}")
            return False

    # Run benchmark
    if reuse_output and benchmark_output.exists():
        print(f"  Reusing existing output: {benchmark_output}")
    else:
        result = run_benchmark(
            binary_path, input_dir, benchmark_output, extra_args
        )

        if result.returncode != 0:
            print(f"  FAILED: Benchmark exited with code {result.returncode}")
            print(
                f"  stdout: {result.stdout[:1000] if result.stdout else '(empty)'}"
            )
            print(
                f"  stderr: {result.stderr[:1000] if result.stderr else '(empty)'}"
            )
            return False

    if not benchmark_output.exists():
        print(
            f"  FAILED: Benchmark did not produce output file: {benchmark_output}"
        )
        return False

    print("  SUCCESS")
    return True


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    # Validate paths
    if not args.benchmark_dir.exists():
        print(
            f"Error: Benchmark directory does not exist: {args.benchmark_dir}"
        )
        return 1

    if not args.sql_dir.exists():
        print(f"Error: SQL directory does not exist: {args.sql_dir}")
        return 1

    if args.generate_data:
        generate_data(args.input_dir)

    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for output and expected
    benchmark_output_dir = output_dir / "output"
    expected_output_dir = output_dir / "expected"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    expected_output_dir.mkdir(parents=True, exist_ok=True)

    # Parse extra benchmark args
    extra_args = args.benchmark_args.split() if args.benchmark_args else None

    # Discover benchmarks
    benchmarks = discover_benchmarks(args.benchmark_dir, args.sql_dir)

    if not benchmarks:
        print("No benchmarks found!")
        return 1

    # Filter to specific queries if requested
    if args.queries:
        benchmarks = [
            (name, binary, sql)
            for name, binary, sql in benchmarks
            if int(name.lstrip("q")) in args.queries
        ]
        if not benchmarks:
            print(f"No matching benchmarks found for queries: {args.queries}")
            return 1

    print(f"Found {len(benchmarks)} benchmark(s) to run:")
    for name, binary, sql in benchmarks:
        print(f"  {name}: {binary} + {sql}")

    # Run benchmarks
    results = {}
    for query_name, binary_path, sql_path in benchmarks:
        passed = run_single_benchmark(
            query_name,
            binary_path,
            sql_path,
            args.input_dir,
            benchmark_output_dir,
            expected_output_dir,
            extra_args,
            reuse_expected=args.reuse_expected,
            reuse_output=args.reuse_output,
        )
        results[query_name] = passed

    # Summary
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    failed = len(results) - passed

    print(f"Total: {passed} succeeded, {failed} failed")
    print(f"\nOutput directory: {output_dir}")
    print(f"  Results: {benchmark_output_dir}")
    print(f"  Expected: {expected_output_dir}")

    return int(failed > 0)


def cmd_run_and_validate(args: argparse.Namespace) -> int:
    """Execute the 'run-and-validate' subcommand."""
    # First run the benchmarks
    run_result = cmd_run(args)
    if run_result != 0:
        print("\nRun phase failed, skipping validation.")
        return run_result

    # Set up paths for validation based on run output
    args.results_path = args.output_dir / "output"
    args.expected_path = args.output_dir / "expected"

    return cmd_validate(args)


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute the 'validate' subcommand."""
    if not args.results_path.exists():
        print(f"Error: Results directory does not exist: {args.results_path}")
        return 1

    if not args.expected_path.exists():
        print(
            f"Error: Expected directory does not exist: {args.expected_path}"
        )
        return 1

    # Discover parquet files in both directories
    # But we treat *results* as the source of truth. If we have a result
    # but not an expected we error; if we have an expected but not a result,
    # that's fine.
    results_files = discover_parquet_files(args.results_path)

    if not results_files:
        print(
            f"No qDD.parquet files found in results directory: {args.results_path}"
        )
        return 1

    # Filter to specific queries if requested
    if args.queries:
        results_files = {
            name: path
            for name, path in results_files.items()
            if int(name.lstrip("q")) in args.queries
        }
        if not results_files:
            print(
                f"No matching result files found for queries: {args.queries}"
            )
            return 1

    print(f"\nValidating {len(results_files)} query(ies):")

    # Validate each matching pair
    results = {}
    for query_name in results_files.keys():
        print(f"\nValidating {query_name}...")
        result_path = results_files[query_name]
        expected_path = args.expected_path / f"{query_name}.parquet"

        if not expected_path.exists():
            print(f"  FAILED: Expected file does not exist: {expected_path}")
            results[query_name] = False
            continue

        is_equal, message = compare_parquet(
            result_path,
            expected_path,
            decimal=args.decimal,
            ignore_timezone=args.ignore_timezone,
            ignore_string_type=args.ignore_string_type,
            ignore_integer_sign=args.ignore_integer_sign,
            ignore_integer_size=args.ignore_integer_size,
            ignore_decimal_int=args.ignore_decimal_int,
        )

        if is_equal:
            print("  PASSED")
            results[query_name] = True
        else:
            print(f"  FAILED:\n{message}")
            results[query_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    failed = len(results) - passed

    for query_name, result in sorted(results.items()):
        status = "PASSED" if result else "FAILED"
        print(f"  {query_name}: {status}")

    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed")

    return int(failed > 0)


def query_type(query: str) -> list[int]:
    if query == "all":
        return list(range(1, 23))
    else:
        return [int(q) for q in query.split(",")]


def main():
    """Run the NDSH validation tool."""
    parser = argparse.ArgumentParser(
        description="NDSH benchmark runner and validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Parent parser for run-related arguments
    run_parent = argparse.ArgumentParser(add_help=False)
    run_parent.add_argument(
        "--benchmark-dir",
        type=Path,
        help="Directory containing benchmark binaries (q04, q09, etc.)",
        default=Path(__file__).parent.parent.parent.joinpath(
            "cpp/build/benchmarks/ndsh"
        ),
    )
    run_parent.add_argument(
        "--sql-dir",
        type=Path,
        help="Directory containing SQL query files (q04.sql, q09.sql, etc.)",
        default=Path(__file__).parent.parent.parent.joinpath(
            "cpp/benchmarks/streaming/ndsh/sql"
        ),
    )
    run_parent.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing TPC-H input parquet files",
    )
    run_parent.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files",
    )
    run_parent.add_argument(
        "-q",
        "--queries",
        help="Comma-separated list of SQL query numbers to run or the string 'all'",
        type=query_type,
        default="all",
    )
    run_parent.add_argument(
        "--benchmark-args",
        type=str,
        default="",
        help="Additional arguments to pass to benchmark binaries (space-separated)",
    )
    run_parent.add_argument(
        "--reuse-expected",
        action="store_true",
        help="Skip generating expected results if the expected file already exists",
    )
    run_parent.add_argument(
        "--reuse-output",
        action="store_true",
        help="Skip running the benchmark if the output file already exists",
    )
    run_parent.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate data for the benchmarks",
    )

    # Parent parser for validation comparison options (not the paths)
    validate_options_parent = argparse.ArgumentParser(add_help=False)
    validate_options_parent.add_argument(
        "-d",
        "--decimal",
        type=int,
        default=2,
        help="Number of decimal places to compare for floating point values (default: 2)",
    )
    validate_options_parent.add_argument(
        "--ignore-timezone",
        action="store_true",
        help="Ignore differences in timezone and precision for timestamp types",
    )
    validate_options_parent.add_argument(
        "--ignore-string-type",
        action="store_true",
        help="Ignore differences between string and large_string types",
    )
    validate_options_parent.add_argument(
        "--ignore-integer-sign",
        action="store_true",
        help="Ignore differences between signed and unsigned integer types",
    )
    validate_options_parent.add_argument(
        "--ignore-integer-size",
        action="store_true",
        help="Ignore differences in integer bit width (e.g., int32 vs int64)",
    )
    validate_options_parent.add_argument(
        "--ignore-decimal-int",
        action="store_true",
        help="Ignore differences between decimal and integer types",
    )

    # 'run' subcommand - inherits from run_parent
    subparsers.add_parser(
        "run",
        parents=[run_parent],
        help="Run benchmarks and generate expected results",
        description="Run C++ benchmark binaries and generate expected results via DuckDB.",
    )

    # 'validate' subcommand - inherits comparison options, adds its own paths
    validate_parser = subparsers.add_parser(
        "validate",
        parents=[validate_options_parent],
        help="Compare results against expected",
        description="Validate benchmark results by comparing parquet files against expected results.",
    )
    validate_parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Directory containing benchmark result parquet files (qDD.parquet)",
    )
    validate_parser.add_argument(
        "--expected-path",
        type=Path,
        required=True,
        help="Directory containing expected parquet files (qDD.parquet)",
    )
    validate_parser.add_argument(
        "-q",
        "--queries",
        help="Comma-separated list of SQL query numbers to validate or the string 'all'",
        type=query_type,
        default="all",
    )

    # 'run-and-validate' subcommand - inherits from BOTH parents
    subparsers.add_parser(
        "run-and-validate",
        parents=[run_parent, validate_options_parent],
        help="Run benchmarks and validate results in one step",
        description="Run C++ benchmark binaries, generate expected results via DuckDB, and validate.",
    )

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "run-and-validate":
        sys.exit(cmd_run_and_validate(args))


if __name__ == "__main__":
    main()
