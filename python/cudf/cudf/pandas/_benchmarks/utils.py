# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions/classes for running the PDS-H and PDS-DS benchmarks."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import pprint
import statistics
import sys
import textwrap
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import nvtx

import cudf.pandas
from cudf.pandas.module_accelerator import disable_module_accelerator

cudf.pandas.install()

import pandas as pd  # noqa: E402

try:
    import pynvml
except ImportError:
    pynvml = None

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ExecutorType = Literal["in-memory", "cpu"]


@dataclasses.dataclass
class ValidationResult:
    """
    Result of a validation run.

    Parameters
    ----------
    status
        The status of the validation. Either 'Passed' or 'Failed'.
    message
        The message from the validation. This should be ``None`` if
        the validation passed, and a string describing the failure otherwise.
    details
        Additional details about the validation failure.
    """

    status: Literal["Passed", "Failed"]
    message: str | None
    details: dict[str, Any] | None = None

    @classmethod
    def from_error(cls, error: Exception) -> ValidationResult:
        """
        Create a ValidationResult from some exception.

        Parameters
        ----------
        error : Exception
            The error to create a ValidationResult from.

        Returns
        -------
        ValidationResult
            The ValidationResult created from the error.
        """
        return cls(status="Failed", message=str(error))


@dataclasses.dataclass
class ValidationMethod:
    """
    Information about how the validation was performed.

    Parameters
    ----------
    expected_source
        A name indicating the source of the expected results.

        - 'pandas': Run pandas against the same data
        - 'duckdb': Compare against pre-computed DuckDB results

    comparison_method
        How the comparison was performed. Currently, only
        'pandas' is supported, which indicates that ``pandas.testing.assert_frame_equal``
        was used.

    comparison_options
        Additional options passed to the comparison method, controlling
        things like the tolerance for floating point comparisons.
    """

    expected_source: Literal["duckdb", "pandas"]
    comparison_method: Literal["pandas"]
    comparison_options: dict[str, Any]


@dataclasses.dataclass(kw_only=True)
class FailedRecord:
    """Records a failed query iteration."""

    query: int
    iteration: int
    status: Literal["error"] = "error"
    traceback: str


@dataclasses.dataclass(kw_only=True)
class SuccessRecord:
    """Results for a single run of a single PDS-H query."""

    query: int
    duration: float
    validation_result: ValidationResult | None = None
    status: Literal["success"] = "success"


@dataclasses.dataclass
class QueryRunResult:
    """Result of running a single query (all iterations)."""

    query_records: list[SuccessRecord | FailedRecord]
    iteration_failures: list[tuple[int, int]]
    validation_failed: bool


@dataclasses.dataclass
class VersionInfo:
    """Information about the commit of the software used to run the query."""

    version: str
    commit: str


@dataclasses.dataclass
class PackageVersions:
    """Information about the versions of the software used to run the query."""

    cudf: str | VersionInfo
    pandas: str
    python: str

    @classmethod
    def collect(cls) -> PackageVersions:
        """Collect the versions of the software used to run the query."""
        packages = [
            "cudf",
            "pandas",
        ]
        versions: dict[str, str | VersionInfo | None] = {}
        for name in packages:
            try:
                package = importlib.import_module(name)
            except (AttributeError, ImportError):
                versions[name] = None
            else:
                if name == "cudf":
                    versions[name] = VersionInfo(
                        version=package.__version__,
                        commit=package.__git_commit__,
                    )
                else:
                    versions[name] = package.__version__

        versions["python"] = ".".join(str(v) for v in sys.version_info[:3])
        # we manually ensure that only cudf and pandas have a VersionInfo
        return cls(**versions)  # type: ignore[arg-type]


@dataclasses.dataclass
class GPUInfo:
    """Information about a specific GPU."""

    name: str
    index: int
    free_memory: int | None
    used_memory: int | None
    total_memory: int | None

    @classmethod
    def from_index(cls, index: int) -> GPUInfo:
        """Create a GPUInfo from an index."""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return cls(
                name=pynvml.nvmlDeviceGetName(handle),
                index=index,
                free_memory=memory.free,
                used_memory=memory.used,
                total_memory=memory.total,
            )
        except pynvml.NVMLError_NotSupported:
            # Happens on systems without traditional GPU memory (e.g., Grace Hopper),
            # where nvmlDeviceGetMemoryInfo is not supported.
            # See: https://github.com/NVIDIA/cudf/issues/19427
            return cls(
                name=pynvml.nvmlDeviceGetName(handle),
                index=index,
                free_memory=None,
                used_memory=None,
                total_memory=None,
            )


@dataclasses.dataclass
class HardwareInfo:
    """Information about the hardware used to run the query."""

    gpus: list[GPUInfo]
    # TODO: ucx

    @classmethod
    def collect(cls) -> HardwareInfo:
        """Collect the hardware information."""
        if pynvml is not None:
            pynvml.nvmlInit()
            gpus = [
                GPUInfo.from_index(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        else:
            # No GPUs -- probably running in CPU mode
            gpus = []
        return cls(gpus=gpus)


def _infer_scale_factor(
    name: str, path: str | Path, suffix: str
) -> int | float:
    if "pdsh" in name:
        supplier = get_data(path, "supplier", suffix)
        num_rows = len(supplier)
        return num_rows / 10_000

    elif "pdsds" in name:
        # TODO: Keep a map of SF-row_count because of nonlinear scaling
        # See: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-DS_v4.0.0.pdf pg.46
        customer = get_data(path, "promotion", suffix)
        num_rows = len(customer)
        return num_rows / 300

    else:
        raise ValueError(f"Invalid benchmark script name: '{name}'.")


@dataclasses.dataclass(kw_only=True)
class RunConfig:
    """Results for a PDS-H or PDS-DS query run."""

    queries: list[int]
    suffix: str
    executor: ExecutorType
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[SuccessRecord | FailedRecord]] = dataclasses.field(
        default_factory=dict
    )
    dataset_path: Path
    scale_factor: int | float
    iterations: int
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hardware: HardwareInfo = dataclasses.field(
        default_factory=HardwareInfo.collect
    )
    run_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    query_set: str
    validation_method: ValidationMethod | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunConfig:
        """Create a RunConfig from command line arguments."""
        executor: ExecutorType = args.executor

        path = args.path
        name = args.query_set
        scale_factor = args.scale

        if scale_factor is None:
            if "pdsds" in name:
                raise ValueError(
                    "--scale is required for PDS-DS benchmarks.\n"
                    "TODO: This will be inferred once we maintain a map of scale factors to row counts."
                )
            if path is None:
                raise ValueError(
                    "Must specify --root and --scale if --path is not specified."
                )
            # For PDS-H, infer scale factor based on row count
            scale_factor = _infer_scale_factor(name, path, args.suffix)
        if path is None:
            path = f"{args.root}/scale-{scale_factor}"
        scale_factor = float(scale_factor)
        try:
            scale_factor_int = int(scale_factor)
        except ValueError:
            pass
        else:
            if scale_factor_int == scale_factor:
                scale_factor = scale_factor_int

        if "pdsh" in name and args.scale is not None:
            # Validate the user-supplied scale factor
            sf_inf = _infer_scale_factor(name, path, args.suffix)
            rel_error = abs((scale_factor - sf_inf) / sf_inf)
            if rel_error > 0.01:
                raise ValueError(
                    f"Specified scale factor is {args.scale}, "
                    f"but the inferred scale factor is {sf_inf}."
                )

        if args.validate_directory:
            validation_method = ValidationMethod(
                expected_source="duckdb",
                comparison_method="pandas",
                comparison_options={},
            )
        elif args.validate:
            validation_method = ValidationMethod(
                expected_source="pandas",
                comparison_method="pandas",
                comparison_options={},
            )
        else:
            validation_method = None

        return cls(
            queries=args.query,
            executor=executor,
            dataset_path=path,
            scale_factor=scale_factor,
            iterations=args.iterations,
            suffix=args.suffix,
            query_set=args.query_set,
            validation_method=validation_method,
        )

    def serialize(self) -> dict:
        """Serialize the run config to a dictionary."""
        result = dataclasses.asdict(self)
        result["run_id"] = str(self.run_id)
        return result

    def summarize(self) -> None:
        """Print a summary of the results."""
        print("Iteration Summary")  # noqa: T201
        print("=======================================")  # noqa: T201

        for query, records in self.records.items():
            print(f"query: {query}")  # noqa: T201
            print(f"path: {self.dataset_path}")  # noqa: T201
            print(f"scale_factor: {self.scale_factor}")  # noqa: T201
            print(f"executor: {self.executor}")  # noqa: T201
            if len(records) > 0:
                valid_durations = [
                    record.duration
                    for record in records
                    if record.status == "success"
                ]
                print(f"iterations: {self.iterations}")  # noqa: T201
                print("---------------------------------------")  # noqa: T201
                print(  # noqa: T201
                    f"min time : {min(valid_durations):0.4f}"
                )
                print(  # noqa: T201
                    f"max time : {max(valid_durations):0.4f}"
                )
                print(  # noqa: T201
                    f"mean time: {statistics.mean(valid_durations):0.4f}"
                )
                print("=======================================")  # noqa: T201
        total_mean_time = sum(
            statistics.mean(
                record.duration
                for record in records
                if record.status == "success"
            )
            for records in self.records.values()
            if records
        )
        print(  # noqa: T201
            f"Total mean time across all queries: {total_mean_time:.4f} seconds"
        )


def get_data(
    path: str | Path,
    table_name: str,
    suffix: str = "",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Get table from dataset."""
    return pd.read_parquet(f"{path}/{table_name}{suffix}", columns=columns)


def execute_query(
    q_id: int,
    i: int,
    q: Callable[[RunConfig], pd.DataFrame],
    run_config: RunConfig,
) -> tuple[pd.DataFrame, float]:
    """Execute a query with NVTX annotation."""
    with nvtx.annotate(
        message=f"Query {q_id} - Iteration {i}",
        domain="cudf.pandas",
        color="green",
    ):
        if run_config.executor == "cpu":
            with disable_module_accelerator():
                start_time = time.monotonic()
                result = q(run_config)
                end_time = time.monotonic()
        else:
            assert cudf.pandas.LOADED
            start_time = time.monotonic()
            result = q(run_config)
            end_time = time.monotonic()
        return result, end_time - start_time


def _query_type(num_queries: int) -> Callable[[str | int], list[int]]:
    def parse(query: str | int) -> list[int]:
        if isinstance(query, int):
            return [query]
        if query == "all":
            return list(range(1, num_queries + 1))

        result: set[int] = set()
        for part in query.split(","):
            if "-" in part:
                start, end = part.split("-")
                result.update(range(int(start), int(end) + 1))
            else:
                result.add(int(part))
        return sorted(result)

    return parse


def list_validation_files(
    validate_directory: Path,
) -> dict[int, Path]:
    """List the validation files in the given directory."""
    validation_files: dict[int, Path] = {}
    for q_path in validate_directory.glob("q*.parquet"):
        q_id = int(q_path.stem.lstrip("q").lstrip("_"))
        validation_files[q_id] = q_path
    return validation_files


def parse_args(
    args: Sequence[str] | None = None, num_queries: int = 22
) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="cudf.pandas PDS-H Benchmarks",
        description="cudf.pandas benchmark runner.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=_query_type(num_queries),
        help=textwrap.dedent("""\
            Query to run. One of the following:
            - A single number (e.g. 11)
            - A comma-separated list of query numbers (e.g. 1,3,7)
            - A range of query number (e.g. 1-11,23-34)
            - The string 'all' to run all queries (1 through 22)"""),
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.environ.get("PDSH_DATASET_PATH"),
        help=textwrap.dedent("""\
            Path to the root directory of the PDS-H dataset.
            Defaults to the PDSH_DATASET_PATH environment variable."""),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ.get("PDSH_DATASET_ROOT"),
        help="Root PDS-H dataset directory (ignored if --path is used).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Dataset scale factor.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".parquet",
        help=textwrap.dedent("""\
            File suffix for input table files.
            Default: .parquet"""),
    )
    parser.add_argument(
        "-e",
        "--executor",
        default="in-memory",
        type=str,
        choices=["in-memory", "cpu"],
        help=textwrap.dedent("""\
            Query executor backend:
                - in-memory : Use cudf.pandas
                - cpu       : Use pandas"""),
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="Number of times to run the same query.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("at"),
        default="pdsh_results.jsonl",
        help="Output file path.",
    )
    parser.add_argument(
        "--summarize",
        action=argparse.BooleanOptionalAction,
        help="Summarize the results.",
        default=True,
    )
    parser.add_argument(
        "--print-results",
        action=argparse.BooleanOptionalAction,
        help="Print the query results",
        default=True,
    )
    parser.add_argument(
        "--validate-directory",
        type=Path,
        default=None,
        help=(
            "Validate the results against a directory with a pre-computed set of 'golden' results. "
            "The directory should contain one parquet file per query, named 'qDD.parquet', where DD is the "
            "zero-padded query number. The JSON output will include the validation results for each record."
        ),
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Validate the result against CPU execution. This will "
            "run the query with both GPU and baseline engine (pandas), collect the "
            "results in memory, and compare them using pandas'. "
            "At larger scale factors, computing the expected result can be slow so "
            "--validate-directory should be used instead."
        ),
    )
    parsed_args = parser.parse_args(args)
    if parsed_args.validate_directory and parsed_args.validate:
        raise ValueError(
            "Specify either --validate-directory or --validate, not both."
        )
    if (
        parsed_args.validate_directory is not None
        and not parsed_args.validate_directory.exists()
    ):
        raise FileNotFoundError(
            f"--validate-directory: {parsed_args.validate_directory} does not exist."
        )
    if parsed_args.validate_directory:
        validation_files = list_validation_files(
            parsed_args.validate_directory
        )
        missing_files = [
            str(x)
            for x in set(parsed_args.query) - set(validation_files.keys())
        ]

        if missing_files:
            raise ValueError(
                f"Missing files for queries: {','.join(missing_files)}"
            )
    return parsed_args


def run_pandas_query_iteration(
    q_id: int,
    iteration: int,
    q: Callable[[RunConfig], pd.DataFrame],
    run_config: RunConfig,
    args: argparse.Namespace,
    expected: pd.DataFrame | None,
) -> SuccessRecord:
    """Run a single query iteration. Caller must wrap in try/except."""
    result, duration = execute_query(q_id, iteration, q, run_config)

    if expected is not None:
        try:
            pd.testing.assert_frame_equal(
                result, expected, check_dtype=False, atol=0.02
            )
        except Exception as e:
            validation_result = ValidationResult.from_error(e)
        else:
            validation_result = ValidationResult(status="Passed", message=None)
    else:
        validation_result = None

    if args.print_results:
        print(result)  # noqa: T201

    return SuccessRecord(
        query=q_id, duration=duration, validation_result=validation_result
    )


def run_pandas_query(
    q_id: int,
    benchmark: Any,
    run_config: RunConfig,
    args: argparse.Namespace,
    validation_files: dict[int, Path] | None,
) -> QueryRunResult:
    """Run all iterations for a single query. Caller must wrap in try/except."""
    try:
        q = getattr(benchmark, f"q{q_id}")
    except AttributeError as err:
        raise NotImplementedError(f"Query {q_id} not implemented.") from err

    expected: pd.DataFrame | None = None
    if args.validate:
        cpu_run_config = dataclasses.replace(run_config, executor="cpu")
        expected, _ = execute_query(q_id, 0, q, cpu_run_config)
    elif validation_files is not None:
        expected = pd._fsproxy_slow.read_parquet(validation_files[q_id])
    else:
        expected = None

    query_records: list[SuccessRecord | FailedRecord] = []
    iteration_failures: list[tuple[int, int]] = []
    validation_failed = False
    record: SuccessRecord | FailedRecord

    for i in range(args.iterations):
        try:
            record = run_pandas_query_iteration(
                q_id, i, q, run_config, args, expected
            )
        except Exception:
            print(f"❌ query={q_id} iteration={i} failed!")  # noqa: T201
            print(traceback.format_exc())  # noqa: T201
            iteration_failures.append((q_id, i))
            record = FailedRecord(
                query=q_id,
                iteration=i,
                traceback=traceback.format_exc(),
            )
        else:
            if (
                record.validation_result
                and record.validation_result.status == "Failed"
            ):
                validation_failed = True
                print(  # noqa: T201
                    f"❌ Query {q_id} failed validation!\n{record.validation_result.message}"
                )
                if record.validation_result.details:
                    pprint.pprint(record.validation_result.details)  # noqa: T203
                else:
                    print(  # noqa: T201
                        f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s",
                        flush=True,
                    )
        query_records.append(record)
    return QueryRunResult(
        query_records=query_records,
        iteration_failures=iteration_failures,
        validation_failed=validation_failed,
    )


def run_pandas(
    benchmark: Any,
    options: Sequence[str] | None = None,
    num_queries: int = 22,
) -> None:
    """Run the queries using the given benchmark and executor options."""
    args = parse_args(options, num_queries=num_queries)
    vars(args).update({"query_set": benchmark.name})
    run_config = RunConfig.from_args(args)
    validation_failures: list[int] = []
    query_failures: list[tuple[int, int]] = []

    records: defaultdict[int, list[SuccessRecord | FailedRecord]] = (
        defaultdict(list)
    )

    if args.validate_directory is not None:
        validation_files = list_validation_files(args.validate_directory)
    else:
        validation_files = None

    for q_id in run_config.queries:
        try:
            result = run_pandas_query(
                q_id=q_id,
                benchmark=benchmark,
                run_config=run_config,
                args=args,
                validation_files=validation_files,
            )
        except Exception:
            print(f"❌ query={q_id} failed (setup or execution)!")  # noqa: T201
            print(traceback.format_exc())  # noqa: T201
            query_failures.append((q_id, -1))
            record = FailedRecord(
                query=q_id,
                iteration=-1,
                traceback=traceback.format_exc(),
            )
            result = QueryRunResult(
                query_records=[record],
                iteration_failures=[],
                validation_failed=False,
            )

        query_failures.extend(result.iteration_failures)
        if result.validation_failed:
            validation_failures.append(q_id)
        records[q_id].extend(result.query_records)

    run_config = dataclasses.replace(run_config, records=dict(records))

    if args.summarize:
        run_config.summarize()

    if args.validate and run_config.executor != "cpu":
        print("\nValidation Summary")  # noqa: T201
        print("==================")  # noqa: T201
        if validation_failures:
            print(  # noqa: T201
                f"{len(validation_failures)} queries failed validation: {sorted(set(validation_failures))}"
            )
        else:
            print("All validated queries passed.")  # noqa: T201

    args.output.write(json.dumps(run_config.serialize()))
    args.output.write("\n")

    exit_code = 1 if (query_failures or validation_failures) else 0
    sys.exit(exit_code)
