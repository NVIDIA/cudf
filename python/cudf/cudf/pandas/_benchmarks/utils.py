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
import shlex
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
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import duckdb

    duckdb_err = None
except ImportError as e:
    duckdb = None
    duckdb_err = e

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__: list[str] = [
    "RunConfig",
    "build_parser",
    "get_data",
    "parse_args",
    "run_pandas",
]

FrontendType = Literal["in-memory", "pandas-cpu"]

_CPU_ENGINES = frozenset({"pandas-cpu"})

PANDAS_VALIDATION_OPTIONS: dict[str, Any] = {
    "check_dtype": False,
    "atol": 0.02,
}

PDSH_TABLE_NAMES: list[str] = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]

EXIT_SUCCESS = 0
EXIT_QUERY_FAILURE = 3
EXIT_VALIDATION_FAILURE = 4


def benchmark_exit_code(
    query_failures: list[tuple[int, int]],
    validation_failures: list[int],
) -> int:
    """
    Map a run's failures to the process exit code.

    Parameters
    ----------
    query_failures
        ``(query, iteration)`` pairs for queries that raised while running.
    validation_failures
        Queries whose results did not match the expected answer.

    Returns
    -------
    int
        0: Success
        3: Query failure
        4: Validation failure
    """
    if query_failures:
        return EXIT_QUERY_FAILURE
    if validation_failures:
        return EXIT_VALIDATION_FAILURE
    return EXIT_SUCCESS


def get_validation_options(args: Any) -> dict[str, Any]:
    """Get validation options dict from parsed arguments."""
    return {
        **PANDAS_VALIDATION_OPTIONS,
        "atol": args.validation_abs_tol,
    }


@dataclasses.dataclass
class NightlyRole:
    """Role indicating a nightly benchmark run."""

    type: Literal["nightly"] = dataclasses.field(default="nightly", init=False)
    date: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).date().isoformat()
    )


@dataclasses.dataclass
class NsysRole:
    """Role indicating a benchmark run with nsys profiling enabled."""

    type: Literal["nsys"] = dataclasses.field(default="nsys", init=False)


Role = NightlyRole | NsysRole


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
        - 'duckdb': Run duckdb against the same data
        - 'duckdb-disk': Compare to duckdb pregenerated results on disk

    comparison_method
        How the comparison was performed. Currently, only
        'pandas' is supported, which indicates that ``pandas.testing.assert_frame_equal``
        was used.

    comparison_options
        Additional options passed to the comparison method, controlling
        things like the tolerance for floating point comparisons.

    expected_location
        Optional path to disk-based expected results, must be provided if
        source is "duckdb-disk".
    """

    expected_source: Literal["pandas", "duckdb", "duckdb-disk"]
    comparison_method: Literal["pandas"]
    comparison_options: dict[str, Any]
    expected_location: str | None

    def expected_file(self, q_id: int) -> str:
        """Return path to disk-based result for the given query."""
        if self.expected_location is None:
            raise RuntimeError("No expected location given")

        return self.expected_location.rstrip("/") + f"/q*{q_id:02d}.parquet"


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
    iteration: int
    duration: float
    validation_result: ValidationResult | None = None
    status: Literal["success"] = "success"

    @classmethod
    def new(
        cls,
        query: int,
        iteration: int,
        duration: float,
    ) -> SuccessRecord:
        """Create a Record from plain data."""
        return cls(
            query=query,
            iteration=iteration,
            duration=duration,
        )


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
    duckdb: str | None

    @classmethod
    def collect(cls) -> PackageVersions:
        """Collect the versions of the software used to run the query."""
        packages = [
            "cudf",
            "duckdb",
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
class CPUInfo:
    """Information about the host CPU."""

    model: str | None
    physical_cores: int | None
    logical_cores: int | None

    @classmethod
    def collect(cls) -> CPUInfo:
        """Collect CPU information."""
        model: str | None = None
        try:
            with Path("/proc/cpuinfo").open() as f:
                for line in f:
                    if line.startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        physical_cores: int | None = None
        logical_cores: int | None = None
        if psutil is not None:
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
        return cls(
            model=model,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
        )


@dataclasses.dataclass
class HardwareInfo:
    """Information about the hardware used to run the query."""

    gpus: list[GPUInfo]
    cpu: CPUInfo
    # TODO: ucx

    @classmethod
    def collect(cls, *, collect_gpus: bool = True) -> HardwareInfo:
        """
        Collect the hardware information.

        Parameters
        ----------
        collect_gpus : bool, optional
            Whether to collect GPU information.

        Returns
        -------
        HardwareInfo
            The hardware information.
        """
        if collect_gpus and pynvml is not None:
            pynvml.nvmlInit()
            gpus = [
                GPUInfo.from_index(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        else:
            # No GPUs -- CPU-only frontend or NVML unavailable
            gpus = []
        return cls(gpus=gpus, cpu=CPUInfo.collect())


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


def record_from_dict(data: dict[str, Any]) -> SuccessRecord | FailedRecord:
    """
    Read one iteration record back from its serialized form.

    Parameters
    ----------
    data
        One entry of a run's ``records``.

    Returns
    -------
    The record, typed by its ``status``.

    Raises
    ------
    ValueError
        If the status is unrecognized.
    """
    status = data["status"]
    if status == "success":
        validation = data.get("validation_result")
        return SuccessRecord(
            query=data["query"],
            iteration=data["iteration"],
            duration=data["duration"],
            validation_result=(
                ValidationResult(**validation)
                if validation is not None
                else None
            ),
        )
    if status == "error":
        return FailedRecord(
            query=data["query"],
            iteration=data["iteration"],
            traceback=data["traceback"],
        )
    raise ValueError(f"Unrecognized iteration status: {status!r}")


@dataclasses.dataclass(kw_only=True)
class RunConfig:
    """Results for a PDS-H or PDS-DS query run."""

    engine_name: Literal["cudf-pandas", "pandas"]
    queries: list[int]
    suffix: str
    frontend: FrontendType
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[SuccessRecord | FailedRecord]] = dataclasses.field(
        default_factory=dict
    )
    dataset_path: Path
    scale_factor: int | float
    iterations: int
    io_mode: Literal["cold", "lukewarm", "hot"] = "lukewarm"
    n_workers: int = 1
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    command_line: str
    capture_env_vars: str
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hardware: HardwareInfo = dataclasses.field(
        default_factory=HardwareInfo.collect
    )
    run_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    query_set: str
    validation_method: ValidationMethod | None = None
    duckdb_threads: int | None = None
    duckdb_memory_limit: str | None = None
    duckdb_temp_dir: str | None = None
    roles: list[Role] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.io_mode == "hot" and self.iterations < 2:
            raise ValueError(
                "--io-mode hot requires at least 2 iterations: "
                "iteration 0 warms the cache, iterations 1+ are the hot measurements."
            )

        # Update `extra_info.environment` with the captured environment variables.
        self.extra_info.setdefault("environment", {})
        for var in self.capture_env_vars.split(","):
            var_ = var.strip()
            self.extra_info["environment"][var_] = os.environ.get(var_)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunConfig:
        """Create a RunConfig from command line arguments."""
        frontend: FrontendType = args.frontend
        engine_name: Literal["cudf-pandas", "pandas"] = (
            "pandas" if frontend == "pandas-cpu" else "cudf-pandas"
        )

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

        skip_scale_factor_inference = (
            "LIBCUDF_IO_REROUTE_LOCAL_DIR_PATTERN" in os.environ
        ) and ("LIBCUDF_IO_REROUTE_REMOTE_DIR_PATTERN" in os.environ)

        if (
            "pdsh" in name
            and args.scale is not None
            and skip_scale_factor_inference is False
        ):
            # Validate the user-supplied scale factor
            sf_inf = _infer_scale_factor(name, path, args.suffix)
            rel_error = abs((scale_factor - sf_inf) / sf_inf)
            if rel_error > 0.01:
                raise ValueError(
                    f"Specified scale factor is {args.scale}, "
                    f"but the inferred scale factor is {sf_inf}."
                )

        if args.validate_directory is not None:
            validation_method = ValidationMethod(
                expected_source="duckdb-disk",
                comparison_method="pandas",
                comparison_options=get_validation_options(args),
                expected_location=str(args.validate_directory),
            )
        elif args.validate_against is not None:
            validation_method = ValidationMethod(
                expected_source=args.validate_against,
                comparison_method="pandas",
                comparison_options=get_validation_options(args),
                expected_location=None,
            )
        else:
            validation_method = None

        roles: list[Role] = []
        if args.role_nightly:
            roles.append(NightlyRole())
        if args.role_nsys:
            roles.append(NsysRole())

        return cls(
            engine_name=engine_name,
            queries=args.query,
            frontend=frontend,
            dataset_path=path,
            scale_factor=scale_factor,
            iterations=args.iterations,
            io_mode=args.io_mode,
            suffix=args.suffix,
            query_set=args.query_set,
            validation_method=validation_method,
            extra_info=args.extra_info,
            duckdb_threads=args.duckdb_threads,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_temp_dir=args.duckdb_temp_dir,
            command_line=shlex.join(sys.argv),
            capture_env_vars=args.capture_env_vars,
            hardware=HardwareInfo.collect(
                collect_gpus=frontend not in _CPU_ENGINES
            ),
            roles=roles,
        )

    def serialize(self) -> dict:
        """Serialize the run config to a dictionary."""
        result: dict[str, Any] = {
            "engine_name": self.engine_name,
            "queries": self.queries,
            "query_set": self.query_set,
            "dataset_path": str(self.dataset_path),
            "scale_factor": self.scale_factor,
            "suffix": self.suffix,
            "frontend": self.frontend,
            "iterations": self.iterations,
            "io_mode": self.io_mode,
            "n_workers": self.n_workers,
            "extra_info": dict(self.extra_info),
            "run_id": str(self.run_id),
            "timestamp": self.timestamp,
            "command_line": self.command_line,
            "records": {
                k: [dataclasses.asdict(r) for r in v]
                for k, v in self.records.items()
            },
            "versions": dataclasses.asdict(self.versions),
            "hardware": dataclasses.asdict(self.hardware),
            "validation_method": dataclasses.asdict(self.validation_method)
            if self.validation_method
            else None,
            "roles": [dataclasses.asdict(r) for r in self.roles],
        }
        return result

    def summarize(self) -> None:
        """Print a summary of the results."""
        print("Iteration Summary")  # noqa: T201
        print("=======================================")  # noqa: T201

        total_mean_time = 0.0
        for query, records in self.records.items():
            print(f"query: {query}")  # noqa: T201
            print(f"path: {self.dataset_path}")  # noqa: T201
            print(f"scale_factor: {self.scale_factor}")  # noqa: T201
            print(f"frontend: {self.frontend}")  # noqa: T201
            valid_durations = [
                record.duration
                for record in records
                if record.status == "success"
            ]
            if len(valid_durations) > 0:
                mean_time = statistics.mean(valid_durations)
                total_mean_time += mean_time
                print(f"iterations: {self.iterations}")  # noqa: T201
                print("---------------------------------------")  # noqa: T201
                print(f"min time : {min(valid_durations):0.4f}")  # noqa: T201
                print(f"max time : {max(valid_durations):0.4f}")  # noqa: T201
                print(f"mean time: {mean_time:0.4f}")  # noqa: T201
                print("=======================================")  # noqa: T201

        if total_mean_time > 0:
            print(  # noqa: T201
                f"Total mean time across all queries: {total_mean_time:.4f} seconds"
            )
        else:
            print("No successful queries")  # noqa: T201


def get_data(
    path: str | Path,
    table_name: str,
    suffix: str = "",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Get table from dataset."""
    return pd.read_parquet(f"{path}/{table_name}{suffix}", columns=columns)


def check_input_numeric_type(
    run_config: RunConfig,
) -> Literal["decimal", "float"]:
    """Return whether PDS-H money columns are decimal or float."""
    import decimal

    sample = get_data(
        run_config.dataset_path,
        "customer",
        run_config.suffix,
        ["c_acctbal"],
    )
    col = sample["c_acctbal"]
    dtype = col.dtype
    if getattr(dtype, "kind", None) == "f":
        return "float"
    name = str(getattr(dtype, "name", dtype)).lower()
    if "decimal" in name:
        return "decimal"
    try:
        import pyarrow as pa

        pa_dtype = getattr(dtype, "pyarrow_dtype", None)
        if pa_dtype is not None and pa.types.is_decimal(pa_dtype):
            return "decimal"
    except Exception:
        pass
    if getattr(dtype, "kind", None) == "O" or name == "object":
        for val in col.head(16):
            if isinstance(val, decimal.Decimal):
                return "decimal"
    return "float"


def apply_validation_casts(
    frame: pd.DataFrame,
    q_id: int,
    benchmark: Any,
    numeric_type: Literal["decimal", "float"],
) -> pd.DataFrame:
    """Apply per-query dtype casts before assert_frame_equal."""
    casts: dict[str, str] = {}
    expected_casts = getattr(benchmark, "EXPECTED_CASTS", None) or {}
    casts.update(expected_casts.get(q_id, {}))
    if numeric_type == "decimal":
        decimal_casts = (
            getattr(benchmark, "EXPECTED_CASTS_DECIMAL", None) or {}
        )
        casts.update(decimal_casts.get(q_id, {}))
    if not casts:
        return frame
    out = frame.copy()
    for col, dtype in casts.items():
        if col in out.columns:
            out[col] = out[col].astype(dtype)
    return out


def _make_duckdb_config(run_config: RunConfig | None) -> dict[str, Any]:
    """Build a DuckDB connection config dict from a RunConfig."""
    config: dict[str, Any] = {
        "threads": run_config.duckdb_threads
        if (run_config and run_config.duckdb_threads is not None)
        else os.cpu_count(),
    }
    if run_config and run_config.duckdb_memory_limit is not None:
        config["memory_limit"] = run_config.duckdb_memory_limit
    if run_config and run_config.duckdb_temp_dir is not None:
        config["temp_directory"] = run_config.duckdb_temp_dir
    return config


def execute_duckdb_query(
    query: str,
    dataset_path: Path,
    *,
    suffix: str = ".parquet",
    run_config: RunConfig | None = None,
) -> pd.DataFrame:
    """Execute a query with DuckDB and return a pandas DataFrame."""
    if duckdb is None:
        raise ImportError(duckdb_err)
    with disable_module_accelerator():
        with duckdb.connect(config=_make_duckdb_config(run_config)) as conn:
            for name in PDSH_TABLE_NAMES:
                pattern = (Path(dataset_path) / name).as_posix() + suffix
                conn.execute(
                    f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM parquet_scan('{pattern}');"
                )
            return conn.execute(query).df()


def drop_file_page_cache_recursively(path: os.PathLike | str) -> None:
    """Drop the Linux page cache for all files under `path`."""
    try:
        import kvikio
    except ImportError as err:
        raise RuntimeError(
            "kvikio is required for cold-run page cache dropping. "
            "Install it or switch to --io-mode lukewarm."
        ) from err
    p = Path(path).expanduser()
    if p.is_file():
        kvikio.drop_file_page_cache(p)
        return
    for f in p.rglob("*"):
        if f.is_file():
            kvikio.drop_file_page_cache(f)


def execute_query(
    q_id: int,
    i: int,
    q: Callable[[RunConfig], pd.DataFrame],
    run_config: RunConfig,
) -> tuple[pd.DataFrame, float]:
    """Execute a query with NVTX annotation."""
    if run_config.io_mode == "cold":
        drop_file_page_cache_recursively(run_config.dataset_path)

    with nvtx.annotate(
        message=f"Query {q_id} - Iteration {i}",
        domain="cudf.pandas",
        color="green",
    ):
        if run_config.frontend == "pandas-cpu":
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


def build_parser(num_queries: int = 22) -> argparse.ArgumentParser:
    """Build the argument parser for PDS-H benchmarks."""
    parser = argparse.ArgumentParser(
        prog="cudf.pandas PDS-H Benchmarks",
        description=textwrap.dedent(f"""\
            cudf.pandas benchmark runner.

            Exit code description:
            - {EXIT_SUCCESS} : Success
            - 1 : Unhandled exception during query run
            - 2 : Invalid command line arguments
            - {EXIT_QUERY_FAILURE} : Query failure (setup or execution)
            - {EXIT_VALIDATION_FAILURE} : Validation failure
            """),
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
        "--frontend",
        default="in-memory",
        type=str,
        choices=["in-memory", "pandas-cpu"],
        help=textwrap.dedent("""\
            Execution frontend:
                - in-memory : Single-process GPU evaluation via cudf.pandas
                - pandas-cpu : pandas CPU execution (no GPU)"""),
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="Number of times to run the same query.",
    )
    parser.add_argument(
        "--sleep-between-iterations",
        default=0,
        type=float,
        dest="sleep_between_iterations",
        metavar="SECONDS",
        help="Sleep this many seconds between iterations (default: 0).",
    )
    parser.add_argument(
        "--io-mode",
        dest="io_mode",
        default="lukewarm",
        choices=["cold", "lukewarm", "hot"],
        help=textwrap.dedent("""\
            Cache state control for each timed iteration:
                - cold     : Drop Linux page cache before each iteration (requires kvikio)
                - lukewarm : No cache manipulation; OS cache state unchanged (default)
                - hot      : One untimed warmup iteration to populate cache before measured runs"""),
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--validate-against",
        choices=["duckdb", "pandas"],
        default=None,
        help=(
            "Validate the result against CPU execution. This will "
            "run the query with both GPU and baseline engine (pandas or DuckDB), collect the "
            "results in memory, and compare them using pandas'. "
            "At larger scale factors, computing the expected result can be slow so "
            "--validate-directory should be used instead."
        ),
    )
    group.add_argument(
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
        "--results-directory",
        type=Path,
        default=None,
        help="Optional directory to write query results as parquet files.",
    )
    parser.add_argument(
        "--output-expected-directory",
        type=Path,
        default=None,
        help="Optional directory to write expected results as parquet files.",
    )
    parser.add_argument(
        "--validation-abs-tol",
        type=float,
        default=0.01,
        help="Absolute tolerance for assert_frame_equal validation. Default: 0.01",
    )
    parser.add_argument(
        "--extra-info",
        type=json.loads,
        default={},
        help="Extra information to add to the output file (must be JSON-serializable).",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=None,
        help="Number of threads for DuckDB to use. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        type=str,
        default=None,
        help="DuckDB memory limit (e.g. '500GB'). If unset, DuckDB uses its default.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        type=str,
        default=None,
        help="Directory for DuckDB to spill temporary data to disk.",
    )
    parser.add_argument(
        "--capture-env-vars",
        type=str,
        default="CUDF_PANDAS_FAIL_ON_FALLBACK,CUDF_PANDAS_FALLBACK_MODE,CUDF_PANDAS_RMM_MODE,CUDF_SPILL,CUDF_SPILL_DEVICE_LIMIT,KVIKIO_COMPAT_MODE,KVIKIO_NTHREADS,LIBCUDF_HOST_DECOMPRESSION,LIBCUDF_NUM_HOST_WORKERS,OMP_NUM_THREADS",
        help="Comma-separated list of environment variables to capture. Written to ``extra_info.environment``.",
    )
    parser.add_argument(
        "--role-nightly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the 'nightly' role to the benchmark run output.",
    )
    parser.add_argument(
        "--role-nsys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the 'nsys' role to the benchmark run output.",
    )

    return parser


def parse_args(
    args: Sequence[str] | None = None,
    num_queries: int = 22,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse command line arguments."""
    if parser is None:
        parser = build_parser(num_queries)
    parsed_args = parser.parse_args(args)

    if (
        parsed_args.suffix
        and not parsed_args.suffix.startswith(".")
        and not parsed_args.suffix.startswith("/")
    ):
        parsed_args.suffix = f".{parsed_args.suffix}"

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
    result_casts: dict[str, str] | None = None,
) -> SuccessRecord:
    """Run a single query iteration. Caller must wrap in try/except."""
    result, duration = execute_query(q_id, iteration, q, run_config)

    if expected is not None:
        if result_casts:
            for col, dtype in result_casts.items():
                result[col] = result[col].astype(dtype)
        result_for_validation = getattr(result, "_fsproxy_slow", result)
        comparison_options = (
            run_config.validation_method.comparison_options
            if run_config.validation_method is not None
            else PANDAS_VALIDATION_OPTIONS
        )
        try:
            native_pd = getattr(pd, "_fsproxy_slow", pd)
            native_pd.testing.assert_frame_equal(
                result_for_validation, expected, **comparison_options
            )
        except Exception as e:
            validation_result = ValidationResult.from_error(e)
        else:
            validation_result = ValidationResult(status="Passed", message=None)
    else:
        validation_result = None

    if args.print_results:
        print(result)  # noqa: T201

    if args.results_directory is not None and iteration == 0:
        results_dir = Path(args.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"q_{q_id:02d}.parquet"
        result.to_parquet(output_path)

    return SuccessRecord(
        query=q_id,
        iteration=iteration,
        duration=duration,
        validation_result=validation_result,
    )


def run_pandas_query(
    q_id: int,
    benchmark: Any,
    run_config: RunConfig,
    args: argparse.Namespace,
) -> QueryRunResult:
    """Run all iterations for a single query. Caller must wrap in try/except."""
    try:
        q = getattr(benchmark, f"q{q_id}")
    except AttributeError as err:
        raise NotImplementedError(f"Query {q_id} not implemented.") from err

    expected: pd.DataFrame | None = None
    result_casts: dict[str, str] | None = None
    validation_method = run_config.validation_method
    if validation_method is not None:
        numeric_type = check_input_numeric_type(run_config)
        match validation_method.expected_source:
            case "pandas":
                cpu_run_config = dataclasses.replace(
                    run_config, frontend="pandas-cpu"
                )
                expected, _ = execute_query(q_id, 0, q, cpu_run_config)
            case "duckdb":
                duckdb_queries_cls = benchmark().duckdb_queries
                get_ddb = getattr(duckdb_queries_cls, f"q{q_id}")
                base_sql = get_ddb(run_config)
                expected = execute_duckdb_query(
                    base_sql,
                    run_config.dataset_path,
                    suffix=run_config.suffix,
                    run_config=run_config,
                )
            case "duckdb-disk":
                if validation_method.expected_location is None:
                    raise RuntimeError("No expected location given")
                expected_dir = Path(validation_method.expected_location)
                matches = list(expected_dir.glob(f"q*{q_id:02d}.parquet"))
                if not matches:
                    matches = list(expected_dir.glob(f"q*{q_id}.parquet"))
                if not matches:
                    raise FileNotFoundError(
                        f"No expected file for query {q_id} in "
                        f"{validation_method.expected_location}"
                    )
                expected = pd._fsproxy_slow.read_parquet(matches[0])
            case baseline:
                raise ValueError(f"Invalid baseline: {baseline}")

        expected = apply_validation_casts(
            expected, q_id, benchmark, numeric_type
        )
        casts: dict[str, str] = {}
        casts.update(getattr(benchmark, "EXPECTED_CASTS", {}).get(q_id, {}))
        if numeric_type == "decimal":
            casts.update(
                getattr(benchmark, "EXPECTED_CASTS_DECIMAL", {}).get(q_id, {})
            )
        result_casts = casts or None

    if args.output_expected_directory is not None:
        assert expected is not None, (
            "Expected result must be computed before writing to disk."
        )
        expected_dir = Path(args.output_expected_directory)
        expected_dir.mkdir(parents=True, exist_ok=True)
        expected.to_parquet(expected_dir / f"q_{q_id:02d}.parquet")

    query_records: list[SuccessRecord | FailedRecord] = []
    iteration_failures: list[tuple[int, int]] = []
    validation_failed = False
    record: SuccessRecord | FailedRecord

    for i in range(args.iterations):
        if i > 0 and args.sleep_between_iterations > 0:
            print(  # noqa: T201
                f"==> Sleeping {args.sleep_between_iterations} seconds "
                "between iterations",
                flush=True,
            )
            time.sleep(args.sleep_between_iterations)

        try:
            record = run_pandas_query_iteration(
                q_id, i, q, run_config, args, expected, result_casts
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
                prefix = "✅ " if record.validation_result else ""
                print(  # noqa: T201
                    f"{prefix}Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s",
                    flush=True,
                )
        query_records.append(record)
    return QueryRunResult(
        query_records=query_records,
        iteration_failures=iteration_failures,
        validation_failed=validation_failed,
    )


def run_pandas(benchmark: Any, args: argparse.Namespace) -> None:
    """Run the queries using the given benchmark and frontend."""
    vars(args).update({"query_set": benchmark.name})
    run_config = RunConfig.from_args(args)
    validation_failures: list[int] = []
    query_failures: list[tuple[int, int]] = []

    records: defaultdict[int, list[SuccessRecord | FailedRecord]] = (
        defaultdict(list)
    )

    for q_id in run_config.queries:
        try:
            result = run_pandas_query(
                q_id=q_id,
                benchmark=benchmark,
                run_config=run_config,
                args=args,
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

    if (
        run_config.validation_method is not None
        and run_config.frontend not in _CPU_ENGINES
    ):
        print("\nValidation Summary")  # noqa: T201
        print("==================")  # noqa: T201
        if validation_failures:
            print(  # noqa: T201
                f"{len(validation_failures)} queries failed validation: {sorted(set(validation_failures))}"
            )
        if query_failures:
            print(  # noqa: T201
                "Validation was skipped for queries that failed to run: "
                f"{sorted({q_id for q_id, _ in query_failures})}"
            )
        if not validation_failures and not query_failures:
            print("All validated queries passed.")  # noqa: T201

    args.output.write(json.dumps(run_config.serialize()))
    args.output.write("\n")

    sys.exit(benchmark_exit_code(query_failures, validation_failures))
