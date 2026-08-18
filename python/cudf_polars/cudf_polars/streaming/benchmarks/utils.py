# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark utilities for the RapidsMPF SPMD and Ray frontends."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import io
import itertools
import json
import logging
import os
import pprint
import shlex
import sys
import textwrap
import time
import traceback
import uuid
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Literal

import nvtx

import polars as pl

__all__: list[str] = [
    "COUNT_DTYPE",
    "QueryResult",
    "RunConfig",
    "build_parser",
    "get_data",
    "parse_args",
    "run_duckdb",
    "run_polars",
]

# The dtype for count() aggregations depends on the presence
# of the polars-runtime-64 package (`polars[rt64]`).
HAS_POLARS_RT_64 = pl.config.plr.RUNTIME_REPR == "rt64"
COUNT_DTYPE = pl.UInt64() if HAS_POLARS_RT_64 else pl.UInt32()

try:
    import psutil
except ImportError:
    psutil = None

try:
    import duckdb

    duckdb_err = None
except ImportError as e:
    duckdb = None
    duckdb_err = e

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    import cudf_polars.dsl.tracing
    import cudf_polars.quent
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.dsl.tracing import Scope
    from cudf_polars.dsl.translate import Translator
    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.streaming.benchmarks.asserts import (
        ValidationError,
        assert_tpch_result_equal,
    )
    from cudf_polars.streaming.explain import explain_query
    from cudf_polars.streaming.parallel import evaluate_streaming
    from cudf_polars.utils.config import ConfigOptions

    CUDF_POLARS_AVAILABLE = True
except ImportError:
    CUDF_POLARS_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.streaming.explain import SerializablePlan

POLARS_VALIDATION_OPTIONS = {
    "check_row_order": True,
    "check_column_order": True,
    "check_dtypes": True,
    "check_exact": False,
    "rel_tol": 1e-5,
    "abs_tol": 1e-2,
}


def get_validation_options(args: Any) -> dict[str, Any]:
    """Get validation options dict from parsed arguments."""
    return {
        **POLARS_VALIDATION_OPTIONS,
        "abs_tol": args.validation_abs_tol,
    }


try:
    import structlog
    import structlog.contextvars
    import structlog.processors
    import structlog.stdlib
except ImportError:
    _HAS_STRUCTLOG = False
else:
    _HAS_STRUCTLOG = True


_STREAMING_FRONTENDS = frozenset({"dask", "ray", "spmd"})
_CPU_ENGINES = frozenset({"polars-cpu", "duckdb"})


@dataclasses.dataclass
class NightlyRole:
    """Role indicating a nightly benchmark run."""

    type: Literal["nightly"] = dataclasses.field(default="nightly", init=False)
    date: str = dataclasses.field(
        default_factory=lambda: datetime.now(UTC).date().isoformat()
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

            This will correctly propagate "message" and "details" from
            ``cudf_polars.testing.asserts.ValidationError``.

        Returns
        -------
        ValidationResult
            The ValidationResult created from the error.
        """
        match error:
            case ValidationError(message=message, details=details):
                return cls(status="Failed", message=message, details=details)
            case _:
                return cls(status="Failed", message=str(error))


@dataclasses.dataclass
class ValidationMethod:
    """
    Information about how the validation was performed.

    Parameters
    ----------
    expected_source
        A name indicating the source of the expected results.

        - 'polars-cpu': Run polars against the same data
        - 'duckdb': Run duckdb against the same data
        - 'duckdb-disk': Compare to duckdb pregenerated results on disk

    comparison_method
        How the comparison was performed. Currently, only
        'polars' is supported, which indicates that ``polars.testing.assert_frame_equal``
        was used.

    comparison_options
        Additional options passed to the comparison method, controlling
        things like the tolerance for floating point comparisons.

    expected_location
        Optional path to disk-based expected results, must be provided if
        source is "duckdb-disk".
    """

    expected_source: Literal["polars-cpu", "duckdb", "duckdb-disk"]
    comparison_method: Literal["polars"]
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
    statistics: dict[str, Any] | None = None
    traces: list[dict[str, Any]] | None = None
    validation_result: ValidationResult | None = None
    status: Literal["success"] = "success"

    @classmethod
    def new(
        cls,
        query: int,
        iteration: int,
        duration: float,
        statistics: dict[str, Any] | None = None,
        traces: list[dict[str, Any]] | None = None,
    ) -> SuccessRecord:
        """Create a Record from plain data."""
        return cls(
            query=query,
            iteration=iteration,
            duration=duration,
            statistics=statistics,
            traces=traces,
        )


@dataclasses.dataclass
class QueryRunResult:
    """Result of running a single query (all iterations)."""

    query_records: list[SuccessRecord | FailedRecord]
    plan: SerializablePlan | None
    iteration_failures: list[tuple[int, int]]
    validation_failed: bool
    partition_plan_rows: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class VersionInfo:
    """Information about the commit of the software used to run the query."""

    version: str
    commit: str


@dataclasses.dataclass
class PackageVersions:
    """Information about the versions of the software used to run the query."""

    cudf_polars: str | VersionInfo
    polars: str
    python: str
    rapidsmpf: str | VersionInfo | None
    duckdb: str | None

    @classmethod
    def collect(cls) -> PackageVersions:
        """Collect the versions of the software used to run the query."""
        packages = [
            "cudf_polars",
            "duckdb",
            "polars",
            "rapidsmpf",
        ]
        versions: dict[str, str | VersionInfo | None] = {}
        for name in packages:
            try:
                package = importlib.import_module(name)
            except (AttributeError, ImportError):
                versions[name] = None
            else:
                if name in ("cudf_polars", "rapidsmpf"):
                    versions[name] = VersionInfo(
                        version=package.__version__,
                        commit=package.__git_commit__,
                    )
                else:
                    versions[name] = package.__version__

        versions["python"] = ".".join(str(v) for v in sys.version_info[:3])
        # we manually ensure that only cudf-polars and rapidsmpf have a VersionInfo
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
            model=model, physical_cores=physical_cores, logical_cores=logical_cores
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
            gpus = [GPUInfo.from_index(i) for i in range(pynvml.nvmlDeviceGetCount())]
        else:
            # No GPUs -- CPU-only frontend or NVML unavailable
            gpus = []
        return cls(gpus=gpus, cpu=CPUInfo.collect())


def get_data(path: str | Path, table_name: str, suffix: str = "") -> pl.LazyFrame:
    """
    Get table from dataset.

    Supports both single-file (e.g. ``supplier.parquet``) and
    directory-based (e.g. ``supplier/``) parquet layouts.
    When the file ``{path}/{table_name}{suffix}`` does not exist on the
    local filesystem, falls back to scanning ``{path}/{table_name}`` as a
    directory of parquet files.
    """
    file_path = str(path).removesuffix("/") + f"/{table_name}{suffix}"
    return pl.scan_parquet(file_path)


def _infer_scale_factor(name: str, path: str | Path, suffix: str) -> int | float:
    if "pdsh" in name:
        supplier = get_data(path, "supplier", suffix)
        num_rows = supplier.select(pl.len()).collect().item(0, 0)
        return num_rows / 10_000

    elif "pdsds" in name:
        # TODO: Keep a map of SF-row_count because of nonlinear scaling
        # See: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-DS_v4.0.0.pdf pg.46
        customer = get_data(path, "promotion", suffix)
        num_rows = customer.select(pl.len()).collect().item(0, 0)
        return num_rows / 300

    else:
        raise ValueError(f"Invalid benchmark script name: '{name}'.")


@dataclasses.dataclass(kw_only=True)
class RunConfig:
    """Benchmark run configuration for SPMD / Ray / DuckDB frontends."""

    engine_name: Literal["polars-cpu", "cudf-polars", "duckdb"]
    # Query selection & dataset
    queries: list[int]
    query_set: str
    dataset_path: Path
    scale_factor: int | float
    suffix: str
    qualification: bool = False

    # Execution mode
    frontend: Literal["dask", "duckdb", "in-memory", "polars-cpu", "ray", "spmd"]
    connect: str | None = None
    num_gpus: int | None = None

    # Run parameters
    iterations: int
    io_mode: Literal["cold", "lukewarm", "hot"] = "lukewarm"
    collect_traces: bool = False
    # All streaming/rapidsmpf/engine knobs
    streaming_options: StreamingOptions = dataclasses.field(
        default_factory=lambda: __import__(
            "cudf_polars.engine.options",
            fromlist=["StreamingOptions"],
        ).StreamingOptions()
    )

    # Validation
    validation_method: ValidationMethod | None = None

    # DuckDB configuration
    duckdb_threads: int | None = None
    duckdb_memory_limit: str | None = None
    duckdb_temp_dir: str | None = None

    # Metadata / output (populated at runtime)
    n_workers: int = 1
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[SuccessRecord | FailedRecord]] = dataclasses.field(
        default_factory=dict
    )
    plans: dict[int, Any] = dataclasses.field(default_factory=dict)
    hardware: HardwareInfo = dataclasses.field(default_factory=HardwareInfo.collect)
    run_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    command_line: str
    capture_env_vars: str
    roles: list[Role] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:  # noqa: D105
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
        from cudf_polars.engine.options import StreamingOptions

        streaming_options = StreamingOptions._from_argparse(args)

        path = args.path
        name = args.query_set
        scale_factor = args.scale

        if args.qualification and "pdsds" not in name:
            raise ValueError("--qualification can only be used with PDS-DS benchmarks.")

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
                comparison_method="polars",
                comparison_options=get_validation_options(args),
                expected_location=args.validate_directory,
            )
        elif args.validate_against is not None:
            validation_method = ValidationMethod(
                args.validate_against,
                comparison_method="polars",
                comparison_options=get_validation_options(args),
                expected_location=None,
            )
        else:
            validation_method = None

        engine_name: Literal["polars-cpu", "cudf-polars", "duckdb"]
        if args.frontend == "duckdb":
            engine_name = "duckdb"
        elif args.frontend == "polars-cpu":
            engine_name = "polars-cpu"
        else:
            engine_name = "cudf-polars"

        roles: list[Role] = []
        if args.role_nightly:
            roles.append(NightlyRole())
        if args.role_nsys:
            roles.append(NsysRole())

        return cls(
            engine_name=engine_name,
            queries=args.query,
            query_set=name,
            dataset_path=path,
            scale_factor=scale_factor,
            suffix=args.suffix,
            qualification=args.qualification,
            frontend=args.frontend,
            iterations=args.iterations,
            io_mode=args.io_mode,
            collect_traces=args.collect_traces,
            streaming_options=streaming_options,
            connect=args.connect,
            num_gpus=args.num_gpus,
            validation_method=validation_method,
            extra_info=args.extra_info,
            duckdb_threads=args.duckdb_threads,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_temp_dir=args.duckdb_temp_dir,
            command_line=shlex.join(sys.argv),
            capture_env_vars=args.capture_env_vars,
            hardware=HardwareInfo.collect(
                collect_gpus=args.frontend not in _CPU_ENGINES
            ),
            roles=roles,
        )

    def serialize(self, engine: StreamingEngine | None) -> dict:
        """Serialize the run config to a dictionary."""
        opts = self.streaming_options
        result: dict[str, Any] = {
            "engine_name": self.engine_name,
            "queries": self.queries,
            "query_set": self.query_set,
            "dataset_path": str(self.dataset_path),
            "scale_factor": self.scale_factor,
            "suffix": self.suffix,
            "qualification": self.qualification,
            "frontend": self.frontend,
            "iterations": self.iterations,
            "io_mode": self.io_mode,
            "collect_traces": self.collect_traces,
            "n_workers": self.n_workers,
            "extra_info": self.extra_info,
            "run_id": str(self.run_id),
            "timestamp": self.timestamp,
            "command_line": self.command_line,
            "streaming_options": {
                "rapidsmpf": opts.to_rapidsmpf_options().get_strings(),
                "executor": opts.to_executor_options(),
                "engine": {k: str(v) for k, v in opts.to_engine_options().items()},
            },
            "records": {
                k: [dataclasses.asdict(r) for r in v] for k, v in self.records.items()
            },
            "plans": {},
            "versions": dataclasses.asdict(self.versions),
            "hardware": dataclasses.asdict(self.hardware),
            "validation_method": dataclasses.asdict(self.validation_method)
            if self.validation_method
            else None,
            "roles": [dataclasses.asdict(r) for r in self.roles],
        }
        if engine is not None:
            config_options = ConfigOptions.from_polars_engine(engine)
            config_options = config_options.drop_unserializable()
            rapidsmpf_options = engine.rapidsmpf_options.get_strings()
            result["config_options"] = {
                "config_options": dataclasses.asdict(
                    config_options, dict_factory=ConfigOptions.dict_factory
                ),
                "rapidsmpf_options": rapidsmpf_options,
            }
            # discard unserializable / unnecessary UUIDs
            result["config_options"]["config_options"]["executor"].pop(
                "quent_context", None
            )

        return result

    def summarize(self) -> None:
        """Print a summary of the results."""
        print("Iteration Summary")
        print("=======================================")

        total_mean_time = 0.0
        for query, records in self.records.items():
            print(f"query: {query}")
            print(f"path: {self.dataset_path}")
            print(f"scale_factor: {self.scale_factor}")
            print(f"frontend: {self.frontend}")
            if self.frontend in _STREAMING_FRONTENDS:
                opts = self.streaming_options.to_executor_options()
                print(f"n_workers: {self.n_workers}")
                print(f"target_partition_size: {opts.get('target_partition_size')}")
                print(f"broadcast_limit: {opts.get('broadcast_limit')}")
                print(f"dynamic_planning: {opts.get('dynamic_planning', 'default')}")
            valid_durations = [
                record.duration for record in records if record.status == "success"
            ]
            if len(valid_durations) > 0:
                mean_time = mean(valid_durations)
                total_mean_time += mean_time
                print(f"iterations: {self.iterations}")
                print("---------------------------------------")
                print(f"min time : {min(valid_durations):0.4f}")
                print(f"max time : {max(valid_durations):0.4f}")
                print(f"mean time: {mean_time:0.4f}")
                print("=======================================")

        if total_mean_time > 0:
            print(f"Total mean time across all queries: {total_mean_time:.4f} seconds")
        else:
            print("No successful queries")


def get_executor_options(
    run_config: RunConfig, benchmark: Any = None
) -> dict[str, Any]:
    """Generate executor_options for GPUEngine."""
    executor_options: dict[str, Any] = (
        run_config.streaming_options.to_executor_options()
    )
    executor_options["quent_context"] = cudf_polars.quent.QuentContext(
        engine=cudf_polars.quent.Engine(id=run_config.run_id)
    )

    return executor_options


def print_query_plan(
    q_id: int,
    q: pl.LazyFrame,
    args: argparse.Namespace,
    run_config: RunConfig,
    engine: None | pl.GPUEngine = None,
    *,
    print_plans: bool = True,
) -> tuple[str | None, str | None]:
    """Print the query plan."""
    logical_plan = plan = None
    if run_config.frontend == "polars-cpu":
        if args.explain_logical:
            logical_plan = q.explain()
        if args.explain:
            plan = q.show_graph(engine="streaming", plan_stage="physical")
    elif CUDF_POLARS_AVAILABLE:
        assert isinstance(engine, pl.GPUEngine)
        if args.explain_logical:
            logical_plan = explain_query(q, engine, physical=False)
        if args.explain and run_config.frontend in _STREAMING_FRONTENDS:
            plan = explain_query(q, engine)
    else:
        raise RuntimeError(
            "Cannot provide the logical or physical plan because cudf_polars is not installed."
        )

    if print_plans:
        if logical_plan:
            print(f"\nQuery {q_id} - Logical plan\n")
            print(logical_plan)
        if plan:
            print(f"\nQuery {q_id} - Physical plan\n")
            print(plan)

    return logical_plan, plan


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
    q: pl.LazyFrame,
    run_config: RunConfig,
    args: argparse.Namespace,
    engine: None | pl.GPUEngine = None,
) -> tuple[pl.DataFrame, float]:
    """Execute a query with NVTX annotation."""
    if run_config.io_mode == "cold":
        drop_file_page_cache_recursively(run_config.dataset_path)

    with nvtx.annotate(
        message=f"Query {q_id} - Iteration {i}",
        domain="cudf_polars",
        color="green",
    ):
        if run_config.frontend == "polars-cpu":
            t0 = time.monotonic()
            result = q.collect(engine="streaming")
            t1 = time.monotonic()

        elif CUDF_POLARS_AVAILABLE:
            assert isinstance(engine, pl.GPUEngine)
            if args.debug:
                translator = Translator(q._ldf.visit(), engine)
                ir = translator.translate_ir()
                context = IRExecutionContext()
                if run_config.frontend == "in-memory":
                    t0 = time.monotonic()
                    result = ir.evaluate(
                        cache={}, timer=None, context=context
                    ).to_polars()
                    t1 = time.monotonic()
                elif run_config.frontend in _STREAMING_FRONTENDS:
                    t0 = time.monotonic()
                    result = evaluate_streaming(
                        ir,
                        translator.config_options,
                    )
                    t1 = time.monotonic()
                else:
                    raise ValueError(
                        f"--debug is not supported with --frontend {run_config.frontend}"
                    )
            else:
                t0 = time.monotonic()
                result = q.collect(engine=engine)
                t1 = time.monotonic()

        else:
            raise RuntimeError("The requested engine is not supported.")

        return result, t1 - t0


def validate_result(
    result: pl.DataFrame,
    expected: pl.DataFrame,
    sort_by: list[tuple[str, bool]],
    limit: int | None = None,
    sort_keys: list[tuple[pl.Expr, bool]] | None = None,
    **kwargs: Any,
) -> ValidationResult:
    """
    Validate the computed result against the expected answer.

    This takes care of special handling for validating TPC-H queries,
    where multiple results might be considered correct.

    See Also
    --------
    cudf_polars.testing.asserts.assert_tpch_result_equal
    """
    try:
        assert_tpch_result_equal(
            result,
            expected,
            sort_by=sort_by,
            limit=limit,
            sort_keys=sort_keys,
            **kwargs,
        )
    except Exception as e:
        return ValidationResult.from_error(e)
    else:
        return ValidationResult(status="Passed", message=None)


@dataclasses.dataclass
class QueryResult:
    """
    Representation of a query's result.

    Parameters
    ----------
    frame: pl.LazyFrame
        The result of the query.
    sort_by: list[tuple[str, bool]]
        The columns that the query sorts by. Each tuple contains (column_name, descending_flag).
        Used for the ties/limit boundary logic in validation.
    sort_keys: list[tuple[pl.Expr, bool]] | None
        Optional Polars expressions for the sortedness check. Use this when the query
        sorts by a conditional expression (e.g. ``CASE WHEN lochierarchy = 0 THEN i_category END``)
        that cannot be represented as a plain column name in ``sort_by``. When provided,
        these expressions are evaluated against the output and used only for the sortedness
        check; ``sort_by`` still drives the ties/limit boundary logic.
    limit: int | None
        The limit of the query, if any.

    """

    frame: pl.LazyFrame
    sort_by: list[tuple[str, bool]]
    limit: int | None = None
    nulls_last: bool = True
    sort_keys: list[tuple[pl.Expr, bool]] | None = None


def _collect_statistics(engine: pl.GPUEngine | None) -> dict[str, Any] | None:
    """Gather + clear per-rank rapidsmpf statistics into a merged dict."""
    if engine is None:
        return None
    if not isinstance(engine, StreamingEngine):
        return None
    return engine.global_statistics(clear=True).to_dict()


def run_polars_query_iteration(
    q_id: int,
    iteration: int,
    q: pl.LazyFrame,
    run_config: RunConfig,
    args: argparse.Namespace,
    engine: pl.GPUEngine | None,
    expected: pl.DataFrame | None,
    query_result: Any,
    prepare_validation_result: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    result_casts: list[pl.Expr] | None = None,
) -> SuccessRecord:
    """Run a single query iteration. Caller must wrap in try/except."""
    result, duration = execute_query(q_id, iteration, q, run_config, args, engine)

    if expected is not None and prepare_validation_result is not None:
        result = prepare_validation_result(result)

    if expected is not None and result_casts:
        # Applying the casts to the polars result is
        # a workaround we need because of a polars bug
        # See https://github.com/pola-rs/polars/issues/27269
        # Once we support polars 1.40, we should remove this
        result = result.with_columns(*result_casts)

    statistics = _collect_statistics(engine)

    if expected is not None:
        validation_result = validate_result(
            result,
            expected,
            query_result.sort_by,
            limit=query_result.limit,
            nulls_last=query_result.nulls_last,
            sort_keys=query_result.sort_keys,
            **get_validation_options(args),
        )
    else:
        validation_result = None

    if args.print_results:
        print(result)

    if args.results_directory is not None and iteration == 0:
        results_dir = Path(args.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"q_{q_id:02d}.parquet"
        result.write_parquet(output_path)

    return SuccessRecord(
        query=q_id,
        iteration=iteration,
        duration=duration,
        statistics=statistics,
        validation_result=validation_result,
    )


def run_polars_query(
    q_id: int,
    benchmark: Any,
    run_config: RunConfig,
    args: argparse.Namespace,
    engine: pl.GPUEngine | None,
    numeric_type: str,
    date_type: str,
    prepare_validation_result: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> QueryRunResult:
    """Run all iterations for a single query. Caller must wrap in try/except."""
    query_result: QueryResult = getattr(benchmark, f"q{q_id}")(run_config)
    q = query_result.frame

    print_query_plan(q_id, q, args, run_config, engine, print_plans=args.print_plans)
    plan = None
    if (args.explain or args.explain_logical) and engine is not None:
        from cudf_polars.streaming.explain import serialize_query

        plan = serialize_query(q, engine)

    part_plan_rows = []
    if (
        getattr(args, "explain_partition_plan", False)
        and engine is not None
        and run_config.frontend in _STREAMING_FRONTENDS
    ):
        from cudf_polars.streaming.explain import collect_partition_plan

        part_plan_rows = collect_partition_plan(q, engine, q_id)

    casts = benchmark.EXPECTED_CASTS.get(q_id, [])
    if numeric_type == "decimal":
        casts.extend(benchmark.EXPECTED_CASTS_DECIMAL.get(q_id, []))
    if date_type == "timestamp":
        casts.extend(benchmark.EXPECTED_CASTS_TIMESTAMP.get(q_id, []))

    expected: pl.DataFrame | None = None
    validation_method = run_config.validation_method
    if validation_method is not None:
        match validation_method.expected_source:
            case "polars-cpu":
                expected = q.collect()
            case "duckdb":
                duckdb_queries_cls = benchmark().duckdb_queries
                get_ddb = getattr(duckdb_queries_cls, f"q{q_id}")
                base_sql = get_ddb(run_config)
                expected = execute_duckdb_query(
                    base_sql,
                    run_config.dataset_path,
                    query_set=duckdb_queries_cls.name,
                    suffix=run_config.suffix,
                    run_config=run_config,
                ).with_columns(*casts)
            case "duckdb-disk":
                expected = pl.read_parquet(
                    validation_method.expected_file(q_id)
                ).with_columns(*casts)
            case baseline:
                raise ValueError(f"Invalid baseline: {baseline}")

    if args.output_expected_directory is not None:
        assert expected is not None, (
            "Expected result must be computed before writing to disk."
        )
        expected_dir = Path(args.output_expected_directory)
        expected_dir.mkdir(parents=True, exist_ok=True)
        expected.write_parquet(expected_dir / f"q_{q_id:02d}.parquet")

    query_records: list[SuccessRecord | FailedRecord] = []
    iteration_failures: list[tuple[int, int]] = []
    validation_failed = False
    record: SuccessRecord | FailedRecord

    for i in range(args.iterations):
        if _HAS_STRUCTLOG and run_config.collect_traces:
            setup_logging(q_id, i)
            if isinstance(engine, StreamingEngine):
                quent_context = engine.config["executor_options"].get("quent_context")
                if quent_context is not None:
                    engine.config["executor_options"]["quent_context"] = (
                        dataclasses.replace(
                            quent_context,
                            query=cudf_polars.quent.Query(
                                instance_name=f"Iteration {i + 1}",
                            ),
                        )
                    )
                    engine._run(setup_logging, q_id, i)

        try:
            record = run_polars_query_iteration(
                q_id=q_id,
                iteration=i,
                q=q,
                run_config=run_config,
                args=args,
                engine=engine,
                expected=expected,
                query_result=query_result,
                prepare_validation_result=prepare_validation_result,
                result_casts=casts if casts else None,
            )
        except Exception:
            print(f"❌ query={q_id} iteration={i} failed!")
            print(traceback.format_exc())
            iteration_failures.append((q_id, i))
            record = FailedRecord(
                query=q_id,
                iteration=i,
                status="error",
                traceback=traceback.format_exc(),
            )

        else:
            if record.validation_result and record.validation_result.status == "Failed":
                validation_failed = True
                print(
                    f"❌ Query {q_id} failed validation!\n{record.validation_result.message}"
                )
                if record.validation_result.details:
                    pprint.pprint(record.validation_result.details)
            else:
                prefix = "✅ " if record.validation_result else ""
                print(
                    f"{prefix}Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s",
                    flush=True,
                )

        query_records.append(record)

    return QueryRunResult(
        query_records=query_records,
        plan=plan,
        iteration_failures=iteration_failures,
        validation_failed=validation_failed,
        partition_plan_rows=part_plan_rows,
    )


def _run_query_loop(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: RunConfig,
    engine: pl.GPUEngine | None,
    numeric_type: str,
    date_type: str,
    prepare_validation_result: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> tuple[
    defaultdict[int, list[SuccessRecord | FailedRecord]],
    dict[int, Any],
    list[int],
    list[tuple[int, int]],
]:
    """Execute all queries in ``run_config`` and return accumulated results."""
    records: defaultdict[int, list[SuccessRecord | FailedRecord]] = defaultdict(list)
    plans: dict[int, SerializablePlan] = {}
    validation_failures: list[int] = []
    query_failures: list[tuple[int, int]] = []
    all_partition_plan_rows: list = []

    for q_id in run_config.queries:
        if engine is not None:
            quent_context = engine.config["executor_options"].get("quent_context")
            if quent_context is not None:
                engine.config["executor_options"]["quent_context"] = (
                    dataclasses.replace(
                        quent_context,
                        query_group=cudf_polars.quent.QueryGroup(
                            instance_name=f"PDSH Query {q_id}",
                        ),
                    )
                )

        try:
            result = run_polars_query(
                q_id=q_id,
                benchmark=benchmark,
                run_config=run_config,
                args=args,
                engine=engine,
                numeric_type=numeric_type,
                date_type=date_type,
                prepare_validation_result=prepare_validation_result,
            )
        except Exception:
            print(f"❌ query={q_id} failed (setup or execution)!")
            print(traceback.format_exc())
            query_failures.append((q_id, -1))
            record = FailedRecord(
                query=q_id,
                iteration=-1,
                traceback=traceback.format_exc(),
            )
            result = QueryRunResult(
                query_records=[record],
                plan=None,
                iteration_failures=[],
                validation_failed=False,
            )

        records[q_id] = result.query_records
        if result.plan is not None:
            plans[q_id] = result.plan
        query_failures.extend(result.iteration_failures)
        if result.validation_failed:
            validation_failures.append(q_id)
        all_partition_plan_rows.extend(result.partition_plan_rows)

    if all_partition_plan_rows and getattr(args, "explain_partition_plan", False):
        from cudf_polars.streaming.explain import format_partition_plan_table

        print(format_partition_plan_table(all_partition_plan_rows), flush=True)

    return records, plans, validation_failures, query_failures


def _finalize_benchmark_run(
    args: argparse.Namespace,
    run_config: RunConfig,
    validation_failures: list[int],
    query_failures: list[tuple[int, int]],
    serializable_engine_config: dict[str, Any],
) -> None:
    """Summarize, serialize, and exit after a benchmark run."""
    if args.summarize:
        run_config.summarize()
    if (
        run_config.validation_method is not None
        and run_config.frontend not in _CPU_ENGINES
    ):
        print("\nValidation Summary")
        print("==================")
        if validation_failures:
            print(
                f"{len(validation_failures)} queries failed validation: "
                f"{sorted(set(validation_failures))}"
            )
        else:
            print("✅ All validated queries passed.")
    args.output.write(json.dumps(serializable_engine_config))
    args.output.write("\n")
    sys.exit(1 if (query_failures or validation_failures) else 0)


def run_polars_cpu(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    numeric_type: str,
    date_type: str,
) -> None:
    """Run benchmark queries using the Polars CPU streaming engine."""
    records, plans, validation_failures, query_failures = _run_query_loop(
        benchmark,
        args,
        run_config,
        engine=None,
        numeric_type=numeric_type,
        date_type=date_type,
    )
    run_config = dataclasses.replace(run_config, records=dict(records), plans=plans)
    _finalize_benchmark_run(
        args,
        run_config,
        validation_failures,
        query_failures,
        serializable_engine_config=run_config.serialize(engine=None),
    )


def run_polars_in_memory(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    parquet_options: dict[str, Any],
    numeric_type: str,
    date_type: str,
) -> None:
    """Run benchmark queries using a single-process GPU in-memory engine."""
    engine_options = {
        **run_config.streaming_options.to_engine_options(),
        "parquet_options": parquet_options,
    }
    engine_options.setdefault("raise_on_fail", True)
    engine = pl.GPUEngine(
        executor="in-memory",
        **engine_options,
    )
    records, plans, validation_failures, query_failures = _run_query_loop(
        benchmark,
        args,
        run_config,
        engine=engine,
        numeric_type=numeric_type,
        date_type=date_type,
    )
    run_config = dataclasses.replace(run_config, records=dict(records), plans=plans)
    run_config = _consolidate_logs(run_config, engine=None)
    _finalize_benchmark_run(
        args,
        run_config,
        validation_failures,
        query_failures,
        serializable_engine_config=run_config.serialize(engine=engine),
    )


def run_polars_spmd(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    parquet_options: dict[str, Any],
    numeric_type: str,
    date_type: str,
) -> None:
    """Run benchmark queries using SPMD execution via the ``rrun`` launcher."""
    from cudf_polars.engine.spmd import SPMDEngine

    executor_options = get_executor_options(run_config, benchmark=benchmark)
    # "cluster" is reserved — SPMDEngine sets it
    executor_options.pop("cluster", None)
    engine_options = {
        **run_config.streaming_options.to_engine_options(),
        "parquet_options": parquet_options,
    }
    engine_options.setdefault("raise_on_fail", True)
    with SPMDEngine(
        rapidsmpf_options=run_config.streaming_options.to_rapidsmpf_options(),
        executor_options=executor_options,
        engine_options=engine_options,
    ) as engine:
        from cudf_polars.engine.spmd import (
            allgather_polars_dataframe,
        )
        from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id

        is_rank_0 = engine.rank == 0

        def _allgather_result(df: pl.DataFrame) -> pl.DataFrame:
            with reserve_op_id() as op_id:
                return allgather_polars_dataframe(
                    engine=engine,
                    local_df=df,
                    op_id=op_id,
                )

        run_config = dataclasses.replace(run_config, n_workers=engine.nranks)
        records, plans, validation_failures, query_failures = _run_query_loop(
            benchmark,
            args,
            run_config,
            engine,
            numeric_type,
            date_type,
            prepare_validation_result=_allgather_result,
        )
        if engine.rank > 0:
            sys.exit(1 if (query_failures or validation_failures) else 0)
        run_config = dataclasses.replace(run_config, records=dict(records), plans=plans)
        run_config = _consolidate_logs(
            run_config, engine=engine, gather_client_logs=False
        )
        # We need to create this before StreamingEngine.shutdown(), which clears engine.config
        serializable_engine_config = run_config.serialize(engine=engine)

    if is_rank_0:
        _write_quent_traces(
            engine=engine,
            run_id=run_config.run_id,
            collect_traces=run_config.collect_traces,
        )
    _finalize_benchmark_run(
        args,
        run_config,
        validation_failures,
        query_failures,
        serializable_engine_config=serializable_engine_config,
    )


def run_polars_ray(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    parquet_options: dict[str, Any],
    numeric_type: str,
    date_type: str,
) -> None:
    """Run benchmark queries using Ray actor-based distributed execution."""
    from cudf_polars.engine.ray import RayEngine

    executor_options = get_executor_options(run_config, benchmark=benchmark)
    # "cluster" is reserved — RayEngine sets it
    executor_options.pop("cluster", None)
    engine_options: dict[str, Any] = {
        **run_config.streaming_options.to_engine_options(),
        "parquet_options": parquet_options,
    }
    engine_options.setdefault("raise_on_fail", True)
    ray_init_options: dict[str, Any] = {}
    if run_config.connect is not None:
        ray_init_options["address"] = run_config.connect
    if run_config.num_gpus is not None:
        ray_init_options["num_gpus"] = run_config.num_gpus

    with RayEngine(
        rapidsmpf_options=run_config.streaming_options.to_rapidsmpf_options(),
        executor_options=executor_options,
        engine_options=engine_options,
        ray_init_options=ray_init_options,
    ) as engine:
        run_config = dataclasses.replace(run_config, n_workers=engine.nranks)
        records, plans, validation_failures, query_failures = _run_query_loop(
            benchmark,
            args,
            run_config,
            engine,
            numeric_type,
            date_type,
        )
        run_config = dataclasses.replace(run_config, records=dict(records), plans=plans)
        run_config = _consolidate_logs(run_config, engine=engine)
        # We need to create this before StreamingEngine.shutdown(), which clears engine.config
        serializable_engine_config = run_config.serialize(engine=engine)

    _write_quent_traces(
        engine=engine,
        run_id=run_config.run_id,
        collect_traces=run_config.collect_traces,
    )
    _finalize_benchmark_run(
        args,
        run_config,
        validation_failures,
        query_failures,
        serializable_engine_config=serializable_engine_config,
    )


def run_polars_dask(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    parquet_options: dict[str, Any],
    numeric_type: str,
    date_type: str,
) -> None:
    """Run benchmark queries using Dask distributed execution."""
    import distributed

    from cudf_polars.engine.dask import DaskEngine

    executor_options = get_executor_options(run_config, benchmark=benchmark)
    # "cluster" is reserved — DaskEngine sets it
    executor_options.pop("cluster", None)
    engine_options: dict[str, Any] = {
        **run_config.streaming_options.to_engine_options(),
        "parquet_options": parquet_options,
    }
    engine_options.setdefault("raise_on_fail", True)
    dask_client = None
    if run_config.connect is not None:
        if Path(run_config.connect).is_file():
            dask_client = distributed.Client(scheduler_file=run_config.connect)
        else:
            dask_client = distributed.Client(address=run_config.connect)

    if run_config.num_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(run_config.num_gpus)
        )

    try:
        with DaskEngine(
            rapidsmpf_options=run_config.streaming_options.to_rapidsmpf_options(),
            executor_options=executor_options,
            engine_options=engine_options,
            dask_client=dask_client,
        ) as engine:
            run_config = dataclasses.replace(run_config, n_workers=engine.nranks)
            records, plans, validation_failures, query_failures = _run_query_loop(
                benchmark, args, run_config, engine, numeric_type, date_type
            )
            run_config = dataclasses.replace(
                run_config, records=dict(records), plans=plans
            )
            run_config = _consolidate_logs(run_config, engine)
            # We need to create this before StreamingEngine.shutdown(), which clears engine.config
            serializable_engine_config = run_config.serialize(engine=engine)

        _write_quent_traces(
            engine=engine,
            run_id=run_config.run_id,
            collect_traces=run_config.collect_traces,
        )
    finally:
        if dask_client is not None:
            dask_client.close()
    _finalize_benchmark_run(
        args,
        run_config,
        validation_failures,
        query_failures,
        serializable_engine_config=serializable_engine_config,
    )


def setup_logging(query_id: int, iteration: int) -> None:
    if not cudf_polars.dsl.tracing.LOG_TRACES:
        msg = (
            "Tracing requested via --collect-traces, but tracking is not enabled. "
            "Verify that 'CUDF_POLARS_LOG_TRACES' is set and structlog is installed."
        )
        raise RuntimeError(msg)

    if _HAS_STRUCTLOG:
        # structlog uses contextvars to propagate context down to where log records
        # are emitted. Ideally, we'd just set the contextvars here using
        # structlog.bind_contextvars; for the distributed cluster we would need
        # to use something like client.run to set the contextvars on the worker.
        # However, there's an unfortunate conflict between structlog's use of
        # context vars and how Dask Workers actually execute tasks, such that
        # the contextvars set via `client.run` aren't visible to the actual
        # tasks.
        #
        # So instead we make a new logger each time we need a new context,
        # i.e. for each query/iteration pair.

        def make_injector(
            query_id: int, iteration: int
        ) -> Callable[[logging.Logger, str, dict[str, Any]], dict[str, Any]]:
            def inject(
                logger: Any, method_name: Any, event_dict: Any
            ) -> dict[str, Any]:
                event_dict["query_id"] = query_id
                event_dict["iteration"] = iteration
                return event_dict

            return inject

        shared_processors = [
            structlog.contextvars.merge_contextvars,
            make_injector(query_id, iteration),
            structlog.processors.add_log_level,
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.PROCESS,
                    structlog.processors.CallsiteParameter.THREAD,
                ],
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=False),
        ]

        # For logging to a file
        json_renderer = structlog.processors.JSONRenderer()

        stream = io.StringIO()
        json_file_handler = logging.StreamHandler(stream)
        json_file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=json_renderer,
                foreign_pre_chain=shared_processors,
            )
        )

        logging.basicConfig(level=logging.INFO, handlers=[json_file_handler])

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            cache_logger_on_first_use=True,
        )


def _write_quent_traces(
    engine: StreamingEngine, run_id: uuid.UUID, *, collect_traces: bool
) -> None:
    """Write collected Quent events to logs/{run_id}.ndjson."""
    if not (_HAS_STRUCTLOG or collect_traces):
        return

    quent_logs = list(engine._quent_events)

    # The quent UI currently requires the filename to match the engine's ID.
    for log in quent_logs:
        if log.get("data", {}).get("Engine", {}).get("Init") and log.get("id") != str(
            run_id
        ):
            msg = (
                f"Engine ID mismatch: Quent ID ({log['id']}) != Run ID ({run_id}). "
                "The data might not load in the Quent UI."
            )
            warnings.warn(msg, stacklevel=2)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / f"{run_id}.ndjson"
    with output_path.open("w") as f:
        for log in quent_logs:
            f.write(json.dumps(log))
            f.write("\n")
    print(f"Wrote {len(quent_logs)} Quent trace events to {output_path}")


def _consolidate_logs(
    run_config: RunConfig,
    engine: StreamingEngine | None,
    *,
    gather_client_logs: bool = True,
) -> RunConfig:
    """
    Gather structlog traces and attach them to ``run_config.records``.

    Parameters
    ----------
    run_config
        The benchmark run config to augment.
    engine
        The streaming engine to fan out the gather across (dask / ray / spmd).
        Pass ``None`` for single-process frontends (e.g. in-memory), only the
        local-process buffer is collected.
    gather_client_logs
        When ``engine`` is not ``None``, also include the client-side
        local-process buffer. Set to ``False`` for SPMD, where rank-0 is
        itself a worker (so the worker fan-out already covered it). Ignored
        when ``engine`` is ``None``.

    Returns
    -------
    The augmented ``run_config``.
    """
    if not (_HAS_STRUCTLOG and run_config.collect_traces):
        return run_config

    def gather_logs() -> str:
        logger = logging.getLogger()
        return logger.handlers[0].stream.getvalue()  # type: ignore[attr-defined]

    parts: list[str] = []
    if engine is not None:
        parts.append("\n".join(engine._run(gather_logs)))
    if engine is None or gather_client_logs:
        parts.append(gather_logs())
    all_logs = "\n".join(parts)

    parsed_logs = [json.loads(log) for log in all_logs.splitlines() if log]
    # Some other log records can end up in here. Filter those out.
    scope_values = {s.value for s in Scope}
    parsed_logs = [log for log in parsed_logs if log.get("scope") in scope_values]
    # Now we want to augment the existing Records with the trace data.

    def group_key(x: dict) -> int:
        return x["query_id"]

    def sort_key(x: dict) -> tuple[int, int]:
        return x["query_id"], x["iteration"]

    grouped = itertools.groupby(
        sorted(parsed_logs, key=sort_key),
        key=group_key,
    )

    for query_id, run_logs_group in grouped:
        traces_by_iteration: dict[int, list[dict[str, Any]]] = {
            iteration: list(group)
            for iteration, group in itertools.groupby(
                run_logs_group, key=lambda x: x["iteration"]
            )
        }
        run_records = run_config.records[query_id]

        new_records: list[SuccessRecord | FailedRecord] = []
        for rec in run_records:
            traces = traces_by_iteration.get(rec.iteration)
            if rec.status == "success" and traces is not None:
                new_records.append(dataclasses.replace(rec, traces=traces))
            else:
                new_records.append(rec)

        run_config.records[query_id] = new_records

    return run_config


PDSDS_TABLE_NAMES: list[str] = [
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
]

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


def print_duckdb_plan(
    q_id: int,
    sql: str,
    dataset_path: Path,
    suffix: str,
    query_set: str,
    args: argparse.Namespace,
    run_config: RunConfig | None = None,
) -> None:
    """Print DuckDB query plan using EXPLAIN."""
    if duckdb is None:
        raise ImportError(duckdb_err)

    if query_set == "pdsds":
        tbl_names = PDSDS_TABLE_NAMES
    else:
        tbl_names = PDSH_TABLE_NAMES

    with duckdb.connect(config=_make_duckdb_config(run_config)) as conn:
        for name in tbl_names:
            pattern = (Path(dataset_path) / name).as_posix() + suffix
            conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM parquet_scan('{pattern}');"
            )

        if args.explain_logical and args.explain:
            conn.execute("PRAGMA explain_output = 'all';")
        elif args.explain_logical:
            conn.execute("PRAGMA explain_output = 'optimized_only';")
        else:
            conn.execute("PRAGMA explain_output = 'physical_only';")

        print(f"\nDuckDB Query {q_id} - Plan\n")

        plan_rows = conn.execute(f"EXPLAIN {sql}").fetchall()
        for _, line in plan_rows:
            print(line)


def execute_duckdb_query(
    query: str,
    dataset_path: Path,
    *,
    suffix: str = ".parquet",
    query_set: str = "pdsh",
    run_config: RunConfig | None = None,
) -> pl.DataFrame:
    """Execute a query with DuckDB."""
    if duckdb is None:
        raise ImportError(duckdb_err)
    if query_set == "pdsds":
        tbl_names = PDSDS_TABLE_NAMES
    else:
        tbl_names = PDSH_TABLE_NAMES
    with duckdb.connect(config=_make_duckdb_config(run_config)) as conn:
        for name in tbl_names:
            pattern = (Path(dataset_path) / name).as_posix() + suffix
            conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM parquet_scan('{pattern}');"
            )
        return conn.execute(query).pl()


def run_duckdb(duckdb_queries_cls: Any, args: argparse.Namespace) -> None:
    """Run the benchmark with DuckDB."""
    vars(args).update({"query_set": duckdb_queries_cls.name})
    run_config = RunConfig.from_args(args)
    records: defaultdict[int, list[SuccessRecord | FailedRecord]] = defaultdict(list)

    for q_id in run_config.queries:
        try:
            get_q = getattr(duckdb_queries_cls, f"q{q_id}")
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        sql = get_q(run_config)

        if args.explain or args.explain_logical:
            print_duckdb_plan(
                q_id=q_id,
                sql=sql,
                dataset_path=run_config.dataset_path,
                suffix=run_config.suffix,
                query_set=duckdb_queries_cls.name,
                args=args,
                run_config=run_config,
            )

        print(f"DuckDB Executing: {q_id}")
        records[q_id] = []

        for i in range(args.iterations):
            if run_config.io_mode == "cold":
                drop_file_page_cache_recursively(run_config.dataset_path)
            t0 = time.time()
            result = execute_duckdb_query(
                sql,
                run_config.dataset_path,
                suffix=run_config.suffix,
                query_set=duckdb_queries_cls.name,
                run_config=run_config,
            )
            t1 = time.time()
            record = SuccessRecord(query=q_id, iteration=i, duration=t1 - t0)
            if args.print_results:
                print(result)
            print(f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s")
            records[q_id].append(record)
            if i == 0 and args.output_expected_directory is not None:
                expected_dir = Path(args.output_expected_directory)
                expected_dir.mkdir(parents=True, exist_ok=True)
                result.write_parquet(expected_dir / f"q_{q_id:02d}.parquet")

    run_config = dataclasses.replace(run_config, records=dict(records))
    if args.summarize:
        run_config.summarize()

    args.output.write(json.dumps(run_config.serialize(engine=None)))
    args.output.write("\n")


def check_input_data_type(
    run_config: RunConfig,
) -> tuple[Literal["decimal", "float"], Literal["date", "timestamp"]]:
    """
    Check the input data types for columns with variable data types.

    Our queries might be run on datasets that use different data types for different
    types of columns. Our validation supports:

    1. 'decimal' or 'float' for non-integer numeric columns (e.g. 'c_acctbal')
    2. 'date' or 'timestamp' for date type columns (e.g. 'o_orderdate')

    For PDS-H, this is determined by the ``c_acctbal`` column in the customer table.
    For PDS-DS, we use ``i_current_price`` from the item table.
    """
    if run_config.query_set == "pdsds":
        table, col = "item", "i_current_price"
    else:
        table, col = "customer", "c_acctbal"
    t = (
        get_data(run_config.dataset_path, table, run_config.suffix)
        .select(pl.col(col))
        .collect_schema()[col]
    )

    num_type: Literal["decimal", "float"]
    date_type: Literal["date", "timestamp"]
    if t.is_decimal():
        num_type = "decimal"
    else:
        num_type = "float"

    if run_config.query_set == "pdsds":
        date_type = "date"
    else:
        t = (
            get_data(run_config.dataset_path, "orders", run_config.suffix)
            .select(pl.col("o_orderdate"))
            .collect_schema()["o_orderdate"]
        )
        if t.to_python().__name__ == "date":
            date_type = "date"
        else:
            date_type = "timestamp"

    return num_type, date_type


def _query_type(num_queries: int) -> Any:
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


def build_parser(num_queries: int = 22) -> argparse.ArgumentParser:
    """Build the argument parser for PDS-H/PDS-DS benchmarks."""
    from cudf_polars.engine.options import StreamingOptions

    parser = argparse.ArgumentParser(
        prog="Cudf-Polars PDS-H/PDS-DS Benchmarks",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=_query_type(num_queries),
        help=textwrap.dedent("""\
            Query to run. One of the following:
            - A single number (e.g. 11)
            - A comma-separated list of query numbers (e.g. 1,3,7)
            - A range of query numbers (e.g. 1-11,23-34)
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
        "--qualification",
        action="store_true",
        help="Use TPC-DS qualification parameters from specification Appendix B (PDS-DS only).",
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
        required=True,
        type=str,
        choices=["dask", "duckdb", "in-memory", "polars-cpu", "ray", "spmd"],
        help=textwrap.dedent("""\
            Execution frontend:
                - dask       : Dask distributed multi-GPU execution
                - duckdb     : DuckDB CPU execution
                - in-memory  : Single-process GPU, in-memory evaluation
                - polars-cpu : Polars CPU streaming engine (no GPU)
                - ray        : Ray actor-based multi-GPU execution
                - spmd       : SPMD execution via rrun launcher"""),
    )
    parser.add_argument(
        "--connect",
        dest="connect",
        default=None,
        type=str,
        help=textwrap.dedent("""\
            Connect to an existing cluster instead of creating a local one.
            Only supported with --frontend dask or ray:
                - dask : a TCP address (e.g. tcp://host:8786) or a scheduler file path
                - ray  : a Ray address (e.g. ray://host:10001 or "auto")"""),
    )
    parser.add_argument(
        "--num-gpus",
        dest="num_gpus",
        default=None,
        type=int,
        help="Number of GPUs for local cluster creation (--frontend ray/dask only). "
        "Cannot be used with --connect. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="Number of times to run the same query.",
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
        "--collect-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect data tracing cudf-polars execution.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug run.",
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
        help="Print the query results.",
        default=True,
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        help="Print an outline of the physical plan.",
        default=False,
    )
    parser.add_argument(
        "--explain-logical",
        action=argparse.BooleanOptionalAction,
        help="Print an outline of the logical plan.",
        default=False,
    )
    parser.add_argument(
        "--explain-partition-plan",
        action=argparse.BooleanOptionalAction,
        help="Print a combined partition plan summary table across all queries.",
        default=False,
    )
    parser.add_argument(
        "--print-plans",
        action=argparse.BooleanOptionalAction,
        help="Print the query plans.",
        default=True,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--validate-against",
        choices=["duckdb", "polars-cpu"],
        default=None,
        help=(
            "Validate the result against CPU execution. This will "
            "run the query, collect the results in memory and validate against a result from the "
            "selected CPU engine (CPU polars or DuckDB), comparing them using polars'. "
            "At larger scale factors, computing the expected result can be slow so "
            "--validate-directory should be used instead."
        ),
    )
    group.add_argument(
        "--validate-directory",
        type=str,
        default=None,
        help=(
            "Validate the results against a directory or object-storage prefix with a pre-computed set "
            "of 'golden' results. The directory should contain one parquet file per query, "
            "named 'q_DD.parquet', or `qDD.parquet` where DD is the zero-padded query number."
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
        default="CUDF_POLARS_LOG_TRACES_MEMORY,CUDF_POLARS_LOG_TRACES,DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT,DASK_DISTRIBUTED__COMM__UCX__CONNECT_TIMEOUT,KVIKIO_NTHREADS,LIBCUDF_NUM_HOST_WORKERS,OMP_NUM_THREADS,POLARS_MAX_THREADS,RAPIDSMPF_NUM_STREAMING_THREADS,UCX_MAX_RNDV_RAILS,UCX_PROTO_ENABLE,UCX_RNDV_FRAG_MEM_TYPES,UCX_RNDV_MTYPE_WORKER_FC_ENABLE,UCX_RNDV_MTYPE_WORKER_MAX_MEM,UCX_RNDV_PIPELINE_ERROR_HANDLING",
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

    StreamingOptions._add_cli_args(parser)

    # Trap legacy flags so we can emit clear errors.
    parser.add_argument(
        "--spill-device",
        dest="spill_device",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blocksize",
        dest="blocksize",
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def parse_args(
    args: Any = None,
    num_queries: int = 22,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse command line arguments."""
    if parser is None:
        parser = build_parser(num_queries)
    parsed_args = parser.parse_args(args)

    if parsed_args.spill_device is not None:
        parser.error(
            "--spill-device is not supported with --frontend; "
            "use --spill-device-limit instead, which takes a "
            'percentage, not a fraction (e.g. "80%").'
        )
    if parsed_args.blocksize is not None:
        parser.error(
            "--blocksize is not supported with --frontend; "
            "use --target-partition-size instead."
        )

    if (
        parsed_args.suffix
        and not parsed_args.suffix.startswith(".")
        and not parsed_args.suffix.startswith("/")
    ):
        parsed_args.suffix = f".{parsed_args.suffix}"

    return parsed_args


def run_polars(benchmark: Any, args: argparse.Namespace) -> None:
    """Run the queries using the given benchmark and frontend."""
    vars(args).update({"query_set": benchmark.name})
    run_config = RunConfig.from_args(args)

    if run_config.connect is not None and run_config.frontend not in ("dask", "ray"):
        raise ValueError("--connect is only supported with --frontend ray or dask.")

    if run_config.collect_traces and run_config.frontend in _CPU_ENGINES:
        raise ValueError(
            f"--collect-traces is not supported with --frontend {run_config.frontend}; "
            "cudf-polars tracing only applies to GPU frontends "
            "(in-memory, dask, ray, spmd)."
        )

    if run_config.validation_method is not None:
        validate_against = run_config.validation_method.expected_source
        if validate_against == run_config.frontend:
            raise ValueError(
                f"--validate-against {validate_against} is not supported with --frontend "
                f"{run_config.frontend}; validation compares a candidate engine against "
                "a baseline, so it only applies when the two are different."
            )
        if run_config.frontend == "duckdb":
            raise ValueError(
                "Validation is not currently supported with --frontend duckdb"
            )

    if args.debug and run_config.frontend in _CPU_ENGINES:
        raise ValueError(
            f"--debug is not supported with --frontend {run_config.frontend}; "
            "debug mode only applies to GPU frontends (in-memory, dask, ray, spmd)."
        )

    if run_config.num_gpus is not None:
        if run_config.connect is not None:
            raise ValueError("--num-gpus cannot be used with --connect.")
        if run_config.frontend not in ("dask", "ray"):
            raise ValueError(
                "--num-gpus is only supported with --frontend ray or dask."
            )
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValueError(
                "--num-gpus cannot be used when CUDA_VISIBLE_DEVICES is already set. "
                "Unset CUDA_VISIBLE_DEVICES or use it directly to control GPU visibility."
            )

    parquet_options: dict[str, Any] = {}
    numeric_type, date_type = check_input_data_type(run_config)
    match args.frontend:
        case "dask":
            run_polars_dask(
                benchmark, args, run_config, parquet_options, numeric_type, date_type
            )
        case "duckdb":
            run_duckdb(benchmark().duckdb_queries, args)
        case "in-memory":
            run_polars_in_memory(
                benchmark, args, run_config, parquet_options, numeric_type, date_type
            )
        case "polars-cpu":
            run_polars_cpu(benchmark, args, run_config, numeric_type, date_type)
        case "ray":
            run_polars_ray(
                benchmark, args, run_config, parquet_options, numeric_type, date_type
            )
        case "spmd":
            run_polars_spmd(
                benchmark, args, run_config, parquet_options, numeric_type, date_type
            )
        case _:
            raise ValueError(f"Unknown --frontend: {args.frontend!r}")
