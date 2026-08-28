# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF009  -- _opt() returns dataclasses.field(), not a mutable default
"""Unified streaming options for the RapidsMPF frontend."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import textwrap
from typing import TYPE_CHECKING, Any, Literal

from rapidsmpf.config import Options
from rapidsmpf.utils.string import parse_boolean

from cudf_polars.engine.hardware_binding import (
    HardwareBindingPolicy,
)
from cudf_polars.utils.config import (
    UNSPECIFIED,
    DynamicPlanningOptions,
    MaxConcurrentIOTasks,
    MemoryResourceConfig,
    Unspecified,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf_polars.quent import QuentContext
    from cudf_polars.utils.config import (
        JoinFilterPushdownOptions,
        ParquetOptions,
    )

__all__: list[str] = [
    "UNSPECIFIED",
    "StreamingOptions",
    "Unspecified",
]


def _opt(
    category: str,
    env_var: str | None = None,
    coerce: Callable[[str], Any] = str,
) -> Any:
    """
    Factory for ``StreamingOptions`` fields with category and env-var metadata.

    Parameters
    ----------
    category
        Which ``to_*`` method claims this field: ``"rapidsmpf"``,
        ``"executor"``, or ``"engine"``.
    env_var
        Environment variable to read at field-default time.  When a
        :class:`StreamingOptions` is instantiated without an explicit value for
        this field, the factory reads the environment variable (if set) on the constructing
        process.  ``None`` means no environment variable; the field defaults to
        :data:`UNSPECIFIED`.
    coerce
        Callable used to convert the raw env-var string to the field's type.
        Defaults to ``str`` (no conversion).
    """

    def _default() -> Any:
        if env_var:
            raw = os.environ.get(env_var)
            if raw is not None:
                return coerce(raw)
        return UNSPECIFIED

    return dataclasses.field(
        default_factory=_default,
        metadata={"category": category, "env_var": env_var, "coerce": coerce},
    )


def _category_opts(
    obj: Any, category: str, fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return fields of *obj* in *category*, falling back to *fallback* for UNSPECIFIED fields."""
    result = {}
    for f in dataclasses.fields(obj):
        if f.metadata.get("category") != category:
            continue
        v = getattr(obj, f.name)
        if isinstance(v, Unspecified):
            if fallback and f.name in fallback:
                result[f.name] = fallback[f.name]
        else:
            result[f.name] = v
    return result


def _parse_memory_resource_config(value: str) -> MemoryResourceConfig:
    """Argparse ``type`` callback: parse a JSON string into a :class:`MemoryResourceConfig`."""
    return MemoryResourceConfig(**json.loads(value))


def _parse_hardware_binding(value: str) -> HardwareBindingPolicy:
    """
    Parse a JSON string into a :class:`HardwareBindingPolicy`.

    Examples: ``'{"enabled": false}'``, ``'{"raise_on_fail": true}'``.
    """
    return HardwareBindingPolicy(**json.loads(value))


@dataclasses.dataclass
class StreamingOptions:
    """
    High-level configuration for the cudf-polars streaming executor and RapidsMPF.

    Options are grouped into three categories:
      - **RapidsMPF**: runtime and memory behavior (e.g. spilling, threading).
      - **Executor**: query execution and partitioning behavior.
      - **Engine**: Polars integration, IO configuration, hardware binding.

    All fields default to :data:`UNSPECIFIED` and follow this precedence:
      1. Explicit value.
      2. Environment variable.
      3. Built-in default.

    Parameters
    ----------
    num_streaming_threads
        Threads used to execute coroutines.
        Env: ``RAPIDSMPF_NUM_STREAMING_THREADS``.
        Default: ``1``.
        Category: rapidsmpf.
    num_streams
        CUDA streams for concurrent GPU execution.
        Env: ``RAPIDSMPF_NUM_STREAMS``.
        Default: ``16``.
        Category: rapidsmpf.
    log
        Log level (``"NONE"``, ``"PRINT"``, ``"WARN"``, ``"INFO"``,
        ``"DEBUG"``, ``"TRACE"``).
        Env: ``RAPIDSMPF_LOG``.
        Default: ``"WARN"``.
        Category: rapidsmpf.
    statistics
        Enable performance metrics.
        Env: ``RAPIDSMPF_STATISTICS``.
        Default: ``False``.
        Category: rapidsmpf.
    memory_reserve_timeout
        Timeout for memory reservations (e.g. ``"100ms"``).
        Env: ``RAPIDSMPF_MEMORY_RESERVE_TIMEOUT``.
        Default: ``"100ms"``.
        Category: rapidsmpf.
    allow_overbooking_by_default
        Allow overallocation in reservation APIs.
        Env: ``RAPIDSMPF_ALLOW_OVERBOOKING_BY_DEFAULT``.
        Default: ``True``.
        Category: rapidsmpf.
    pinned_memory
        Enable pinned host memory.
        Env: ``RAPIDSMPF_PINNED_MEMORY``.
        Default: ``False``.
        Category: rapidsmpf.
    pinned_initial_pool_size
        Initial pinned memory pool size (bytes).
        Env: ``RAPIDSMPF_PINNED_INITIAL_POOL_SIZE``.
        Default: ``0``.
        Category: rapidsmpf.
    pinned_max_pool_size
        Maximum pinned host memory pool size (e.g. ``"4GiB"``, ``"50%"``).
        Env: ``RAPIDSMPF_PINNED_MAX_POOL_SIZE``.
        Default: 80% of per-GPU host memory.
        Category: rapidsmpf.
    spill_device_limit
        Device memory soft limit before spilling (e.g. ``"80%"`` or bytes).
        Env: ``RAPIDSMPF_SPILL_DEVICE_LIMIT``.
        Default: ``"80%"``.
        Category: rapidsmpf.
    periodic_spill_check
        Interval between spill checks (e.g. ``"1ms"``).
        Env: ``RAPIDSMPF_PERIODIC_SPILL_CHECK``.
        Default: ``"1ms"``.
        Category: rapidsmpf.
    unbounded_file_read_cache
        Cache file-read results in the Context's message storage.
        Accepts a memory type (``"host"``, ``"pinned"``, ``"device"``) or
        ``"disabled"``. Primarily for benchmarking. Each file slice must be
        read with identical parameters (see rapidsmpf docs).
        Env: ``RAPIDSMPF_UNBOUNDED_FILE_READ_CACHE``.
        Default: ``"disabled"``.
        Category: rapidsmpf.
    ucxx_progress_mode
        UCXX progress mode (``"polling"``, ``"thread-blocking"``, or
        ``"thread-polling"``).
        Env: ``RAPIDSMPF_UCXX_PROGRESS_MODE``.
        Default: ``"thread-blocking"``.
        Category: rapidsmpf.
    num_py_executors
        Workers for the internal Python ``ThreadPoolExecutor``.
        Env: ``CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS``.
        Default: ``8``.
        Category: executor.
    kvikio_statistics
        Collect KvikIO I/O statistics, reachable through
        :meth:`~cudf_polars.engine.core.StreamingEngine.gather_io_summary`.
        Env: ``CUDF_POLARS__EXECUTOR__KVIKIO_STATISTICS``.
        Default: ``False``.
        Category: executor.
    max_concurrent_io_tasks
        Maximum concurrent IO tasks for each scan node.
        Env: ``CUDF_POLARS__EXECUTOR__MAX_CONCURRENT_IO_TASKS``.
        Default: automatic, resolved separately for each scan based on its paths.
        Python and config values may be an ``int``, a dict with ``local``
        and/or ``remote`` keys, or omitted/``None`` for the default policy.
        The environment variable accepts an int or a JSON dict.
        Category: executor.
    fallback_mode
        Fallback behavior (``"warn"``, ``"raise"``, ``"silent"``).
        Env: ``CUDF_POLARS__EXECUTOR__FALLBACK_MODE``.
        Default: ``"warn"``.
        Category: executor.
    max_rows_per_partition
        Maximum rows per partition.
        Env: ``CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION``.
        Default: ``1_000_000``.
        Category: executor.
    broadcast_limit
        Maximum byte size for broadcast joins.
        Env: ``CUDF_POLARS__EXECUTOR__BROADCAST_LIMIT``.
        Default: ``"auto"``.
        Category: executor.
    target_partition_size
        Target IO partition size (bytes). ``0`` = auto.
        Env: ``CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE``.
        Default: auto.
        Category: executor.
    dynamic_planning
        Dynamic planning config, dict or
        :class:`~cudf_polars.utils.config.DynamicPlanningOptions`. ``None`` disables.
        Env: ``CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING``.
        Default: enabled.
        Category: executor.
    join_filter_pushdown
        Config for join filter pushdown optimizations, dict or
        :class:`~cudf_polars.utils.config.JoinFilterPushdownOptions`. ``None``
        disables the rewrite.
        Env: ``CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN`` and
        ``CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN__*``.
        Default: enabled.
        Category: executor.
    sink_to_directory
        Whether multi-partition sink operations should write to a directory
        rather than a single file. The ``spmd``/``ray``/``dask`` engines
        always use ``True``; passing ``False`` raises ``ValueError``.
        Env: ``CUDF_POLARS__EXECUTOR__SINK_TO_DIRECTORY``.
        Default: ``True`` (forced by the streaming engines).
        Category: executor.
    quent_context
        Quent tracing context, or ``None`` to disable tracing.
        Env: ``CUDF_POLARS__EXECUTOR__QUENT_CONTEXT`` (``true``/``false``).
        Default: ``None`` (disabled).
    raise_on_fail
        Raise instead of falling back to CPU.
        Default: ``False``.
        Category: engine.
    parquet_options
        Parquet configuration, dict or
        :class:`~cudf_polars.utils.config.ParquetOptions`.
        Env: ``CUDF_POLARS__PARQUET_OPTIONS__*``.
        Category: engine.
    memory_resource_config
        RMM configuration, dict or
        :class:`~cudf_polars.utils.config.MemoryResourceConfig`.
        Env: ``CUDF_POLARS__MEMORY_RESOURCE_CONFIG__*``.
        Category: engine.
    hardware_binding
        Hardware binding policy. Pass a :class:`~cudf_polars.engine.hardware_binding.HardwareBindingPolicy`
        instance for fine-grained control.
        Env: ``CUDF_POLARS__HARDWARE_BINDING`` (JSON object,
        e.g. ``'{"enabled": false}'``).
        Default: ``HardwareBindingPolicy()``.
        Category: engine.
    allow_gpu_sharing
        When ``False`` (default), the engine raises if multiple ranks share the same physical GPU.
        Env: ``CUDF_POLARS__ALLOW_GPU_SHARING``.
        Default: ``False``.
        Category: engine.

    Examples
    --------
    >>> StreamingOptions(num_streaming_threads=8, log="DEBUG", fallback_mode="silent")
    StreamingOptions(...)
    """

    # ---- RapidsMPF runtime ----
    num_streaming_threads: int | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_NUM_STREAMING_THREADS", int
    )
    num_streams: int | Unspecified = _opt("rapidsmpf", "RAPIDSMPF_NUM_STREAMS", int)
    log: Literal["NONE", "PRINT", "WARN", "INFO", "DEBUG", "TRACE"] | Unspecified = (
        _opt("rapidsmpf", "RAPIDSMPF_LOG")
    )
    statistics: bool | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_STATISTICS", parse_boolean
    )
    memory_reserve_timeout: str | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_MEMORY_RESERVE_TIMEOUT"
    )
    allow_overbooking_by_default: bool | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_ALLOW_OVERBOOKING_BY_DEFAULT", parse_boolean
    )
    pinned_memory: bool | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_PINNED_MEMORY", parse_boolean
    )
    pinned_initial_pool_size: int | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_PINNED_INITIAL_POOL_SIZE", int
    )
    pinned_max_pool_size: str | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_PINNED_MAX_POOL_SIZE"
    )
    spill_device_limit: str | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_SPILL_DEVICE_LIMIT"
    )
    periodic_spill_check: str | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_PERIODIC_SPILL_CHECK"
    )
    unbounded_file_read_cache: str | Unspecified = _opt(
        "rapidsmpf", "RAPIDSMPF_UNBOUNDED_FILE_READ_CACHE"
    )
    ucxx_progress_mode: (
        Literal["polling", "thread-blocking", "thread-polling"] | Unspecified
    ) = _opt("rapidsmpf", "RAPIDSMPF_UCXX_PROGRESS_MODE")
    # ---- Executor ----
    num_py_executors: int | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS", int
    )
    kvikio_nthreads: int | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__KVIKIO_NTHREADS", int
    )
    kvikio_statistics: bool | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__KVIKIO_STATISTICS", parse_boolean
    )
    max_concurrent_io_tasks: int | dict[str, int] | Unspecified | None = _opt(
        "executor",
        "CUDF_POLARS__EXECUTOR__MAX_CONCURRENT_IO_TASKS",
        MaxConcurrentIOTasks.parse_env,
    )
    fallback_mode: str | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__FALLBACK_MODE"
    )
    max_rows_per_partition: int | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION", int
    )
    broadcast_limit: int | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__BROADCAST_LIMIT", int
    )
    target_partition_size: int | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE", int
    )
    dynamic_planning: dict[str, Any] | DynamicPlanningOptions | None | Unspecified = (
        _opt("executor")
    )
    join_filter_pushdown: (
        dict[str, Any] | JoinFilterPushdownOptions | None | Unspecified
    ) = _opt("executor")
    sink_to_directory: bool | Unspecified = _opt(
        "executor", "CUDF_POLARS__EXECUTOR__SINK_TO_DIRECTORY", parse_boolean
    )
    quent_context: QuentContext | None | Unspecified = _opt(
        "executor",
    )

    # ---- Engine ----
    raise_on_fail: bool | Unspecified = _opt("engine")
    parquet_options: dict[str, Any] | ParquetOptions | Unspecified = _opt("engine")
    memory_resource_config: MemoryResourceConfig | Unspecified = _opt("engine")
    hardware_binding: HardwareBindingPolicy | Unspecified = _opt(
        "engine", "CUDF_POLARS__HARDWARE_BINDING", _parse_hardware_binding
    )
    allow_gpu_sharing: bool | Unspecified = _opt(
        "engine", "CUDF_POLARS__ALLOW_GPU_SHARING", parse_boolean
    )

    # ------------------------------------------------------------------
    # Conversion helpers used by the engines
    # ------------------------------------------------------------------

    def to_rapidsmpf_options(self) -> Options:
        """
        Build a ``rapidsmpf.config.Options`` from the RapidsMPF fields.

        ``RAPIDSMPF_*`` environment variables are resolved at
        :class:`StreamingOptions` construction time, so any field still
        :data:`UNSPECIFIED` here has no environment variable and no explicit value; the
        rapidsmpf C++ library will apply its own built-in default for those.
        """
        return Options(
            {k: str(v) for k, v in _category_opts(self, "rapidsmpf").items()}
        )

    def to_executor_options(self) -> dict[str, Any]:
        """
        Build a ``StreamingExecutor`` kwargs dict from the executor fields.

        Only fields that are not :data:`UNSPECIFIED` are included.
        ``StreamingExecutor`` reads ``CUDF_POLARS__EXECUTOR__*`` environment
        variables for any omitted fields.
        """
        return _category_opts(self, "executor")

    def to_engine_options(self) -> dict[str, Any]:
        """
        Build a ``pl.GPUEngine`` kwargs dict from the engine fields.

        Only fields that are not :data:`UNSPECIFIED` are included.
        ``ConfigOptions.from_polars_engine`` handles environment variables for
        any omitted fields.
        """
        return _category_opts(self, "engine")

    def to_dict(self) -> dict[str, Any]:
        """
        Return all explicitly-set fields as a plain dictionary.

        Fields that are :data:`UNSPECIFIED` are omitted. The result can be
        round-tripped via :meth:`from_dict`.

        Returns
        -------
        Mapping of field name to value for every non-UNSPECIFIED field.

        Examples
        --------
        >>> StreamingOptions(fallback_mode="silent").to_dict()
        {'fallback_mode': 'silent'}
        >>> StreamingOptions.from_dict(
        ...     StreamingOptions(fallback_mode="silent").to_dict()
        ... )  # doctest: +ELLIPSIS
        StreamingOptions(...)
        """
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not isinstance(getattr(self, f.name), Unspecified)
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamingOptions:
        """
        Build a :class:`StreamingOptions` from a plain dictionary.

        Keys must be field names of :class:`StreamingOptions`. Values of ``None``
        and missing keys both leave the corresponding field at :data:`UNSPECIFIED`.

        Parameters
        ----------
        data
            Flat dictionary of option name to value. Unknown keys raise
            :exc:`TypeError`.

        Returns
        -------
        A new :class:`StreamingOptions` instance.

        Examples
        --------
        >>> StreamingOptions.from_dict(
        ...     {"fallback_mode": "silent", "num_streaming_threads": 4}
        ... )  # doctest: +ELLIPSIS
        StreamingOptions(...)
        >>> StreamingOptions.from_dict({})  # all fields UNSPECIFIED
        StreamingOptions(...)
        """
        return cls(**{k: (UNSPECIFIED if v is None else v) for k, v in data.items()})

    @classmethod
    def _from_argparse(cls, args: argparse.Namespace) -> StreamingOptions:
        """
        Build a :class:`StreamingOptions` from a parsed :class:`argparse.Namespace`.

        Designed to work with namespaces produced by :func:`argparse.ArgumentParser`
        parsers that have been augmented with :meth:`_add_cli_args`. Fields not present
        in the namespace (or set to ``None``) let their default factories read the
        environment variable. Fields with no explicit value or environment variable
        remain :data:`UNSPECIFIED`; their downstream consumer applies its built-in
        default.

        Parameters
        ----------
        args
            Parsed namespace, typically from ``parser.parse_args()``.

        Returns
        -------
        A new :class:`StreamingOptions` instance.

        Examples
        --------
        >>> parser = argparse.ArgumentParser()
        >>> StreamingOptions._add_cli_args(parser)
        >>> opts = StreamingOptions._from_argparse(parser.parse_args([]))
        """

        def _get(attr: str) -> Any:
            """Return the attr value, or UNSPECIFIED if absent or None."""
            v = getattr(args, attr, None)
            return UNSPECIFIED if v is None else v

        # dynamic_planning is a BooleanOptionalAction with default=None, so
        # dyn is True only when --dynamic-planning was passed explicitly,
        # False only for --no-dynamic-planning, and None when absent.
        # An explicit True must be preserved (not UNSPECIFIED), otherwise an
        # env var disabling dynamic planning would silently override it.
        dyn = getattr(args, "dynamic_planning", None)
        dynamic_planning: Any = (
            UNSPECIFIED
            if dyn is None
            else None
            if dyn is False
            else DynamicPlanningOptions()
        )

        # target_partition_size: canonical dest from _add_cli_args; fall back to
        # "blocksize" for legacy benchmark scripts that predate this module.
        target_partition_size = (
            _get("target_partition_size")
            if "target_partition_size" in vars(args)
            else _get("blocksize")
        )

        # A field's CLI dest is usually its own name, except where a bare name
        # would be ambiguous on the command line (e.g. "log" -> "--log" reads as
        # a generic logging flag, so its dest is namespaced to "rapidsmpf_log").
        cli_dest_overrides = {
            "log": "rapidsmpf_log",
            "statistics": "rapidsmpf_statistics",
        }
        special_cased = {
            "target_partition_size": target_partition_size,
            "dynamic_planning": dynamic_planning,
        }
        kwargs: dict[str, Any] = {
            f.name: special_cased.get(
                f.name, _get(cli_dest_overrides.get(f.name, f.name))
            )
            for f in dataclasses.fields(cls)
        }
        # Omit UNSPECIFIED entries rather than passing them explicitly, so
        # each field's own default_factory (env var, then built-in default)
        # runs instead of being short-circuited.
        return cls(
            **{k: v for k, v in kwargs.items() if not isinstance(v, Unspecified)}
        )

    @staticmethod
    def _add_cli_args(parser: argparse.ArgumentParser) -> None:
        """
        Register :class:`StreamingOptions`-specific CLI arguments on *parser*.

        Helpful to implement all available options as CLI arguments.

        Parameters
        ----------
        parser
            The parser to augment in-place.

        Examples
        --------
        >>> parser = argparse.ArgumentParser()
        >>> StreamingOptions._add_cli_args(parser)
        >>> opts = StreamingOptions._from_argparse(parser.parse_args([]))
        """
        g = parser.add_argument_group("Streaming and RapidsMPF Options")
        g.add_argument(
            "--num-streaming-threads",
            dest="num_streaming_threads",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Number of threads used to execute coroutines. Must be > 0.
                Env: RAPIDSMPF_NUM_STREAMING_THREADS. Built-in default: 1."""),
        )
        g.add_argument(
            "--num-streams",
            dest="num_streams",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Number of CUDA streams used by RapidsMPF for concurrent GPU execution.
                Env: RAPIDSMPF_NUM_STREAMS. Built-in default: 16."""),
        )
        g.add_argument(
            "--rapidsmpf-log",
            dest="rapidsmpf_log",
            default=None,
            type=str,
            choices=["NONE", "PRINT", "WARN", "INFO", "DEBUG", "TRACE"],
            help=textwrap.dedent("""\
                RapidsMPF log verbosity level.
                Env: RAPIDSMPF_LOG. Built-in default: WARN."""),
        )
        g.add_argument(
            "--rapidsmpf-statistics",
            dest="rapidsmpf_statistics",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Collect RapidsMPF performance metrics.
                Env: RAPIDSMPF_STATISTICS. Built-in default: false."""),
        )
        g.add_argument(
            "--memory-reserve-timeout",
            dest="memory_reserve_timeout",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Global timeout for memory reservation requests (e.g. "100ms").
                Env: RAPIDSMPF_MEMORY_RESERVE_TIMEOUT. Built-in default: 100ms."""),
        )
        g.add_argument(
            "--allow-overbooking",
            dest="allow_overbooking_by_default",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Allow memory overallocation by default in high-level reservation APIs.
                Env: RAPIDSMPF_ALLOW_OVERBOOKING_BY_DEFAULT. Built-in default: true."""),
        )
        g.add_argument(
            "--pinned-memory",
            dest="pinned_memory",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Enable pinned host memory if available on the system.
                Env: RAPIDSMPF_PINNED_MEMORY. Built-in default: false."""),
        )
        g.add_argument(
            "--pinned-initial-pool-size",
            dest="pinned_initial_pool_size",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Starting allocation for the pinned memory pool in bytes.
                Env: RAPIDSMPF_PINNED_INITIAL_POOL_SIZE. Built-in default: 0."""),
        )
        g.add_argument(
            "--pinned-max-pool-size",
            dest="pinned_max_pool_size",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Maximum size of the pinned memory pool. Accepts byte counts
                (e.g. "4GiB") or a percentage (e.g. "80%%").
                Env: RAPIDSMPF_PINNED_MAX_POOL_SIZE.
                Built-in default: 80%% of per-GPU host memory."""),
        )
        g.add_argument(
            "--spill-device-limit",
            dest="spill_device_limit",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Soft upper limit on device memory usage before spilling,
                expressed as a percentage (e.g. "80%%") or byte count.
                Env: RAPIDSMPF_SPILL_DEVICE_LIMIT. Built-in default: 80%%."""),
        )
        g.add_argument(
            "--periodic-spill-check",
            dest="periodic_spill_check",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Interval between periodic spill checks (e.g. "1ms").
                Env: RAPIDSMPF_PERIODIC_SPILL_CHECK. Built-in default: 1ms."""),
        )
        g.add_argument(
            "--unbounded-file-read-cache",
            dest="unbounded_file_read_cache",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Cache file-read results in the Context's message storage.
                One of "host", "pinned", "device", or "disabled".
                Env: RAPIDSMPF_UNBOUNDED_FILE_READ_CACHE. Built-in default: disabled."""),
        )
        g.add_argument(
            "--hardware-binding",
            dest="hardware_binding",
            default=None,
            type=_parse_hardware_binding,
            help=textwrap.dedent("""\
                Hardware binding policy as a JSON object
                (e.g. '{"enabled": false}', '{"raise_on_fail": true}').
                Env: CUDF_POLARS__HARDWARE_BINDING.
                Built-in default: enabled with auto GPU detection."""),
        )
        g.add_argument(
            "--num-py-executors",
            dest="num_py_executors",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Max workers for the Python ThreadPoolExecutor inside RapidsMPF.
                Env: CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS.
                Built-in default: 8."""),
        )
        g.add_argument(
            "--kvikio-statistics",
            dest="kvikio_statistics",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Collect KvikIO I/O statistics, reported per rank.
                Env: CUDF_POLARS__EXECUTOR__KVIKIO_STATISTICS.
                Built-in default: false."""),
        )
        g.add_argument(
            "--max-concurrent-io-tasks",
            dest="max_concurrent_io_tasks",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Maximum concurrent IO tasks for each scan node.
                Env: CUDF_POLARS__EXECUTOR__MAX_CONCURRENT_IO_TASKS.
                Omit to use the path-dependent default."""),
        )
        g.add_argument(
            "--raise-on-fail",
            dest="raise_on_fail",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Raise an exception instead of falling back to CPU when the GPU
                engine cannot execute a query.
                Built-in default: false."""),
        )
        g.add_argument(
            "--fallback-mode",
            dest="fallback_mode",
            default=None,
            type=str,
            choices=["warn", "raise", "silent"],
            help=textwrap.dedent("""\
                Behavior when a query node cannot run on the GPU.
                Env: CUDF_POLARS__EXECUTOR__FALLBACK_MODE. Built-in default: warn."""),
        )
        g.add_argument(
            "--max-rows-per-partition",
            dest="max_rows_per_partition",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Maximum number of rows per partition.
                Env: CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION.
                Built-in default: 1000000."""),
        )
        g.add_argument(
            "--broadcast-limit",
            dest="broadcast_limit",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Maximum byte size for broadcast joins. 0 = auto.
                Env: CUDF_POLARS__EXECUTOR__BROADCAST_LIMIT."""),
        )
        g.add_argument(
            "--target-partition-size",
            dest="target_partition_size",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Target IO partition size in bytes. 0 = auto.
                Env: CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE. Built-in default: auto."""),
        )
        g.add_argument(
            "--dynamic-planning",
            dest="dynamic_planning",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Enable dynamic planning. Use --no-dynamic-planning to disable.
                Env: CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING. Built-in default: enabled."""),
        )
        g.add_argument(
            "--parquet-options",
            dest="parquet_options",
            default=None,
            type=json.loads,
            help=textwrap.dedent("""\
                Parquet configuration as a JSON object.
                Env: CUDF_POLARS__PARQUET_OPTIONS__*."""),
        )
        g.add_argument(
            "--memory-resource-config",
            dest="memory_resource_config",
            default=None,
            type=_parse_memory_resource_config,
            help=textwrap.dedent("""\
                RMM memory resource configuration as a JSON object.
                Env: CUDF_POLARS__MEMORY_RESOURCE_CONFIG__*."""),
        )
