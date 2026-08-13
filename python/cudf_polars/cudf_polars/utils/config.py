# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration utilities for the cudf-polars engine.

Most users will not construct these objects directly. Instead, you'll pass
keyword arguments to :class:`~polars.lazyframe.engine_config.GPUEngine`. The
majority of the options are passed as `**kwargs` and collected into the
configuration described below:

.. code-block:: python

   >>> import polars as pl
   >>> engine = pl.GPUEngine(
   ...     executor="streaming",
   ...     executor_options={"fallback_mode": "raise"}
   ... )

"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib.util
import json
import os
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable
    from concurrent.futures import ThreadPoolExecutor

    import distributed
    from ray.actor import ActorHandle

    import polars.lazyframe.engine_config

    import rmm.mr
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.engine.ray import RankActor
    from cudf_polars.quent._context import QuentContext
    from cudf_polars.quent._logging import QuentLogger


__all__ = [
    "Cluster",
    "ConfigOptions",
    "DaskContext",
    "DynamicPlanningOptions",
    "InMemoryExecutor",
    "JoinFilterPushdownOptions",
    "ParquetOptions",
    "RayContext",
    "SPMDContext",
    "StreamingExecutor",
    "StreamingFallbackMode",
]


def _env_get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):  # pragma: no cover
        return default  # pragma: no cover


@functools.cache
def get_device_handle() -> Any:
    # Gets called for each IR.do_evaluate node, so we'll cache it.
    import pynvml

    try:
        pynvml.nvmlInit()
        index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        if index and not index.isnumeric():  # pragma: no cover
            # This means device_index is UUID.
            # This works for both MIG and non-MIG device UUIDs.
            handle = pynvml.nvmlDeviceGetHandleByUUID(str.encode(index))
            if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                # Additionally get parent device handle
                # if the device itself is a MIG instance
                handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
    except pynvml.NVMLError_NotSupported:  # pragma: no cover
        # System doesn't have proper "GPU memory".
        return None
    else:
        return handle


@functools.cache
def get_total_device_memory() -> int | None:
    """Return the total memory of the current device."""
    import pynvml

    maybe_handle = get_device_handle()

    if maybe_handle is not None:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(maybe_handle).total
        except pynvml.NVMLError_NotSupported:  # pragma: no cover
            # System doesn't have proper "GPU memory".
            return None
    else:  # pragma: no cover
        return None


class StreamingFallbackMode(enum.StrEnum):
    """
    How the streaming executor handles operations that don't support multiple partitions.

    Upon encountering an unsupported operation, the streaming executor will fall
    back to using a single partition, which might use a large amount of memory.

    * ``StreamingFallbackMode.WARN`` : Emit a warning and fall back to a single partition.
    * ``StreamingFallbackMode.SILENT``: Silently fall back to a single partition.
    * ``StreamingFallbackMode.RAISE`` : Raise an exception.
    """

    WARN = "warn"
    RAISE = "raise"
    SILENT = "silent"


class Cluster(enum.StrEnum):
    """
    The cluster configuration for the streaming executor.

    * ``Cluster.DEFAULT_SINGLETON`` : Single-GPU execution via the DefaultSingletonEngine.
    * ``Cluster.SPMD`` : Multi-GPU SPMD execution via the SPMDEngine.
    * ``Cluster.RAY`` : Multi-GPU execution via the RayEngine.
    * ``Cluster.DASK`` : Multi-GPU execution via the DaskEngine.
    """

    DEFAULT_SINGLETON = "default_singleton"
    SPMD = "spmd"
    RAY = "ray"
    DASK = "dask"


T = TypeVar("T")
DefaultT = TypeVar("DefaultT")


def _make_default_factory(
    key: str, converter: Callable[[str], T], *, default: DefaultT
) -> Callable[[], T | DefaultT]:
    def default_factory() -> T | DefaultT:
        v = os.environ.get(key)
        if v is None:
            return default
        return converter(v)

    return default_factory


def _bool_converter(v: str) -> bool:
    lowered = v.lower()
    if lowered in {"true", "yes", "y", "1"}:
        return True
    elif lowered in {"false", "no", "n", "0"}:
        return False
    else:
        raise ValueError(f"Invalid boolean value: '{v}'")


def _quent_context_converter(v: str) -> QuentContext | None:
    from cudf_polars.quent._context import QuentContext

    try:
        enabled = _bool_converter(v)
    except ValueError as e:
        raise ValueError(f"Invalid value for quent_context: '{v}'") from e
    else:
        if enabled:
            return QuentContext()
        else:
            return None


@dataclasses.dataclass(frozen=True)
class ParquetOptions:
    """
    Configuration for the cudf-polars Parquet engine.

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__PARQUET_OPTIONS__``.

    Parameters
    ----------
    chunked
        Whether to use libcudf's ``ChunkedParquetReader`` or ``ChunkedParquetWriter``
        to read/write the parquet dataset in chunks. This is useful when reading/writing
        very large parquet files.
    n_output_chunks
        Split the dataframe in ``n_output_chunks`` when using libcudf's ``ChunkedParquetWriter``.
    chunk_read_limit
        Limit on total number of bytes to be returned per read, or 0 if
        there is no limit.
    pass_read_limit
        Limit on the amount of memory used for reading and decompressing data
        or 0 if there is no limit.
    max_footer_samples
        Maximum number of file footers to sample for metadata. This
        option is currently used by the streaming executor to gather
        datasource statistics before generating a physical plan. Set to
        0 to avoid metadata sampling. Default is 3.
    max_row_group_samples
        Maximum number of row-groups to sample for unique-value statistics.
        This option may be used by the streaming executor to optimize
        the physical plan. Default is 1.

        Set to 0 to avoid row-group sampling. Note that row-group sampling
        will also be skipped if ``max_footer_samples`` is 0.
    prefetch_file_metadata
        Whether to prefetch parquet file metadata and pass it through
        `parquet_metadatas` to avoid rereading file footers.
    use_jit_filter
        Whether to use JIT compilation for post-read filtering in Parquet scans.
        When enabled, filter predicates are JIT-compiled to CUDA kernels for
        improved performance on large datasets with complex filters.
        Default is False.
    """

    _env_prefix = "CUDF_POLARS__PARQUET_OPTIONS"

    chunked: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CHUNKED", _bool_converter, default=True
        )
    )
    n_output_chunks: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__N_OUTPUT_CHUNKS", int, default=1
        )
    )
    chunk_read_limit: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CHUNK_READ_LIMIT", int, default=0
        )
    )
    pass_read_limit: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__PASS_READ_LIMIT", int, default=0
        )
    )
    max_footer_samples: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_FOOTER_SAMPLES", int, default=3
        )
    )
    max_row_group_samples: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_ROW_GROUP_SAMPLES", int, default=1
        )
    )
    prefetch_file_metadata: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__PREFETCH_FILE_METADATA",
            _bool_converter,
            default=False,
        )
    )
    use_jit_filter: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__USE_JIT_FILTER",
            _bool_converter,
            default=False,
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.chunked, bool):
            raise TypeError("chunked must be a bool")
        if not isinstance(self.n_output_chunks, int):
            raise TypeError("n_output_chunks must be an int")
        if not isinstance(self.chunk_read_limit, int):
            raise TypeError("chunk_read_limit must be an int")
        if not isinstance(self.pass_read_limit, int):
            raise TypeError("pass_read_limit must be an int")
        if not isinstance(self.max_footer_samples, int):
            raise TypeError("max_footer_samples must be an int")
        if not isinstance(self.max_row_group_samples, int):
            raise TypeError("max_row_group_samples must be an int")
        if not isinstance(self.prefetch_file_metadata, bool):
            raise TypeError("prefetch_file_metadata must be a bool")
        if not isinstance(self.use_jit_filter, bool):
            raise TypeError("use_jit_filter must be a bool")


def default_target_partition_size(min_device_size: int | None) -> int:
    """Return the default target partition size."""
    _DEFAULT_TARGET_PARTITION_SIZE = 1_500_000_000
    if min_device_size is None:  # pragma: no cover
        return _DEFAULT_TARGET_PARTITION_SIZE
    # Limit to 2.5% of the minimum device memory across all ranks.
    return min(max(int(min_device_size * 0.025), 1), _DEFAULT_TARGET_PARTITION_SIZE)


def default_broadcast_limit(min_device_size: int | None) -> int:
    """Return the default broadcast limit."""
    # TODO: Need empirical data to determine the optimal value.
    _DEFAULT_BROADCAST_LIMIT = 16_000_000_000
    if min_device_size is None:  # pragma: no cover
        return _DEFAULT_BROADCAST_LIMIT
    # Limit to 15% of the minimum device memory across all ranks.
    return min(max(int(min_device_size * 0.15), 1), _DEFAULT_BROADCAST_LIMIT)


@dataclasses.dataclass(frozen=True)
class DynamicPlanningOptions:
    """
    Configuration for dynamic shuffle planning.

    When enabled, shuffle decisions for GroupBy/Join/Unique operations
    are made at runtime by sampling real chunks.

    To enable dynamic planning, pass a ``DynamicPlanningOptions`` instance
    to ``StreamingExecutor(dynamic_planning=...)``. To disable it, pass
    ``None`` (the default).

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING__``.

    Parameters
    ----------
    sample_chunk_count
        The maximum number of chunks to sample before making
        dynamic-planning decisions. Default is 2.
    """

    _env_prefix = "CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING"

    sample_chunk_count: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SAMPLE_CHUNK_COUNT", int, default=2
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.sample_chunk_count, int):
            raise TypeError("sample_chunk_count must be an int")
        if self.sample_chunk_count < 1:
            raise ValueError("sample_chunk_count must be at least 1")


@dataclasses.dataclass(frozen=True)
class JoinFilterPushdownOptions:
    """
    Configuration options for join filter pushdown in the logical plan.

    When performing a join between two tables, it is often favourable
    to pre-filter one side of the join with the keys (full or partial) of
    the other side. This can reduce the size of tables that actually
    participate in the join.

    cudf-polars supports a form of this where we can rewrite inner joins by
    selecting a side to be filtered by the keys of the other side.

    Pass ``None`` to ``StreamingExecutor(join_filter_pushdown=...)`` to
    disable the rewrite.

    These options can be configured via environment variables with the prefix
    ``CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN__``.

    Parameters
    ----------
    threshold
        Row-count ratio (key-provider-rows / to-be-filtered-table-rows) below which a
        filter on is inserted on the to-be-filtered table. Default is 0.5.
    trace
        Whether to emit plan-time trace decisions for filter decisions. Default is False.
    """

    _env_prefix = "CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN"

    threshold: float = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__THRESHOLD", float, default=0.5
        )
    )
    trace: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__TRACE", _bool_converter, default=False
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        threshold = self.threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a float or int")
        threshold = float(threshold)
        object.__setattr__(self, "threshold", threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not isinstance(self.trace, bool):
            raise TypeError("trace must be a bool")


@dataclasses.dataclass(frozen=True, eq=True)
class MemoryResourceConfig:
    """
    Configuration for the default memory resource.

    Parameters
    ----------
    qualname
        The fully qualified name of the memory resource class to use.
    options
        This can be either a dictionary representing the options to pass
        to the memory resource class, or, a dictionary representing a
        nested memory resource configuration. The presence of "qualname"
        field indicates a nested memory resource configuration.

    Examples
    --------
    Create a memory resource config for a single memory resource:

    >>> MemoryResourceConfig(
    ...     qualname="rmm.mr.CudaAsyncMemoryResource",
    ...     options={"initial_pool_size": 100},
    ... )

    Create a memory resource config for a nested memory resource configuration:

    >>> MemoryResourceConfig(
    ...     qualname="rmm.mr.PrefetchResourceAdaptor",
    ...     options={
    ...         "upstream_mr": {
    ...             "qualname": "rmm.mr.PoolMemoryResource",
    ...             "options": {
    ...                 "upstream_mr": {
    ...                     "qualname": "rmm.mr.ManagedMemoryResource",
    ...                 },
    ...                 "initial_pool_size": 256,
    ...             },
    ...         }
    ...     },
    ... )
    """

    _env_prefix = "CUDF_POLARS__MEMORY_RESOURCE_CONFIG"
    qualname: str = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__QUALNAME",
            str,
            # We shouldn't reach here if qualname isn't set in the environment.
            default=None,  # type: ignore[assignment]
        )
    )
    options: dict[str, Any] | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__OPTIONS",
            json.loads,
            default=None,
        )
    )

    def __post_init__(self) -> None:
        if self.qualname.count(".") < 1:
            raise ValueError(
                f"MemoryResourceConfig.qualname '{self.qualname}' must be a fully qualified name to a class, including the module name."
            )

    def create_memory_resource(self) -> rmm.mr.DeviceMemoryResource:
        """Create a memory resource from the configuration."""

        def create_mr(
            qualname: str, options: dict[str, Any] | None
        ) -> rmm.mr.DeviceMemoryResource:
            module_name, class_name = qualname.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(**options or {})

        def process_options(opts: dict[str, Any] | None) -> dict[str, Any]:
            if opts is None:
                return {}

            processed = {}
            for key, value in opts.items():
                if isinstance(value, dict) and "qualname" in value:
                    # This is a nested memory resource config
                    nested_qualname = value["qualname"]
                    nested_options = process_options(value.get("options"))
                    processed[key] = create_mr(nested_qualname, nested_options)
                else:
                    processed[key] = value
            return processed

        # Create the top-level memory resource
        return create_mr(self.qualname, process_options(self.options))

    def __hash__(self) -> int:
        return hash((self.qualname, json.dumps(self.options, sort_keys=True)))

    @classmethod
    def default(cls) -> MemoryResourceConfig:
        """
        The default memory resource config.

        This defaults to a CUDA Async Memory Resource with

        - No initial pool size
        - A release threshold equal to 90% of the size of the device's memory.
        """
        if (device_size := get_total_device_memory()) is None:  # pragma: no cover
            # System doesn't have proper "GPU memory".
            # We probably want to use the default async memory resource.
            release_threshold = None
        else:
            release_threshold = int(0.9 * device_size)
        return cls(
            qualname="rmm.mr.CudaAsyncMemoryResource",
            options={
                "release_threshold": release_threshold,
            },
        )


@dataclasses.dataclass(frozen=True)
class SPMDContext:
    """
    Configuration for SPMD (Single Program Multiple Data) execution.

    .. note::
        This dataclass is **not picklable** because :class:`Communicator`,
        :class:`Context`, and :class:`~concurrent.futures.ThreadPoolExecutor`
        cannot be serialized. In SPMD mode each rank constructs its own
        ``SPMDContext`` locally inside :class:`~cudf_polars.engine.spmd.SPMDEngine`,
        so pickling is never required. Do not use this class with Dask or any other
        framework that serializes executor configuration across process boundaries.

    Parameters
    ----------
    comm
        The active RapidsMPF communicator.
    context
        The active RapidsMPF context.
    py_executor
        Thread-pool executor used to drive the actor network on each rank.
    """

    comm: Communicator
    context: Context
    py_executor: ThreadPoolExecutor
    engine_id: uuid.UUID
    worker_id: uuid.UUID
    quent_logger: QuentLogger | None


@dataclasses.dataclass(frozen=True)
class RayContext:
    """
    Configuration for Ray cluster execution.

    .. note::
        This dataclass holds Ray actor handles, which are only valid within the
        Ray session that created them. It is stripped from ``config_options``
        before pickling for remote actor calls in :func:`~cudf_polars.engine.ray.evaluate_pipeline_ray_mode`
        by :class:`~cudf_polars.engine.ray.RayEngine`. Do not persist or transfer
        this object across Ray sessions.

    Parameters
    ----------
    rank_actors
        List of :class:`~cudf_polars.engine.ray.RankActor` handles, one per GPU
        in the cluster.
    """

    rank_actors: list[ActorHandle[RankActor]]
    quent_logger: QuentLogger | None


@dataclasses.dataclass(frozen=True)
class DaskContext:
    """
    Configuration for Dask cluster execution.

    .. note::
        This dataclass holds a :class:`~distributed.Client` handle, which is
        only valid within the Dask session that created it. It is stripped from
        ``config_options`` before pickling for remote worker calls in
        :func:`~cudf_polars.engine.dask.evaluate_pipeline_dask_mode`.
        Do not persist or transfer this object across Dask sessions.

    Parameters
    ----------
    client
        Active :class:`~distributed.Client` connected to the cluster.
    rapidsmpf_id
        Unique identifier for this RapidsMPF bootstrap session.
    owned_client
        Client to close on shutdown, if created internally by
        :class:`~cudf_polars.engine.dask.DaskEngine`.
    owned_cluster
        Cluster to close on shutdown, if created internally.
    """

    client: distributed.Client
    rapidsmpf_id: str
    quent_logger: QuentLogger | None
    owned_client: distributed.Client | None = None
    owned_cluster: Any | None = None


@dataclasses.dataclass(frozen=True, eq=True)
class StreamingExecutor:
    """
    Configuration for the cudf-polars streaming executor.

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__EXECUTOR__``.

    Parameters
    ----------
    cluster
        The cluster configuration for the streaming executor.
        ``Cluster.DEFAULT_SINGLETON`` by default.

        * ``Cluster.DEFAULT_SINGLETON``: Single-GPU execution
        * ``Cluster.SPMD``: Multi-GPU SPMD execution
        * ``Cluster.RAY``: Multi-GPU Ray execution
        * ``Cluster.DASK``: Multi-GPU Dask execution

    fallback_mode
        How to handle errors when the GPU engine fails to execute a query.
        ``StreamingFallbackMode.WARN`` by default.

        This can be set using the ``CUDF_POLARS__EXECUTOR__FALLBACK_MODE``
        environment variable.
    max_rows_per_partition
        The maximum number of rows to process per partition. 1_000_000 by default.
        When the number of rows exceeds this value, the query will be split into
        multiple partitions and executed in parallel.
    target_partition_size
        Target partition size, in bytes, for IO tasks. This configuration currently
        controls how large parquet files are split into multiple partitions.
        Files larger than ``target_partition_size`` bytes are split into multiple
        partitions.

        This can be set via

        - ``executor_options`` passed to ``polars.GPUEngine``
        - the ``CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE`` environment variable

        By default, cudf-polars uses the minimum of 1.5GB or 2.5% of the minimum
        device size in the cluster. If pynvml cannot query the the device size(s),
        the default ``target_partition_size`` will be 1.5GB.
    broadcast_limit
        The maximum number of bytes to broadcast in a single operation.
        By default, cudf-polars uses the minimum of 16GB or 15% of the minimum
        device size in the cluster. If pynvml cannot query the the device size(s),
        the default ``broadcast_limit`` will be 16GB.
    client_device_threshold
        Threshold for spilling data from device memory.
        Default is 50% of device memory on the client process.
    sink_to_directory
        Whether multi-partition sink operations write to a directory rather
        than a single file. For the spmd, ray, and dask clusters this is
        always True; setting it to False raises a ValueError.
    dynamic_planning
        Options controlling dynamic shuffle planning. See
        :class:`~cudf_polars.utils.config.DynamicPlanningOptions` for more.
    join_filter_pushdown
        Options controlling the logical join-domain prefilter rewrite. See
        :class:`~cudf_polars.utils.config.JoinFilterPushdownOptions` for more.
        ``None`` disables the rewrite.

        Enable through environment variables with
        ``CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN=1``.
    max_concurrent_io_tasks
        Maximum number of concurrent IO tasks for each scan node. Default is 2.
        This can be set via

        - ``executor_options`` passed to ``polars.GPUEngine``
        - the ``CUDF_POLARS__EXECUTOR__MAX_CONCURRENT_IO_TASKS`` environment variable
    num_py_executors
        Maximum number of workers for the Python ThreadPoolExecutor.
        Default is 8.
    quent_context
        Quent tracing context. When ``None`` (default), Quent tracing is disabled.
        Pass a :class:`~cudf_polars.quent.QuentContext` instance to enable tracing.
        Can be set via the ``CUDF_POLARS__EXECUTOR__QUENT_CONTEXT`` environment
        variable (``true`` enables tracing with a default context, ``false``
        disables it).

    Notes
    -----
    The streaming executor does not currently support profiling a query via
    the ``.profile()`` method. We recommend using nsys to profile queries.
    """

    _env_prefix = "CUDF_POLARS__EXECUTOR"

    name: Literal["streaming"] = dataclasses.field(default="streaming", init=False)
    cluster: Cluster | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CLUSTER",
            Cluster.__call__,
            default=None,
        )
    )
    fallback_mode: StreamingFallbackMode = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__FALLBACK_MODE",
            StreamingFallbackMode.__call__,
            default=StreamingFallbackMode.WARN,
        )
    )
    max_rows_per_partition: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_ROWS_PER_PARTITION", int, default=1_000_000
        )
    )
    target_partition_size: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__TARGET_PARTITION_SIZE", int, default=0
        )
    )
    broadcast_limit: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__BROADCAST_LIMIT", int, default=0
        )
    )
    client_device_threshold: float = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CLIENT_DEVICE_THRESHOLD", float, default=0.5
        )
    )
    sink_to_directory: bool | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SINK_TO_DIRECTORY", _bool_converter, default=None
        )
    )
    dynamic_planning: DynamicPlanningOptions | None = dataclasses.field(
        default_factory=DynamicPlanningOptions
    )
    join_filter_pushdown: JoinFilterPushdownOptions | None = dataclasses.field(
        default_factory=JoinFilterPushdownOptions
    )
    max_concurrent_io_tasks: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_CONCURRENT_IO_TASKS", int, default=2
        )
    )
    num_py_executors: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__NUM_PY_EXECUTORS", int, default=8
        )
    )

    min_device_size: int | None = None
    spmd_context: SPMDContext | None = None
    ray_context: RayContext | None = None
    dask_context: DaskContext | None = None
    quent_context: QuentContext | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__QUENT_CONTEXT", _quent_context_converter, default=None
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        if self.cluster is None:
            object.__setattr__(self, "cluster", Cluster.DEFAULT_SINGLETON)
        assert self.cluster is not None, "Expected cluster to be set."

        # frozen dataclass, so use object.__setattr__
        object.__setattr__(
            self, "fallback_mode", StreamingFallbackMode(self.fallback_mode)
        )
        if (
            isinstance(self.target_partition_size, int)
            and self.target_partition_size < 1
        ):
            object.__setattr__(
                self,
                "target_partition_size",
                default_target_partition_size(self.min_device_size),
            )
        if isinstance(self.broadcast_limit, int) and self.broadcast_limit < 1:
            object.__setattr__(
                self,
                "broadcast_limit",
                default_broadcast_limit(self.min_device_size),
            )
        object.__setattr__(self, "cluster", Cluster(self.cluster))

        # Handle dynamic_planning.
        # Can be None, dict, or DynamicPlanningOptions
        if isinstance(self.dynamic_planning, dict):
            object.__setattr__(
                self,
                "dynamic_planning",
                DynamicPlanningOptions(**self.dynamic_planning),
            )

        if isinstance(self.join_filter_pushdown, dict):
            object.__setattr__(
                self,
                "join_filter_pushdown",
                JoinFilterPushdownOptions(**self.join_filter_pushdown),
            )
        if self.join_filter_pushdown is not None and not isinstance(
            self.join_filter_pushdown, JoinFilterPushdownOptions
        ):
            raise TypeError(
                "join_filter_pushdown must be a JoinFilterPushdownOptions "
                "instance, dict, or None"
            )

        if self.cluster in ("spmd", "ray", "dask"):
            if self.sink_to_directory is False:
                raise ValueError(
                    f"The {self.cluster} cluster requires sink_to_directory=True"
                )
            object.__setattr__(self, "sink_to_directory", True)
        elif self.sink_to_directory is None:
            object.__setattr__(self, "sink_to_directory", False)

        # Type / value check everything else
        if not isinstance(self.max_rows_per_partition, int):
            raise TypeError("max_rows_per_partition must be an int")
        if not isinstance(self.target_partition_size, int):
            raise TypeError("target_partition_size must be an int")
        if not isinstance(self.broadcast_limit, int):
            raise TypeError("broadcast_limit must be an int")
        if not isinstance(self.sink_to_directory, bool):
            raise TypeError("sink_to_directory must be bool")
        if not isinstance(self.client_device_threshold, float):
            raise TypeError("client_device_threshold must be a float")
        if not isinstance(self.max_concurrent_io_tasks, int):
            raise TypeError("max_concurrent_io_tasks must be an int")
        if not isinstance(self.num_py_executors, int):
            raise TypeError("num_py_executors must be an int")

    def __hash__(self) -> int:  # noqa: D105
        # dynamic_planning factory, a dataclass, isn't natively hashable. We'll dump it
        # to json and hash that.
        d = dataclasses.asdict(self)
        d["dynamic_planning"] = json.dumps(d["dynamic_planning"])
        d["join_filter_pushdown"] = json.dumps(d["join_filter_pushdown"])

        # Hash the quent context UUIDs as ints
        quent_context = d["quent_context"]
        if quent_context is not None:
            for key in ["engine", "query_group", "query"]:
                quent_context[key]["id"] = int(quent_context[key]["id"])
            d["quent_context"] = json.dumps(quent_context)
        return hash(tuple(sorted(d.items())))

    def drop_unserializable(self) -> StreamingExecutor:
        """
        Return a copy without the per-cluster contexts that cannot be pickled.

        The streaming executor holds live, process-local handles (communicators,
        streaming contexts, thread pools) that must not be shipped to a worker/actor.

        Returns
        -------
        A copy of this executor with the cluster contexts set to ``None``.
        """
        return dataclasses.replace(
            self, spmd_context=None, ray_context=None, dask_context=None
        )


@dataclasses.dataclass(frozen=True, eq=True)
class InMemoryExecutor:
    """
    Configuration for the cudf-polars in-memory executor.

    The in-memory executor only supports single-GPU execution.
    """

    name: Literal["in-memory"] = dataclasses.field(default="in-memory", init=False)

    def drop_unserializable(self) -> InMemoryExecutor:
        """Return ``self``, the in-memory executor holds no unserializable state."""
        return self


ExecutorType = TypeVar("ExecutorType", StreamingExecutor, InMemoryExecutor)


@dataclasses.dataclass(frozen=True, eq=True)
class ConfigOptions(Generic[ExecutorType]):
    """
    Configuration for the polars GPUEngine.

    Parameters
    ----------
    raise_on_fail
        Whether to raise an exception when the GPU engine cannot execute a
        query. ``False`` by default.
    parquet_options
        Options controlling parquet file reading and writing. See
        :class:`~cudf_polars.utils.config.ParquetOptions` for more.
    executor
        The executor to use for the GPU engine. See :class:`~cudf_polars.utils.config.StreamingExecutor`
        and :class:`~cudf_polars.utils.config.InMemoryExecutor` for more.
    device
        The GPU used to run the query. If not provided, the
        query uses the current CUDA device.
    """

    raise_on_fail: bool = False
    parquet_options: ParquetOptions = dataclasses.field(default_factory=ParquetOptions)
    # We need the type-ignore to pass type checking. Because StreamingExecutor
    # is in ExecutorType, this is safe.
    executor: ExecutorType = dataclasses.field(
        default_factory=StreamingExecutor  # type: ignore[assignment]
    )
    device: int | None = None
    memory_resource_config: MemoryResourceConfig | None = None

    def drop_unserializable(self) -> ConfigOptions[ExecutorType]:
        """
        Return a copy safe to pickle to a worker/actor.

        Returns
        -------
        A copy of these options with unserializable executor state removed.
        """
        return dataclasses.replace(self, executor=self.executor.drop_unserializable())

    @classmethod
    def from_polars_engine(
        cls, engine: polars.lazyframe.engine_config.GPUEngine
    ) -> ConfigOptions[ExecutorType]:
        """Create a :class:`ConfigOptions` from a :class:`~polars.lazyframe.engine_config.GPUEngine`."""
        # these are the valid top-level keys in the engine.config that
        # the user passes as **kwargs to GPUEngine.
        valid_options = {
            "executor",
            "executor_options",
            "parquet_options",
            "raise_on_fail",
            "memory_resource_config",
            "hardware_binding",
        }

        extra_options = set(engine.config.keys()) - valid_options
        if extra_options:
            raise TypeError(f"Unsupported executor_options: {extra_options}")

        env_prefix = "CUDF_POLARS"
        user_executor = engine.config.get("executor")
        if user_executor is None:
            user_executor = os.environ.get(f"{env_prefix}__EXECUTOR", "streaming")
        user_executor_options = engine.config.get("executor_options", {})
        user_parquet_options = engine.config.get("parquet_options", {})
        if user_parquet_options is None:
            user_parquet_options = {}

        if isinstance(user_parquet_options, dict):
            parquet_options = ParquetOptions(**user_parquet_options)
        else:
            parquet_options = user_parquet_options
        # This is set in polars, and so can't be overridden by the environment
        user_raise_on_fail = engine.config.get("raise_on_fail", False)
        user_memory_resource_config = engine.config.get("memory_resource_config", None)
        if user_memory_resource_config is None and (
            os.environ.get(f"{MemoryResourceConfig._env_prefix}__QUALNAME", "") != ""
        ):
            # We'll pick up the qualname / options from the environment.
            user_memory_resource_config = MemoryResourceConfig()
        elif isinstance(user_memory_resource_config, dict):
            user_memory_resource_config = MemoryResourceConfig(
                **user_memory_resource_config
            )

        # These are user-provided options, so we need to actually validate
        # them.

        if user_executor not in {"in-memory", "streaming"}:
            raise ValueError(f"Unknown executor '{user_executor}'")

        if not isinstance(user_raise_on_fail, bool):
            raise TypeError("GPUEngine option 'raise_on_fail' must be a boolean.")

        executor: InMemoryExecutor | StreamingExecutor

        match user_executor:
            case "in-memory":
                executor = InMemoryExecutor(**user_executor_options)
                if parquet_options.prefetch_file_metadata:
                    raise NotImplementedError(
                        "Prefetching is not supported for the in-memory executor."
                    )
            case "streaming":
                user_executor_options = user_executor_options.copy()
                if "min_device_size" not in user_executor_options:
                    user_executor_options["min_device_size"] = get_total_device_memory()

                # Handle dynamic_planning: check user config, then env var
                user_dynamic_planning = user_executor_options.get(
                    "dynamic_planning", None
                )
                if user_dynamic_planning is None:
                    env_dynamic_planning = os.environ.get(
                        "CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING", "1"
                    )
                    if not _bool_converter(env_dynamic_planning):
                        user_executor_options["dynamic_planning"] = None

                # Handle join_filter_pushdown: check user config, then env var
                user_join_filter_pushdown = user_executor_options.get(
                    "join_filter_pushdown", None
                )
                if user_join_filter_pushdown is None:
                    env_join_filter_pushdown = os.environ.get(
                        "CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN", "0"
                    )
                    if not _bool_converter(env_join_filter_pushdown):
                        user_executor_options["join_filter_pushdown"] = None

                executor = StreamingExecutor(**user_executor_options)
            case _:  # pragma: no cover; Unreachable
                raise ValueError(f"Unsupported executor: {user_executor}")

        kwargs = {
            "raise_on_fail": user_raise_on_fail,
            "parquet_options": parquet_options,
            "executor": executor,
            "device": engine.device,
            "memory_resource_config": user_memory_resource_config,
        }

        return cls(**kwargs)
