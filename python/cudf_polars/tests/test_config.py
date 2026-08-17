# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

import rmm
from rmm._cuda import gpu

import cudf_polars.callback
import cudf_polars.quent
import cudf_polars.utils.config
from cudf_polars.callback import (
    _is_concurrent_managed_access_supported,
    default_memory_resource,
    set_memory_resource,
)
from cudf_polars.dsl.ir import DataFrameScan, IRExecutionContext
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.config import (
    Cluster,
    ConfigOptions,
    DynamicPlanningOptions,
    InMemoryExecutor,
    JoinFilterPushdownOptions,
    MemoryResourceConfig,
    ParquetOptions,
    StreamingExecutor,
    Unspecified,
)
from cudf_polars.utils.cuda_stream import get_cuda_stream


def test_polars_verbose_warns(engine: pl.GPUEngine, monkeypatch: pytest.MonkeyPatch):
    def raise_unimplemented(self, *args):
        raise NotImplementedError("We don't support this")

    monkeypatch.setattr(DataFrameScan, "__init__", raise_unimplemented)
    q = pl.LazyFrame({})
    # Ensure that things raise
    assert_ir_translation_raises(q, engine, NotImplementedError)
    with (
        pl.Config(verbose=True),
        pytest.raises(pl.exceptions.ComputeError),
        pytest.warns(
            pl.exceptions.PerformanceWarning,
            match="Query execution with GPU not possible",
        ),
    ):
        # And ensure that collecting issues the correct warning.
        assert_gpu_result_equal(q, engine=engine)


def test_unsupported_config_raises():
    q = pl.LazyFrame({})

    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=pl.GPUEngine(unknown_key=True))


def test_use_device_not_current(monkeypatch):
    # This is testing the set/restore device functionality in callback
    # for the case where the device to use is not the current device and no
    # previous query has used a device.
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", None)
    monkeypatch.setattr(gpu, "setDevice", lambda arg: None)
    # Fake that the current device is 1.
    monkeypatch.setattr(gpu, "getDevice", lambda: 1)
    q = pl.LazyFrame({})
    assert_gpu_result_equal(q, engine=pl.GPUEngine(executor="in-memory", device=0))


@pytest.mark.parametrize("device", [-1, "foo"])
def test_invalid_device_raises(device, monkeypatch):
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", None)
    q = pl.LazyFrame({})
    if isinstance(device, int):
        with pytest.raises(rmm._cuda.gpu.CUDARuntimeError):
            q.collect(engine=pl.GPUEngine(device=device))
    elif isinstance(device, str):
        with pytest.raises(TypeError):
            q.collect(engine=pl.GPUEngine(device=device))


def test_failed_device_switch_does_not_set_seen_device(monkeypatch):
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", None)
    monkeypatch.setattr(gpu, "getDevice", lambda: 1)
    calls = []

    def fail_for_invalid_device(device):
        calls.append(device)
        if device == -1:
            raise RuntimeError("invalid device")

    monkeypatch.setattr(gpu, "setDevice", fail_for_invalid_device)

    with (
        pytest.raises(RuntimeError, match="invalid device"),
        cudf_polars.callback.set_device(-1),
    ):
        pass

    assert cudf_polars.callback.SEEN_DEVICE is None

    with cudf_polars.callback.set_device(0):
        pass

    assert cudf_polars.callback.SEEN_DEVICE == 0
    assert calls == [-1, 0, 1]


def test_multiple_devices_in_same_process_raise(monkeypatch):
    # A device we haven't already seen
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", 4)
    q = pl.LazyFrame({})
    with pytest.raises(RuntimeError):
        q.collect(engine=pl.GPUEngine())


@pytest.mark.parametrize("mr", [1, object()])
def test_invalid_memory_resource_raises(mr, monkeypatch):
    monkeypatch.setattr(cudf_polars.callback, "SEEN_DEVICE", None)
    q = pl.LazyFrame({})
    with pytest.raises(TypeError):
        q.collect(engine=pl.GPUEngine(memory_resource=mr))


@pytest.fixture
def clear_memory_resource_cache():
    """
    Clear the cudf_polars.callback.default_memory_resource cache before and after a test.

    This function caches memory resources for the duration of the process. Any test that
    creates a pool (e.g. ``CudaAsyncMemoryResource``) should use this fixture to ensure that
    the pool is freed after the test.
    """
    cudf_polars.callback.default_memory_resource.cache_clear()
    try:
        yield
    finally:
        cudf_polars.callback.default_memory_resource.cache_clear()


@pytest.mark.skipif(
    not _is_concurrent_managed_access_supported(),
    reason="managed memory not supported",
)
@pytest.mark.parametrize("enable_managed_memory", ["1", "0"])
def test_cudf_polars_enable_disable_managed_memory(
    monkeypatch, enable_managed_memory, clear_memory_resource_cache
):
    q = pl.LazyFrame({"a": [1, 2, 3]})

    with monkeypatch.context() as monkeycontext:
        monkeycontext.setenv(
            "POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY", enable_managed_memory
        )
        result = q.collect(engine=pl.GPUEngine(executor="in-memory"))
        mr = default_memory_resource(
            0,
            cuda_managed_memory=bool(enable_managed_memory == "1"),
            memory_resource_config=None,
        )
        if enable_managed_memory == "1":
            assert isinstance(mr, rmm.mr.PrefetchResourceAdaptor)
            assert isinstance(mr.upstream_mr, rmm.mr.PoolMemoryResource)
        else:
            assert isinstance(mr, rmm.mr.CudaAsyncMemoryResource)
        monkeycontext.delenv("POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY")
    assert_frame_equal(q.collect(), result)


def test_explicit_device_zero():
    q = pl.LazyFrame({"a": [1, 2, 3]})

    result = q.collect(engine=pl.GPUEngine(executor="in-memory", device=0))
    assert_frame_equal(q.collect(), result)


def test_explicit_memory_resource():
    upstream = rmm.mr.CudaMemoryResource()
    n_allocations = 0

    def allocate(bytes, stream):
        nonlocal n_allocations
        n_allocations += 1
        return upstream.allocate(bytes, stream)

    mr = rmm.mr.CallbackMemoryResource(allocate, upstream.deallocate)

    q = pl.LazyFrame({"a": [1, 2, 3]})
    result = q.collect(engine=pl.GPUEngine(executor="in-memory", memory_resource=mr))
    assert_frame_equal(q.collect(), result)
    assert n_allocations > 0


def test_nested_memory_resource_config():
    spec = {
        "qualname": "rmm.mr.PrefetchResourceAdaptor",
        "options": {
            "upstream_mr": {
                "qualname": "rmm.mr.PoolMemoryResource",
                "options": {
                    "upstream_mr": {
                        "qualname": "rmm.mr.ManagedMemoryResource",
                    },
                    "initial_pool_size": 256,
                },
            }
        },
    }

    engine = pl.GPUEngine(
        executor="streaming",
        memory_resource_config=MemoryResourceConfig(
            **spec,
        ),
    )
    config = ConfigOptions.from_polars_engine(engine)
    mr = config.memory_resource_config.create_memory_resource()
    assert isinstance(mr, rmm.mr.PrefetchResourceAdaptor)
    assert isinstance(mr.upstream_mr, rmm.mr.PoolMemoryResource)
    assert mr.upstream_mr.pool_size() == 256
    assert isinstance(mr.upstream_mr.upstream_mr, rmm.mr.ManagedMemoryResource)

    assert hash(config.memory_resource_config) == hash(config.memory_resource_config)


@pytest.mark.parametrize("executor", ["streaming", "in-memory"])
def test_parquet_options(executor: str) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor=executor,
        )
    )
    assert config.parquet_options.chunked is True
    assert config.parquet_options.n_output_chunks == 1
    assert config.parquet_options.use_jit_filter is False

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor=executor,
            parquet_options={
                "chunked": False,
                "n_output_chunks": 16,
                "use_jit_filter": True,
            },
        )
    )
    assert config.parquet_options.chunked is False
    assert config.parquet_options.n_output_chunks == 16
    assert config.parquet_options.use_jit_filter is True


def test_parquet_options_from_none() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            parquet_options=None,
        )
    )
    assert config.parquet_options.chunked is True


@pytest.mark.parametrize("executor", ["in-memory", "streaming"])
def test_hashable(executor: str) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor=executor,
        )
    )
    assert hash(config) == hash(config)


def test_validate_fallback_mode() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.fallback_mode == "warn"

    with pytest.raises(ValueError, match="'foo' is not a valid StreamingFallbackMode"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"fallback_mode": "foo"},
            )
        )


def test_validate_cluster() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.cluster == "default_singleton"

    with pytest.raises(ValueError, match="'foo' is not a valid Cluster"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"cluster": "foo"},
            )
        )


@pytest.mark.parametrize(
    "option",
    [
        "max_rows_per_partition",
        "target_partition_size",
        "broadcast_limit",
        "sink_to_directory",
        "client_device_threshold",
        "max_concurrent_io_tasks",
        "num_py_executors",
        "kvikio_nthreads",
    ],
)
def test_validate_streaming_executor_options(option: str) -> None:
    with pytest.raises(TypeError, match=f"{option} must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={option: object()},
            )
        )


def test_kvikio_nthreads_non_positive_raises() -> None:
    with pytest.raises(ValueError, match="kvikio_nthreads must be positive"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"kvikio_nthreads": 0},
            )
        )


def test_executor_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR", "in-memory")
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "in-memory"

    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR", "invalid")
        engine = pl.GPUEngine()
        with pytest.raises(ValueError, match="Unknown executor 'invalid'"):
            ConfigOptions.from_polars_engine(engine)


def test_parquet_options_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__CHUNKED", "0")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__N_OUTPUT_CHUNKS", "2")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__CHUNK_READ_LIMIT", "100")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__PASS_READ_LIMIT", "200")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__MAX_FOOTER_SAMPLES", "0")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__MAX_ROW_GROUP_SAMPLES", "0")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__PREFETCH_FILE_METADATA", "1")
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__USE_JIT_FILTER", "1")

        # Test default
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.parquet_options.chunked is False
        assert config.parquet_options.n_output_chunks == 2
        assert config.parquet_options.chunk_read_limit == 100
        assert config.parquet_options.pass_read_limit == 200
        assert config.parquet_options.max_footer_samples == 0
        assert config.parquet_options.max_row_group_samples == 0
        assert config.parquet_options.prefetch_file_metadata is True
        assert config.parquet_options.use_jit_filter is True

    with monkeypatch.context() as m:
        # Env must win over the executor-derived default (streaming => True).
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__PREFETCH_FILE_METADATA", "0")
        engine = pl.GPUEngine(executor="streaming")
        config = ConfigOptions.from_polars_engine(engine)
        assert config.parquet_options.prefetch_file_metadata is False

    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__PARQUET_OPTIONS__CHUNKED", "foo")
        engine = pl.GPUEngine()
        with pytest.raises(ValueError, match="Invalid boolean value: 'foo'"):
            ConfigOptions.from_polars_engine(engine)


def test_config_option_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__CLUSTER", "default_singleton")
        m.setenv("CUDF_POLARS__EXECUTOR__FALLBACK_MODE", "silent")
        m.setenv("CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION", "42")
        m.setenv("CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE", "100")
        m.setenv("CUDF_POLARS__EXECUTOR__BROADCAST_LIMIT", "44")
        m.setenv("CUDF_POLARS__EXECUTOR__MAX_CONCURRENT_IO_TASKS", "6")
        m.setenv("CUDF_POLARS__EXECUTOR__QUENT_CONTEXT", "1")

        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.cluster == "default_singleton"
        assert config.executor.fallback_mode == "silent"
        assert config.executor.max_rows_per_partition == 42
        assert config.executor.target_partition_size == 100
        assert config.executor.broadcast_limit == 44
        assert config.executor.max_concurrent_io_tasks == 6
        assert config.executor.quent_context is not None


def test_quent_context_from_env_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__QUENT_CONTEXT", "0")
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.quent_context is None


def test_quent_context_from_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__QUENT_CONTEXT", "foo")
        engine = pl.GPUEngine()
        with pytest.raises(ValueError, match="Invalid value for quent_context: 'foo'"):
            ConfigOptions.from_polars_engine(engine)


def test_hash_streaming_executor() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"quent_context": cudf_polars.quent.QuentContext()},
        )
    )
    assert hash(config.executor) == hash(config.executor)


def test_target_partition_from_env(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE", "100")

        engine = pl.GPUEngine(executor="streaming")
        ConfigOptions.from_polars_engine(engine)  # no warning
        assert len(recwarn) == 0


def test_target_partition_defaults_to_device_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.delenv("CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE", raising=False)
        m.setattr(cudf_polars.utils.config, "get_total_device_memory", lambda: 32 << 30)

        config = ConfigOptions.from_polars_engine(pl.GPUEngine(executor="streaming"))
        assert config.executor.min_device_size == 32 << 30
        assert config.executor.target_partition_size == int((32 << 30) * 0.025)


def test_fallback_mode_default(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__FALLBACK_MODE", "silent")
        engine = pl.GPUEngine(executor="streaming")
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.fallback_mode == "silent"

    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__FALLBACK_MODE", "foo")
        engine = pl.GPUEngine(executor="streaming")
        with pytest.raises(
            ValueError, match="'foo' is not a valid StreamingFallbackMode"
        ):
            ConfigOptions.from_polars_engine(engine)


@pytest.mark.parametrize(
    "option",
    [
        "chunked",
        "n_output_chunks",
        "chunk_read_limit",
        "pass_read_limit",
        "max_footer_samples",
        "max_row_group_samples",
        "prefetch_file_metadata",
        "use_jit_filter",
    ],
)
def test_validate_parquet_options(option: str) -> None:
    with pytest.raises(TypeError, match=f"{option} must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                parquet_options={option: object()},
            )
        )


def test_prefetch_file_metadata_default() -> None:
    config = ConfigOptions.from_polars_engine(pl.GPUEngine(executor="streaming"))
    assert isinstance(config.parquet_options.prefetch_file_metadata, Unspecified)

    config = ConfigOptions.from_polars_engine(pl.GPUEngine(executor="in-memory"))
    assert config.parquet_options.prefetch_file_metadata is False

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming", parquet_options={"prefetch_file_metadata": True}
        )
    )
    assert config.parquet_options.prefetch_file_metadata is True


def test_parquet_options_object_passthrough() -> None:
    parquet_options = ParquetOptions(prefetch_file_metadata=False)
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(executor="streaming", parquet_options=parquet_options)
    )
    assert config.parquet_options is parquet_options


def test_parquet_options_object_engine_default() -> None:
    # If a user passes in a ParquetOptions object instead of a plain dict, and
    # doesn't set prefetch_file_metadata on it, we still need to fill in the
    # right default for the chosen executor.
    parquet_options = ParquetOptions()
    assert isinstance(parquet_options.prefetch_file_metadata, Unspecified)

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(executor="in-memory", parquet_options=parquet_options)
    )
    assert config.parquet_options.prefetch_file_metadata is False

    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(executor="streaming", parquet_options=parquet_options)
    )
    assert isinstance(config.parquet_options.prefetch_file_metadata, Unspecified)


def test_validate_raise_on_fail() -> None:
    with pytest.raises(TypeError, match="'raise_on_fail' must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(executor="streaming", raise_on_fail=cast("bool", object()))
        )


def test_validate_executor() -> None:
    with pytest.raises(ValueError, match="Unknown executor 'foo'"):
        ConfigOptions.from_polars_engine(pl.GPUEngine(executor="foo"))


def test_default_executor() -> None:
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"


@pytest.mark.parametrize(
    "memory_resource, memory_resource_config",
    [
        (None, None),
        (
            None,
            MemoryResourceConfig(
                qualname="rmm.mr.CudaAsyncMemoryResource",
                options={"initial_pool_size": 123, "release_threshold": 456},
            ),
        ),
        (rmm.mr.CudaAsyncMemoryResource(initial_pool_size=100), None),
        # prioritize the concrete MR
        (
            rmm.mr.CudaAsyncMemoryResource(initial_pool_size=100),
            MemoryResourceConfig(qualname="rmm.mr.CudaMemoryResource"),
        ),
    ],
)
def test_memory_resource(memory_resource, memory_resource_config) -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            memory_resource=memory_resource,
            memory_resource_config=memory_resource_config,
        )
    )

    with set_memory_resource(
        memory_resource, memory_resource_config, config.executor
    ) as result:
        if memory_resource is None and memory_resource_config is None:
            # The default case: We make a new RMM MR, whose type depends on the GPU's features.

            assert isinstance(result, rmm.mr.CudaAsyncMemoryResource)

        elif memory_resource is None:
            # Configured through memory_resource_config
            assert isinstance(result, rmm.mr.CudaAsyncMemoryResource)
            assert config.memory_resource_config is not None
            assert (
                config.memory_resource_config.qualname
                == "rmm.mr.CudaAsyncMemoryResource"
            )
            assert config.memory_resource_config.options == {
                "initial_pool_size": 123,
                "release_threshold": 456,
            }
            assert isinstance(
                config.memory_resource_config.create_memory_resource(),
                rmm.mr.CudaAsyncMemoryResource,
            )

        elif memory_resource is not None:
            assert result is memory_resource
        else:  # pragma: no cover; Unreachable
            raise ValueError("Unreachable")


def test_memory_resource_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setenv(
            "CUDF_POLARS__MEMORY_RESOURCE_CONFIG__QUALNAME",
            "rmm.mr.CudaAsyncMemoryResource",
        )
        m.setenv(
            "CUDF_POLARS__MEMORY_RESOURCE_CONFIG__OPTIONS",
            '{"initial_pool_size": 123, "release_threshold": 456}',
        )
        engine = pl.GPUEngine()
        config = ConfigOptions.from_polars_engine(engine)
        assert config.memory_resource_config is not None
        assert (
            config.memory_resource_config.qualname == "rmm.mr.CudaAsyncMemoryResource"
        )
        assert config.memory_resource_config.options == {
            "initial_pool_size": 123,
            "release_threshold": 456,
        }


def test_ir_execution_context() -> None:
    context = IRExecutionContext()
    assert context.get_cuda_stream is get_cuda_stream
    context.get_cuda_stream()  # no exception


def test_validate_dynamic_planning() -> None:
    with pytest.raises(TypeError, match="sample_chunk_count must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"dynamic_planning": {"sample_chunk_count": object()}},
            )
        )


def test_dynamic_planning_sample_chunk_count_min() -> None:
    with pytest.raises(ValueError, match="sample_chunk_count must be at least 1"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"dynamic_planning": {"sample_chunk_count": 0}},
            )
        )


def test_dynamic_planning_defaults() -> None:
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
    # Dynamic planning is enabled by default
    assert config.executor.dynamic_planning is not None
    assert config.executor.dynamic_planning.sample_chunk_count == 2
    assert config.executor.join_filter_pushdown is None


def test_dynamic_planning_disabled_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that env var can disable dynamic planning
    monkeypatch.setenv("CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING", "0")
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
    assert config.executor.dynamic_planning is None


def test_dynamic_planning_sample_chunk_count_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test that sample_chunk_count_reduce can be configured via env var
    monkeypatch.setenv(
        "CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING__SAMPLE_CHUNK_COUNT", "3"
    )
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.name == "streaming"
    assert config.executor.dynamic_planning is not None
    assert config.executor.dynamic_planning.sample_chunk_count == 3


def test_join_filter_pushdown_options_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN", "1")
    monkeypatch.setenv(
        "CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN__THRESHOLD", "0.125"
    )
    monkeypatch.setenv("CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN__TRACE", "1")
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.join_filter_pushdown is not None
    assert config.executor.join_filter_pushdown.threshold == 0.125
    assert config.executor.join_filter_pushdown.trace


def test_join_filter_pushdown_disabled_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN", "0")
    monkeypatch.setenv("CUDF_POLARS__EXECUTOR__JOIN_FILTER_PUSHDOWN__TRACE", "1")
    config = ConfigOptions.from_polars_engine(pl.GPUEngine())
    assert config.executor.join_filter_pushdown is None


def test_validate_join_filter_pushdown_options() -> None:
    with pytest.raises(TypeError, match="threshold must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"join_filter_pushdown": {"threshold": "bad"}},
            )
        )
    with pytest.raises(ValueError, match="threshold must be between"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"join_filter_pushdown": {"threshold": 1.5}},
            )
        )
    with pytest.raises(TypeError, match="trace must be"):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"join_filter_pushdown": {"trace": "bad"}},
            )
        )


def test_validate_join_filter_pushdown_type() -> None:
    with pytest.raises(
        TypeError,
        match="join_filter_pushdown must be a JoinFilterPushdownOptions instance",
    ):
        ConfigOptions.from_polars_engine(
            pl.GPUEngine(
                executor="streaming",
                executor_options={"join_filter_pushdown": object()},
            )
        )


def test_join_filter_pushdown_from_instance() -> None:
    options = JoinFilterPushdownOptions(threshold=0.25, trace=True)
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"join_filter_pushdown": options},
        )
    )
    assert config.executor.join_filter_pushdown is options


def test_join_filter_pushdown_disabled_from_options() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"join_filter_pushdown": None},
        )
    )
    assert config.executor.join_filter_pushdown is None
    assert hash(config) == hash(config)


def test_dynamic_planning_from_instance() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"dynamic_planning": DynamicPlanningOptions()},
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.dynamic_planning is not None
    assert config.executor.dynamic_planning.sample_chunk_count == 2  # default


def test_parse_memory_resource_config() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            memory_resource_config={
                "qualname": "rmm.mr.CudaAsyncMemoryResource",
                "options": {
                    "initial_pool_size": 123,
                    "release_threshold": 456,
                },
            }
        )
    )
    assert isinstance(config.memory_resource_config, MemoryResourceConfig)
    assert config.memory_resource_config.qualname == "rmm.mr.CudaAsyncMemoryResource"


def test_memory_resource_config_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"MemoryResourceConfig.qualname 'foo' must be a fully qualified name to a class",
    ):
        MemoryResourceConfig(qualname="foo")


@pytest.mark.parametrize("options", [None, {}])
def test_memory_resource_config_hash(options) -> None:
    config = MemoryResourceConfig(qualname="rmm.mr.CudaMemoryResource", options=options)
    assert hash(config) == hash(config)


def test_num_py_executors_default() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.num_py_executors == 8


def test_num_py_executors_from_executor_options() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"num_py_executors": 4},
        )
    )
    assert config.executor.name == "streaming"
    assert config.executor.num_py_executors == 4


def test_num_py_executors_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS", "8")
        engine = pl.GPUEngine(executor="streaming")
        config = ConfigOptions.from_polars_engine(engine)
        assert config.executor.name == "streaming"
        assert config.executor.num_py_executors == 8


def test_kvikio_nthreads_default(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.delenv("CUDF_POLARS__EXECUTOR__KVIKIO_NTHREADS", raising=False)
        m.delenv("KVIKIO_NTHREADS", raising=False)
        config = ConfigOptions.from_polars_engine(pl.GPUEngine(executor="streaming"))
        assert config.executor.kvikio_nthreads == 256


def test_kvikio_nthreads_from_executor_options() -> None:
    config = ConfigOptions.from_polars_engine(
        pl.GPUEngine(
            executor="streaming",
            executor_options={"kvikio_nthreads": 128},
        )
    )
    assert config.executor.kvikio_nthreads == 128


def test_kvikio_nthreads_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__KVIKIO_NTHREADS", "64")
        config = ConfigOptions.from_polars_engine(pl.GPUEngine(executor="streaming"))
        assert config.executor.kvikio_nthreads == 64


def test_kvikio_nthreads_from_kvikio_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.delenv("CUDF_POLARS__EXECUTOR__KVIKIO_NTHREADS", raising=False)
        m.setenv("KVIKIO_NTHREADS", "32")
        config = ConfigOptions.from_polars_engine(pl.GPUEngine(executor="streaming"))
        assert config.executor.kvikio_nthreads == 32


def test_kvikio_nthreads_cudf_polars_env_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("CUDF_POLARS__EXECUTOR__KVIKIO_NTHREADS", "64")
        m.setenv("KVIKIO_NTHREADS", "32")
        config = ConfigOptions.from_polars_engine(pl.GPUEngine(executor="streaming"))
        assert config.executor.kvikio_nthreads == 64


def test_dask_sink_to_directory_false_raises() -> None:
    with pytest.raises(
        ValueError, match="The dask cluster requires sink_to_directory=True"
    ):
        StreamingExecutor(cluster=Cluster.DASK, sink_to_directory=False)


def test_in_memory_executor_drop_unserializable() -> None:
    executor = InMemoryExecutor()
    assert executor.drop_unserializable() is executor
