# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for StreamingOptions."""

from __future__ import annotations

import argparse
import json
import os

import pytest

from cudf_polars.engine.hardware_binding import HardwareBindingPolicy
from cudf_polars.engine.options import (
    UNSPECIFIED,
    StreamingOptions,
    Unspecified,
    _parse_memory_resource_config,
)
from cudf_polars.utils.config import DynamicPlanningOptions, MemoryResourceConfig

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------


def test_unspecified_is_singleton() -> None:
    assert Unspecified() is UNSPECIFIED


def test_unspecified_repr() -> None:
    assert repr(UNSPECIFIED) == "UNSPECIFIED"


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


def test_all_fields_unspecified_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear all env vars that StreamingOptions reads so every field stays UNSPECIFIED.
    for key in list(os.environ):
        if key.startswith(("RAPIDSMPF_", "CUDF_POLARS__")):
            monkeypatch.delenv(key, raising=False)
    opts = StreamingOptions()
    # Fields with no env var are always UNSPECIFIED.
    assert isinstance(opts.raise_on_fail, Unspecified)
    assert isinstance(opts.parquet_options, Unspecified)
    # Fields whose env vars were cleared are also UNSPECIFIED.
    assert isinstance(opts.fallback_mode, Unspecified)
    assert isinstance(opts.log, Unspecified)


# ---------------------------------------------------------------------------
# to_executor_options
# ---------------------------------------------------------------------------


def test_executor_options_empty_when_all_unspecified() -> None:
    assert StreamingOptions().to_executor_options() == {}


def test_executor_options_includes_set_fields() -> None:
    opts = StreamingOptions(fallback_mode="raise", max_rows_per_partition=500_000)
    result = opts.to_executor_options()
    assert result["fallback_mode"] == "raise"
    assert result["max_rows_per_partition"] == 500_000
    assert "log" not in result


def test_executor_options_num_py_executors() -> None:
    result = StreamingOptions(num_py_executors=4).to_executor_options()
    assert result["num_py_executors"] == 4


def test_executor_options_max_concurrent_io_tasks() -> None:
    result = StreamingOptions(max_concurrent_io_tasks=6).to_executor_options()
    assert result["max_concurrent_io_tasks"] == 6


def test_executor_options_kvikio_nthreads() -> None:
    result = StreamingOptions(kvikio_nthreads=128).to_executor_options()
    assert result["kvikio_nthreads"] == 128


@pytest.mark.parametrize("value", [True, False])
def test_executor_options_sink_to_directory(*, value: bool) -> None:
    result = StreamingOptions(sink_to_directory=value).to_executor_options()
    assert result["sink_to_directory"] is value


def test_executor_options_sink_to_directory_absent_when_unspecified() -> None:
    assert "sink_to_directory" not in StreamingOptions().to_executor_options()


def test_executor_options_join_filter_pushdown_disabled() -> None:
    result = StreamingOptions(join_filter_pushdown=None).to_executor_options()
    assert result["join_filter_pushdown"] is None


# ---------------------------------------------------------------------------
# to_engine_options
# ---------------------------------------------------------------------------


def test_engine_options_empty_when_all_unspecified() -> None:
    assert StreamingOptions().to_engine_options() == {}


def test_engine_options_includes_set_fields() -> None:
    result = StreamingOptions(raise_on_fail=True).to_engine_options()
    assert result["raise_on_fail"] is True
    assert "log" not in result


# ---------------------------------------------------------------------------
# to_rapidsmpf_options
# ---------------------------------------------------------------------------


def test_rapidsmpf_options_serialized() -> None:
    opts = StreamingOptions(
        statistics=True,
        pinned_memory=False,
        num_streaming_threads=8,
        ucxx_progress_mode="thread-polling",
        log="DEBUG",
        pinned_max_pool_size="4GiB",
        unbounded_file_read_cache="host",
    )
    strings = opts.to_rapidsmpf_options().get_strings()
    assert strings["statistics"] == "True"
    assert strings["pinned_memory"] == "False"
    assert strings["num_streaming_threads"] == "8"
    assert strings["ucxx_progress_mode"] == "thread-polling"
    assert strings["log"] == "DEBUG"
    assert strings["pinned_max_pool_size"] == "4GiB"
    assert strings["unbounded_file_read_cache"] == "host"


def test_rapidsmpf_options_unspecified_fields_absent() -> None:
    env_backup = {
        k: os.environ.pop(k) for k in list(os.environ) if k.startswith("RAPIDSMPF_")
    }
    try:
        assert StreamingOptions().to_rapidsmpf_options().get_strings() == {}
    finally:
        os.environ.update(env_backup)


def test_rapidsmpf_options_picks_up_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPIDSMPF_LOG", "TRACE")
    assert StreamingOptions().to_rapidsmpf_options().get_strings()["log"] == "TRACE"


def test_rapidsmpf_options_explicit_overrides_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAPIDSMPF_LOG", "TRACE")
    strings = StreamingOptions(log="WARN").to_rapidsmpf_options().get_strings()
    assert strings["log"] == "WARN"


def test_rapidsmpf_options_env_var_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAPIDSMPF_LOG", raising=False)
    assert "log" not in StreamingOptions().to_rapidsmpf_options().get_strings()


def test_ucxx_progress_mode_picks_up_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAPIDSMPF_UCXX_PROGRESS_MODE", "polling")
    strings = StreamingOptions().to_rapidsmpf_options().get_strings()
    assert strings["ucxx_progress_mode"] == "polling"


def test_ucxx_progress_mode_explicit_overrides_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAPIDSMPF_UCXX_PROGRESS_MODE", "polling")
    strings = (
        StreamingOptions(ucxx_progress_mode="thread-polling")
        .to_rapidsmpf_options()
        .get_strings()
    )
    assert strings["ucxx_progress_mode"] == "thread-polling"


def test_ucxx_progress_mode_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAPIDSMPF_UCXX_PROGRESS_MODE", raising=False)
    strings = StreamingOptions().to_rapidsmpf_options().get_strings()
    assert "ucxx_progress_mode" not in strings


def test_pinned_max_pool_size_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPIDSMPF_PINNED_MAX_POOL_SIZE", "4GiB")
    strings = StreamingOptions().to_rapidsmpf_options().get_strings()
    assert strings["pinned_max_pool_size"] == "4GiB"


def test_unbounded_file_read_cache_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAPIDSMPF_UNBOUNDED_FILE_READ_CACHE", "host")
    strings = StreamingOptions().to_rapidsmpf_options().get_strings()
    assert strings["unbounded_file_read_cache"] == "host"


# ---------------------------------------------------------------------------
# memory_resource_config forwarding
# ---------------------------------------------------------------------------


def test_parse_memory_resource_config() -> None:
    """_parse_memory_resource_config converts a JSON string to MemoryResourceConfig."""
    config = _parse_memory_resource_config('{"qualname": "rmm.mr.CudaMemoryResource"}')
    assert isinstance(config, MemoryResourceConfig)
    assert config.qualname == "rmm.mr.CudaMemoryResource"


def test_from_argparse_memory_resource_config_passthrough() -> None:
    """MemoryResourceConfig instances pass through _from_argparse unchanged."""
    config = MemoryResourceConfig(qualname="rmm.mr.CudaMemoryResource")
    ns = argparse.Namespace(memory_resource_config=config)
    opts = StreamingOptions._from_argparse(ns)
    assert opts.memory_resource_config is config


def test_cli_memory_resource_config_roundtrip() -> None:
    """--memory-resource-config JSON roundtrips through CLI parsing."""
    parser = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(parser)
    args = parser.parse_args(
        ["--memory-resource-config", '{"qualname": "rmm.mr.CudaAsyncMemoryResource"}']
    )
    opts = StreamingOptions._from_argparse(args)
    assert isinstance(opts.memory_resource_config, MemoryResourceConfig)
    assert opts.memory_resource_config.qualname == "rmm.mr.CudaAsyncMemoryResource"


def test_memory_resource_config_in_engine_options() -> None:
    """memory_resource_config is included in to_engine_options() when set."""
    config = MemoryResourceConfig(qualname="rmm.mr.CudaMemoryResource")
    opts = StreamingOptions(memory_resource_config=config)
    assert opts.to_engine_options()["memory_resource_config"] is config


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_empty_equals_default() -> None:
    assert StreamingOptions.from_dict({}) == StreamingOptions()


def test_from_dict_maps_known_fields() -> None:
    opts = StreamingOptions.from_dict(
        {"fallback_mode": "raise", "num_streaming_threads": 8}
    )
    assert opts.fallback_mode == "raise"
    assert opts.num_streaming_threads == 8
    assert isinstance(opts.log, Unspecified)


def test_from_dict_none_value_is_unspecified() -> None:
    opts = StreamingOptions.from_dict({"fallback_mode": None})
    assert isinstance(opts.fallback_mode, Unspecified)


def test_from_dict_unknown_key_raises() -> None:
    with pytest.raises(TypeError):
        StreamingOptions.from_dict({"no_such_field": 42})


def test_from_dict_roundtrip() -> None:
    original = StreamingOptions(fallback_mode="silent", num_streaming_threads=4)
    reconstructed = StreamingOptions.from_dict(
        {"fallback_mode": "silent", "num_streaming_threads": 4}
    )
    assert reconstructed == original


# ---------------------------------------------------------------------------
# _from_argparse
# ---------------------------------------------------------------------------


def test_from_argparse_empty_namespace_equals_default() -> None:
    assert StreamingOptions._from_argparse(argparse.Namespace()) == StreamingOptions()


def test_from_argparse_direct_fields() -> None:
    opts = StreamingOptions._from_argparse(
        argparse.Namespace(fallback_mode="raise", max_rows_per_partition=500_000)
    )
    assert opts.fallback_mode == "raise"
    assert opts.max_rows_per_partition == 500_000


def test_from_argparse_renames() -> None:
    opts = StreamingOptions._from_argparse(
        argparse.Namespace(
            rapidsmpf_log="DEBUG",
            rapidsmpf_statistics=True,
            blocksize=1_000_000,
        )
    )
    assert opts.log == "DEBUG"
    assert opts.statistics is True
    assert opts.target_partition_size == 1_000_000


def test_from_argparse_dynamic_planning() -> None:
    assert isinstance(
        StreamingOptions._from_argparse(
            argparse.Namespace(dynamic_planning=None)
        ).dynamic_planning,
        Unspecified,
    )
    assert (
        StreamingOptions._from_argparse(
            argparse.Namespace(dynamic_planning=True)
        ).dynamic_planning
        == DynamicPlanningOptions()
    )
    assert (
        StreamingOptions._from_argparse(
            argparse.Namespace(dynamic_planning=False)
        ).dynamic_planning
        is None
    )


# ---------------------------------------------------------------------------
# _add_cli_args
# ---------------------------------------------------------------------------


def test_add_cli_args_then_from_argparse_roundtrip() -> None:
    parser = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(parser)
    args = parser.parse_args(
        [
            "--num-streaming-threads",
            "8",
            "--rapidsmpf-log",
            "DEBUG",
            "--raise-on-fail",
            "--pinned-max-pool-size",
            "4GiB",
            "--unbounded-file-read-cache",
            "host",
            "--max-concurrent-io-tasks",
            "6",
        ]
    )
    opts = StreamingOptions._from_argparse(args)
    assert opts.num_streaming_threads == 8
    assert opts.log == "DEBUG"
    assert opts.raise_on_fail is True
    assert opts.pinned_max_pool_size == "4GiB"
    assert opts.unbounded_file_read_cache == "host"
    assert opts.max_concurrent_io_tasks == 6
    # Unprovided args default to None → UNSPECIFIED
    assert isinstance(opts.fallback_mode, Unspecified)


def test_from_argparse_omitted_flag_still_picks_up_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS", "16")
    monkeypatch.setenv("RAPIDSMPF_NUM_STREAMING_THREADS", "16")

    parser = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(parser)
    args = parser.parse_args([])
    opts = StreamingOptions._from_argparse(args)

    assert opts.num_py_executors == 16
    assert opts.to_executor_options()["num_py_executors"] == 16
    assert opts.num_streaming_threads == 16


# ---------------------------------------------------------------------------
# to_dict / roundtrip
# ---------------------------------------------------------------------------


def test_to_dict_empty_when_all_unspecified() -> None:
    assert StreamingOptions().to_dict() == {}


def test_to_dict_contains_only_set_fields() -> None:
    opts = StreamingOptions(fallback_mode="silent", num_streaming_threads=4)
    d = opts.to_dict()
    assert d == {"fallback_mode": "silent", "num_streaming_threads": 4}


def test_to_dict_roundtrip() -> None:
    original = StreamingOptions(
        fallback_mode="silent",
        num_streaming_threads=4,
        log="DEBUG",
        raise_on_fail=True,
    )
    assert StreamingOptions.from_dict(original.to_dict()) == original


def test_to_dict_roundtrip_empty() -> None:
    opts = StreamingOptions()
    assert StreamingOptions.from_dict(opts.to_dict()) == opts


# ---------------------------------------------------------------------------
# hardware_binding
# ---------------------------------------------------------------------------


def test_hardware_binding_in_engine_options() -> None:
    result = StreamingOptions(
        hardware_binding=HardwareBindingPolicy(enabled=False)
    ).to_engine_options()
    assert result["hardware_binding"] == HardwareBindingPolicy(enabled=False)


def test_hardware_binding_env_var_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDF_POLARS__HARDWARE_BINDING", '{"enabled": false}')
    result = StreamingOptions().to_engine_options()
    assert result["hardware_binding"] == HardwareBindingPolicy(enabled=False)


def test_hardware_binding_cli_json() -> None:
    parser = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(parser)
    args = parser.parse_args(["--hardware-binding", '{"raise_on_fail": true}'])
    opts = StreamingOptions._from_argparse(args)
    assert opts.hardware_binding == HardwareBindingPolicy(raise_on_fail=True)


def test_hardware_binding_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDF_POLARS__HARDWARE_BINDING", "not json")
    with pytest.raises(json.JSONDecodeError):
        StreamingOptions()


def test_hardware_binding_cli_disabled() -> None:
    parser = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(parser)
    args = parser.parse_args(["--hardware-binding", '{"enabled": false}'])
    opts = StreamingOptions._from_argparse(args)
    assert opts.hardware_binding == HardwareBindingPolicy(enabled=False)
