# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest

import polars as pl

structlog = pytest.importorskip("structlog")


@pytest.fixture(name="log_output")
def fixture_log_output():
    return structlog.testing.LogCapture()


@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output):
    structlog.configure(processors=[log_output])


def test_trace_basic(
    log_output: Any,
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: int,
) -> None:
    # Whether tracing is enabled is determined when cudf_polars is imported.
    # So our best way of testing this is to run things in a subprocess
    # to control the environment and isolate it from the rest of the test suite.
    code = textwrap.dedent("""\
    import polars as pl
    import rmm

    q = pl.DataFrame({"a": [1, 2, 3]}).lazy().select(pl.col("a").sum())
    q.collect(
        engine=pl.GPUEngine(
            executor="streaming", memory_resource=rmm.mr.ManagedMemoryResource()
        )
    )
    """)

    env = {
        "CUDF_POLARS_LOG_TRACES": "1",
    }

    result = subprocess.check_output(
        [sys.executable, "-c", code], env=env, timeout=timeout_seconds
    )
    # Just ensure that the default structlog output is in the result
    assert b"Execute IR" in result
    assert b"frames_output" in result
    assert b"frames_input" in result
    assert b"total_bytes_output" in result
    assert b"total_bytes_input" in result
    # TODO: With rapidsmpf are the rmm fields not supposed to be logged?
    assert b"rmm_total_bytes_output" not in result
    assert b"rmm_total_bytes_input" not in result
    assert b"rmm_current_bytes_output" not in result
    assert b"overhead_duration" in result


def test_import_without_structlog(timeout_seconds: int) -> None:
    # This test could avoid the subprocess by monkeypatching sys.modules, but
    # that was flaky. https://github.com/rapidsai/cudf/pull/22012#issuecomment-4284536686
    # has more details.
    code = textwrap.dedent("""\
    import sys
    sys.modules["structlog"] = None

    import cudf_polars.dsl.tracing
    assert not cudf_polars.dsl.tracing._HAS_STRUCTLOG

    import polars as pl
    q = pl.DataFrame({"a": [1, 2, 3]}).lazy().select(pl.col("a").sum())
    q.collect(engine="gpu")
    """)
    subprocess.check_call([sys.executable, "-c", code], timeout=timeout_seconds)


def test_log_query_plan(timeout_seconds: int) -> None:
    """Test that log_query_plan emits a Query Plan event."""
    import os

    code = textwrap.dedent("""\
    import polars as pl
    import rmm

    df = pl.DataFrame({"x": range(10), "y": ["a", "b"] * 5})
    q = df.lazy().filter(pl.col("x") > 5).group_by("y").agg(pl.col("x").sum())
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": "default_singleton",
            "max_rows_per_partition": 5,
        },
        memory_resource=rmm.mr.ManagedMemoryResource(),
    )
    q.collect(engine=engine)
    """)

    env = os.environ.copy()
    env["CUDF_POLARS_LOG_TRACES"] = "1"

    result = subprocess.check_output(
        [sys.executable, "-c", code],
        env=env,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
    )

    # Check for Query Plan event generated from SerializablePlan
    assert b"Query Plan" in result
    assert b"scope=plan" in result or b"'scope': 'plan'" in result
    assert b"actor_ir_id" in result
    assert b"actor_ir_type" in result
    assert b"children" in result


@pytest.mark.skipif(
    os.environ.get("CUDF_POLARS_LOG_TRACES") != "1",
    reason="Requires CUDF_POLARS_LOG_TRACES=1.",
)
def test_sets_cudf_polars_query_id():
    left = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    right = pl.LazyFrame({"a": [1, 2, 3], "c": [7, 8, 9]})

    q = left.join(right, on="a", how="inner").select(
        pl.col("b") + pl.col("c").alias("d")
    )
    engine = pl.GPUEngine(
        executor="streaming",
        raise_on_fail=True,
    )

    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as cap:
        q.collect(engine=engine)

    assert len(cap) > 0
    plan_log = cap[0]
    assert "scope" in plan_log
    assert plan_log["scope"] == "plan"
    assert "cudf_polars_query_id" in plan_log
    query_id = plan_log["cudf_polars_query_id"]

    for log in cap:
        assert "scope" in log
        assert "cudf_polars_query_id" in log
        assert log["cudf_polars_query_id"] == query_id

        match log["scope"]:
            case "plan":
                assert "plan" in log
            case "actor":
                keys = set(log.keys())
                assert keys >= {
                    "actor_ir_id",
                    "actor_ir_type",
                    "cudf_polars_query_id",
                    "duplicated",
                    "scope",
                    "start",
                    "stop",
                }
            case "evaluate_ir_node":
                keys = set(log.keys())
                assert keys >= {
                    "start",
                    "stop",
                    "cudf_polars_query_id",
                    "type",
                    "overhead_duration",
                    "scope",
                    "actor_ir_id",
                }
            case _:
                pytest.fail(f"Unexpected scope: {log['scope']}")
