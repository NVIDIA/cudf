# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gather per-worker evidence for the temporary Arm64 Polars CI replay."""

import ctypes
import os
import subprocess
import time
from pathlib import Path

import pytest

_output = Path(os.environ["CUDF_POLARS_DIAGNOSTICS_DIR"])


def _attach_debugger(worker):
    path = _output / f"{worker}.gdb.log"
    libc = ctypes.CDLL(None, use_errno=True)
    # Yama can restrict a debugger spawned by its own tracee. The CI container
    # also grants SYS_PTRACE because attachment uses a separate process.
    pr_set_ptracer = 0x59616D61
    libc.prctl(
        ctypes.c_int(pr_set_ptracer),
        ctypes.c_ulong(-1),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    command_file = Path(__file__).with_name("native.gdb")
    with path.open("wb") as log:
        debugger = subprocess.Popen(
            [
                "gdb",
                "-nx",
                "-batch",
                "-p",
                str(os.getpid()),
                "-x",
                command_file,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                k: v
                for k, v in os.environ.items()
                if k not in {"GH_TOKEN", "GITHUB_TOKEN"}
            },
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if b"CUDF_GDB_ATTACHED" in path.read_bytes():
            return debugger
        if debugger.poll() is not None:
            break
        time.sleep(0.1)
    debugger.terminate()
    pytest.exit(
        f"GDB could not attach to {worker}; see {path.name}", returncode=2
    )


def _trace_partition_and_pack(worker):
    from cudf_polars.streaming.actor_graph.collectives import shuffle

    original = shuffle.py_partition_and_pack
    path = _output / f"{worker}.shuffle.log"

    def partition_and_pack(**kwargs):
        test = os.environ.get("PYTEST_CURRENT_TEST", "")
        started = time.monotonic_ns()
        with path.open("a") as trace:
            trace.write(
                f"{started} begin {test} keys={kwargs['columns_to_hash']} "
                f"partitions={kwargs['num_partitions']}\n"
            )
        result = original(**kwargs)
        with path.open("a") as trace:
            trace.write(f"{started} end\n")
        return result

    shuffle.py_partition_and_pack = partition_and_pack


def pytest_sessionstart(session):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        with (_output / "conda-explicit.txt").open("w") as packages:
            subprocess.run(
                ["conda", "list", "--explicit"], stdout=packages, check=True
            )
    elif os.environ["CUDF_POLARS_DIAGNOSTICS_TRIAL"].startswith("native-"):
        session._cudf_debugger = _attach_debugger(worker)
        _trace_partition_and_pack(worker)


def pytest_runtest_logstart(nodeid):
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        with (_output / f"{worker}.tests.log").open("a") as log:
            log.write(f"{time.monotonic_ns()} {nodeid}\n")
