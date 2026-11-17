# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from shutil import which

import pytest

GDB_COMMANDS = """
set confirm off
set breakpoint pending on
break cuInit
run
exit
"""


@pytest.fixture(scope="module")
def cuda_gdb(request):
    gdb = which("cuda-gdb")
    if gdb is None:
        request.applymarker(
            pytest.mark.xfail(reason="No cuda-gdb found, can't detect cuInit"),
        )
        return gdb
    else:
        output = subprocess.run(
            [gdb, "--version"], capture_output=True, text=True, cwd="/"
        )
        if output.returncode != 0:
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "cuda-gdb not working on this platform, "
                        f"can't detect cuInit: {output.stderr}"
                    )
                ),
            )
        return gdb


@pytest.mark.parametrize(
    "cudf_call, should_be_initialized",
    [
        ("import cudf", False),
        ("import cudf; cudf.Series([1])", True),
    ],
)
def test_rapids_no_initialize_cuinit(
    cuda_gdb, monkeypatch, cudf_call, should_be_initialized
):
    # When RAPIDS_NO_INITIALIZE is set, importing cudf should _not_
    # create a CUDA context (i.e. cuInit should not be called).
    # Intercepting the call to cuInit programmatically is tricky since
    # the way it is resolved from dynamic libraries by
    # cuda-python/numba/cupy is multitudinous (see discussion at
    # https://github.com/NVIDIA/cudf/pull/12361 which does this, but
    # needs provide hooks that override dlsym, cuGetProcAddress, and
    # cuInit.
    # Instead, we just run under GDB and see if we hit a breakpoint
    with monkeypatch.context() as m:
        m.setenv("RAPIDS_NO_INITIALIZE", "1")
        output = subprocess.run(
            [
                cuda_gdb,
                "-x",
                "-",
                "--args",
                sys.executable,
                "-c",
                cudf_call,
            ],
            input=GDB_COMMANDS,
            capture_output=True,
            text=True,
            cwd="/",
        )

    print("Command output:\n")  # noqa: T201
    print("*** STDOUT ***")  # noqa: T201
    print(output.stdout)  # noqa: T201
    print("*** STDERR ***")  # noqa: T201
    print(output.stderr)  # noqa: T201
    assert output.returncode == 0
    assert ("in cuInit ()" in output.stdout) == should_be_initialized
