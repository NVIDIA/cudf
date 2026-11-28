# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture the fault in the worker process before its nanny reports the exit."""

import faulthandler
import os
from pathlib import Path

from repro_diagnostics import _install_fault_report


def dask_setup(worker):
    name = f"dask-{os.getpid()}"
    path = (
        Path(os.environ["CUDF_POLARS_DIAGNOSTICS_DIR"])
        / f"{name}.python-fault.log"
    )
    python_log = path.open("x", buffering=1)
    python_log.write(f"pid={os.getpid()} worker={worker.address}\n")
    # The native reporter chains to the Python handler that was installed first.
    faulthandler.enable(file=python_log)
    worker._cudf_fault_capture = (python_log, _install_fault_report(name))
