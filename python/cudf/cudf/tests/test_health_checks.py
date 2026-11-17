# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from importlib.metadata import entry_points

_CUDF_HEALTH_CHECKS = {
    "cudf_import": "cudf._health_checks:import_check",
    "cudf_functional": "cudf._health_checks:functional_check",
    "cudf_functional_numba": "cudf._health_checks:functional_numba_check",
}


def test_cudf_health_checks_are_registered():
    """Verify cuDF exposes its health checks to RAPIDS Doctor."""
    registered_checks = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="rapids_doctor_check")
        if entry_point.name in _CUDF_HEALTH_CHECKS
    }

    assert registered_checks == _CUDF_HEALTH_CHECKS


def test_rapids_doctor_runs_required_and_cudf_health_checks():
    """Verify RAPIDS Doctor successfully runs all discovered checks."""
    result = subprocess.run(
        ["rapids", "doctor", "--verbose"],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    output = f"{result.stdout}\n{result.stderr}"

    # A successful RAPIDS Doctor exit code means all discovered checks passed.
    assert result.returncode == 0, (
        f"rapids doctor exited with return code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    for name in _CUDF_HEALTH_CHECKS:
        assert f"Found check '{name}'" in output, output
