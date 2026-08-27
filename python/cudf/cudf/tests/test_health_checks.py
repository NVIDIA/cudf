# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess

_CUDF_HEALTH_CHECKS = {
    "cudf_import": "cudf._health_checks:import_check",
    "cudf_functional": "cudf._health_checks:functional_check",
    "cudf_functional_numba": "cudf._health_checks:functional_numba_check",
}


def test_rapids_doctor_runs_required_and_cudf_health_checks():
    """Verify RAPIDS Doctor runs required checks and discovers/runs cuDF checks."""
    result = subprocess.run(
        ["rapids", "doctor", "--verbose"],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    output = f"{result.stdout}\n{result.stderr}"

    # A successful RAPIDS Doctor exit code means all discovered checks passed.
    assert result.returncode == 0, output
    for name, target in _CUDF_HEALTH_CHECKS.items():
        assert f"Found check '{name}' provided by '{target}'" in output
