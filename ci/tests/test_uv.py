# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise CI installer compatibility without building CUDA code."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def uv_environment(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("Install uv>=0.12.0 before running uv integration checks")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PIP_", "UV_", "NV_GHA_RUNNERS_"))
    }
    env.update(
        GITHUB_WORKSPACE=str(tmp_path),
        UV_NO_INDEX="1",
        UV_CACHE_DIR=str(tmp_path / "cache"),
    )
    return env


@pytest.mark.parametrize("proxy", [False, True])
def test_indexes_and_constraints(tmp_path, uv_environment, proxy):
    env = uv_environment
    if proxy:
        env.update(
            PIP_INDEX_URL="https://pypi-proxy.example/simple",
            NV_GHA_RUNNERS_PYPI_ANACONDA_CACHE_URL="https://anaconda-proxy.example",
        )
    result = subprocess.run(
        [
            "bash",
            "-euc",
            'source ci/init_uv.sh; source ci/init_uv.sh; exec "$1" -c "import json,os; print(json.dumps(dict(os.environ)))"',
            "bash",
            sys.executable,
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    configured = json.loads(result.stdout)
    assert configured["UV_INDEX_URL"] == (
        "https://pypi-proxy.example/simple"
        if proxy
        else "https://pypi.org/simple"
    )
    assert (
        configured["UV_EXTRA_INDEX_URL"]
        == (
            "https://anaconda-proxy.example"
            if proxy
            else "https://pypi.anaconda.org"
        )
        + "/rapidsai-wheels-nightly/simple"
    )
    assert configured["UV_INDEX_STRATEGY"] == "unsafe-best-match"
    assert configured["UV_CONSTRAINT"] == configured["PIP_CONSTRAINT"]
    assert Path(configured["UV_CONSTRAINT"]).exists()
    assert not (tmp_path / ".cache/uv-bin").exists()
