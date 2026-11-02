# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise CI installer compatibility without building CUDA code."""

import json
import os
import runpy
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
split_environment = runpy.run_path(str(ROOT / "ci/utils/split_conda_pip.py"))[
    "split_environment"
]


def test_split_environment(tmp_path):
    original = {
        "channels": ["local", "conda-forge"],
        "dependencies": [
            "python=3.12",
            {
                "pip": [
                    "package>=1",
                    "--extra-index-url https://example.org/simple",
                ]
            },
            "numpy<3",
            {"pip": ["other @ file:///tmp/other.whl"]},
        ],
    }
    source = tmp_path / "input.yaml"
    conda = tmp_path / "conda.yaml"
    requirements = tmp_path / "requirements.txt"
    source.write_text(yaml.safe_dump(original))
    split_environment(source, conda, requirements)
    assert yaml.safe_load(source.read_text()) == original
    assert yaml.safe_load(conda.read_text()) == {
        "channels": original["channels"],
        "dependencies": ["python=3.12", "numpy<3"],
    }
    assert requirements.read_text().splitlines() == [
        "package>=1",
        "--extra-index-url https://example.org/simple",
        "other @ file:///tmp/other.whl",
    ]


@pytest.fixture
def uv_environment(tmp_path):
    uv = ROOT / ".cache/uv-bin/uv"
    if not uv.exists():
        pytest.skip(
            "Source ci/install_uv.sh before running uv integration checks"
        )
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
    uv_bin = tmp_path / ".cache/uv-bin"
    uv_bin.mkdir(parents=True)
    (uv_bin / "uv").symlink_to(uv)
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


def test_conda_pypi_install_targets_new_environment(tmp_path, uv_environment):
    # A tiny wheel lets the real uv installer run entirely offline.
    wheel = tmp_path / "cudf_uv_probe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cudf_uv_probe.py", "VALUE = 42\n")
        archive.writestr(
            "cudf_uv_probe-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: cudf-uv-probe\nVersion: 1.0\n",
        )
        archive.writestr(
            "cudf_uv_probe-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr("cudf_uv_probe-1.0.dist-info/RECORD", "")
    source = tmp_path / "env.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "dependencies": [
                    "python",
                    {"pip": [f"cudf-uv-probe @ {wheel.as_uri()}"]},
                ]
            }
        )
    )
    prefix = tmp_path / "target environment"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mamba = bin_dir / "rapids-mamba-retry"
    mamba.write_text(
        f"#!{sys.executable}\n"
        + """import subprocess, sys, yaml
from pathlib import Path
args = sys.argv[1:]
assert args[:3] == ['env', 'create', '--yes']
assert yaml.safe_load(Path(args[4]).read_text())['dependencies'] == ['python']
assert args[5] == '--prefix'
subprocess.run([sys.argv[0].replace('/bin/rapids-mamba-retry', '/.cache/uv-bin/uv'), 'venv', '--python', sys.executable, args[6]], check=True)
"""
    )
    conda = bin_dir / "conda"
    conda.write_text(
        '#!/bin/bash\n[[ $1 == run && $2 == --prefix && $4 == python ]] || exit 1\nprefix=$3; shift 4\nexec "${prefix}/bin/python" "$@"\n'
    )
    mamba.chmod(0o755)
    conda.chmod(0o755)
    env = uv_environment | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}
    python = bin_dir / "python"
    python.write_text(
        f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    python.chmod(0o755)
    subprocess.run(
        [
            "bash",
            "ci/create_conda_env.sh",
            str(source),
            "--prefix",
            str(prefix),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            str(prefix / "bin/python"),
            "-c",
            "import cudf_uv_probe; assert cudf_uv_probe.VALUE == 42",
        ],
        check=True,
    )
    assert yaml.safe_load(source.read_text())["dependencies"][1]["pip"]
