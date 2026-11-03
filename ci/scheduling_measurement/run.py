# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary matched measurements; removed before the production matrix."""

import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

tools = Path(__file__).resolve().parent
test_dir = Path(os.environ["CONDA_PREFIX"]) / "bin/gtests/libcudf"
output = Path(os.environ["RAPIDS_ARTIFACTS_DIR"]) / (
    f"libcudf-scheduling-{os.environ.get('GITHUB_RUN_ID', 'local')}-"
    f"{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}-{socket.gethostname()}"
)
output.mkdir(parents=True, exist_ok=False)
with tempfile.TemporaryDirectory(
    prefix="cudf-scheduling-metadata-"
) as temporary:
    directory = Path(temporary)
    metadata = directory / "tests.json"
    metadata.write_text(
        subprocess.check_output(
            ["ctest", "--show-only=json-v1"], cwd=test_dir, text=True
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(tools / "prepare_join_metadata.py"),
            str(metadata),
            str(directory),
        ],
        check=True,
    )
    candidate = directory / "candidate.json"
    candidate.write_text("{}")
    subprocess.run(
        [
            sys.executable,
            str(tools / "ci_compare.py"),
            "--test-dir",
            str(test_dir),
            "--candidate",
            str(candidate),
            "--metadata",
            str(metadata),
            "--baseline-metadata",
            str(directory / "baseline-tests.json"),
            "--candidate-metadata",
            str(directory / "candidate-tests.json"),
            "--output",
            str(output),
            "--regime",
            "warm",
            "--pairs",
            "3",
        ],
        check=True,
    )
