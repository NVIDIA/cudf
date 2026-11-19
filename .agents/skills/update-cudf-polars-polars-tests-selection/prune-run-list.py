#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "packaging",
# ]
# ///
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Remove GPU run-list entries for Polars versions below the minimum version
# supported by cudf-polars.
#
# Usage:
#    uv run .agents/skills/update-cudf-polars-polars-tests-selection/prune-run-list.py

import json
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "python" / "cudf_polars" / "pyproject.toml"
RUN_LIST = REPO_ROOT / "ci" / "cudf_polars" / "gpu_test_run_list.json"


def minimum_polars_version() -> Version:
    pyproject = tomllib.loads(PYPROJECT.read_text())
    (polars,) = (
        req
        for dep in pyproject["project"]["dependencies"]
        if (req := Requirement(dep)).name == "polars"
    )
    return max(
        Version(spec.version)
        for spec in polars.specifier
        if spec.operator in (">=", "==", "~=")
    )


def main() -> None:
    minimum = minimum_polars_version()
    manifest = json.loads(RUN_LIST.read_text())
    selections = manifest["selections"]
    for key in sorted(selections):
        if Version(key.split("|", 1)[0]) < minimum:
            print(f"Removing {key} (minimum supported Polars is {minimum})")  # noqa: T201
            del selections[key]
    if not selections:
        raise SystemExit("Refusing to write a run list with no selections")
    RUN_LIST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
