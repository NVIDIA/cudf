#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Separate PyPI requirements for installation with uv."""

import sys
from pathlib import Path

import yaml


def split_environment(source, conda_output, requirements_output):
    environment = yaml.safe_load(Path(source).read_text())
    dependencies = []
    requirements = []
    for dependency in environment.get("dependencies", []):
        if isinstance(dependency, dict) and "pip" in dependency:
            if dependency.keys() != {"pip"}:
                raise ValueError(
                    f"Unexpected dependency installer: {dependency}"
                )
            requirements.extend(dependency["pip"])
        else:
            dependencies.append(dependency)
    environment["dependencies"] = dependencies
    Path(conda_output).write_text(yaml.safe_dump(environment, sort_keys=False))
    Path(requirements_output).write_text(
        "".join(f"{item}\n" for item in requirements)
    )


if __name__ == "__main__":
    split_environment(*sys.argv[1:])
