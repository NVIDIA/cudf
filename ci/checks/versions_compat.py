#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Check that cudf_polars's polars version-compat flags are not stale.

``cudf_polars/utils/versions.py`` defines ``POLARS_VERSION_LT_*`` flags
used to branch on the installed polars version. Once the minimum polars
version pinned in ``cudf_polars/pyproject.toml`` reaches one of these
thresholds, the flag can never be true again, so it (and the code path
it guards) is dead and should be removed.

See https://github.com/rapidsai/cudf/issues/16736.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
import typing
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

# Any assignment to a POLARS_VERSION_LT_* name, regardless of its form.
# Used to make sure FLAG_PATTERN below didn't silently miss one: a flag
# this checker fails to recognize is worse than one it correctly flags,
# since it would make the check report "clean" when it isn't.
ASSIGNMENT_PATTERN = re.compile(
    r"^[ \t]*(?P<name>POLARS_VERSION_LT_\w+)(?:\s*:\s*[^=\n]+)?\s*=",
    re.MULTILINE,
)

# The specific form we know how to extract a version threshold from.
FLAG_PATTERN = re.compile(
    r"^[ \t]*(?P<name>POLARS_VERSION_LT_\w+)(?:\s*:\s*[^=\n]+)?\s*="
    r"\s*POLARS_VERSION\s*<\s*"
    r"parse\(\s*[\"'](?P<version>[0-9]+(?:\.[0-9]+){1,2})[\"']\s*\)",
    re.MULTILINE,
)


class StaleFlag(typing.NamedTuple):
    name: str
    version: str
    lineno: int


class UnparseableFlagError(ValueError):
    """A POLARS_VERSION_LT_* assignment didn't match the expected form."""


def minimum_polars_version(pyproject: Path) -> Version:
    """Return the minimum polars version pinned in ``pyproject``."""
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    for dep in data["project"]["dependencies"]:
        req = Requirement(dep)
        if req.name != "polars":
            continue
        specs = {spec.operator: spec.version for spec in req.specifier}
        for operator in (">=", "==", "~="):
            if operator in specs:
                return Version(specs[operator])
        if ">" in specs:
            # A strict > floor uses its bound intentionally to avoid over-flagging.
            return Version(specs[">"])
        raise ValueError(
            f"'polars' dependency in {pyproject} has no '>=', '==', "
            f"'~=', or '>' specifier to establish a minimum version: "
            f"{dep!r}"
        )
    raise ValueError(f"No 'polars' dependency found in {pyproject}")


def find_stale_flags(versions_py: Path, minimum: Version) -> list[StaleFlag]:
    """Find ``POLARS_VERSION_LT_*`` flags whose threshold has passed.

    Raises ``UnparseableFlagError`` if a ``POLARS_VERSION_LT_*``
    assignment is found that doesn't match the expected form, rather
    than silently skipping it.
    """
    content = versions_py.read_text()

    matches = {m.group("name"): m for m in FLAG_PATTERN.finditer(content)}
    all_names = {m.group("name") for m in ASSIGNMENT_PATTERN.finditer(content)}
    unparseable = sorted(all_names - matches.keys())
    if unparseable:
        raise UnparseableFlagError(
            f"{versions_py} defines {unparseable} in a form this "
            "checker does not recognize (expected `NAME = "
            'POLARS_VERSION < parse("X.Y.Z")`). Update the flag to '
            "that form, or teach FLAG_PATTERN in this script to "
            "recognize it."
        )

    stale = []
    for match in matches.values():
        version = Version(match.group("version"))
        if version <= minimum:
            lineno = content.count("\n", 0, match.start()) + 1
            stale.append(
                StaleFlag(match.group("name"), match.group("version"), lineno)
            )
    stale.sort(key=lambda flag: flag.lineno)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Check that cudf_polars polars version-compat flags are "
            "not stale relative to the pinned minimum polars version."
        )
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("python/cudf_polars/pyproject.toml"),
        help="Path to cudf_polars' pyproject.toml",
    )
    parser.add_argument(
        "--versions-file",
        type=Path,
        default=Path("python/cudf_polars/cudf_polars/utils/versions.py"),
        help="Path to cudf_polars' utils/versions.py",
    )
    args = parser.parse_args(argv)

    try:
        minimum = minimum_polars_version(args.pyproject)
        stale = find_stale_flags(args.versions_file, minimum)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if stale:
        print(
            f"Stale polars version-compat flag(s) in "
            f"{args.versions_file} (minimum supported polars version "
            f"is {minimum}):",
            end="\n\n",
        )
        for flag in stale:
            print(
                f"  {args.versions_file}:{flag.lineno}: {flag.name} "
                f"guards polars < {flag.version}, which can never be "
                f"true given the pinned minimum of {minimum}. Remove "
                f"this flag and the code path(s) it guards."
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
