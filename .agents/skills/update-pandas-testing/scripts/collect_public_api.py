# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collect the public pandas API documented in the pandas API reference.

Every name listed in an ``autosummary`` directive, or declared by a Python
object directive such as ``.. type::``, in ``doc/source/reference/*.rst`` is
joined with the active ``currentmodule`` and printed as a fully qualified
name, one per line, sorted.

Usage:
    python collect_public_api.py <path to pandas repository> > public_api.txt
"""

import argparse
import re
from collections.abc import Iterator
from pathlib import Path

CURRENTMODULE = re.compile(r"^\s*\.\. currentmodule:: (?P<module>\S+)\s*$")
AUTOSUMMARY = re.compile(r"^(?P<indent>\s*)\.\. autosummary::\s*$")
ENTRY = re.compile(r"^~?(?P<name>[\w.]+)$")
OBJECT_DIRECTIVE = re.compile(
    r"\.\. (?:py:)?(?:attribute|class|data|exception|function|method|type)::"
    r"\s+(?P<name>[\w.]+)"
)


def indentation(line: str) -> int:
    return len(line) - len(line.lstrip())


def iter_public_names(rst_file: Path) -> Iterator[str]:
    module = "pandas"
    block_indent = None
    for line in rst_file.read_text().splitlines():
        if block_indent is not None:
            stripped = line.strip()
            if not stripped:
                continue
            if indentation(line) > block_indent:
                if stripped.startswith(":"):
                    continue
                if (match := ENTRY.match(stripped)) is not None:
                    yield f"{module}.{match['name']}"
                continue
            block_indent = None
        if (match := CURRENTMODULE.match(line)) is not None:
            module = match["module"]
        elif (match := AUTOSUMMARY.match(line)) is not None:
            block_indent = len(match["indent"])
        elif (match := OBJECT_DIRECTIVE.search(line)) is not None:
            yield f"{module}.{match['name']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "pandas_repo", type=Path, help="Path to a pandas repository"
    )
    args = parser.parse_args()
    reference_dir = args.pandas_repo / "doc" / "source" / "reference"
    rst_files = sorted(reference_dir.glob("*.rst"))
    if not rst_files:
        parser.error(f"No .rst files found in {reference_dir}")
    names = sorted(
        {name for rst in rst_files for name in iter_public_names(rst)}
    )
    print("\n".join(names))  # noqa: T201


if __name__ == "__main__":
    main()
