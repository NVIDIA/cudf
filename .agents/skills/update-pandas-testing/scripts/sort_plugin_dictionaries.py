# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sort the node id dictionaries in pandas-testing-plugin.py by key, in place.

Each entry must start on its own line. Comment lines directly above an entry
move with that entry. Comment lines after the last entry stay at the end of
the dictionary.

Usage:
    python sort_plugin_dictionaries.py <path to pandas-testing-plugin.py>
    python sort_plugin_dictionaries.py --check <path to pandas-testing-plugin.py>
"""

import argparse
import ast
import sys
from collections.abc import Sequence
from pathlib import Path

DICTIONARIES = (
    "NODEIDS_PATHS_TO_SKIP",
    "NODEIDS_THAT_FAIL",
    "NODEIDS_THAT_MAY_FAIL",
    "NODEIDS_TO_SKIP",
    "NODEIDS_TO_SKIP_WHEN_SHARDED",
)


def find_dictionaries(tree: ast.Module) -> dict[str, ast.Dict]:
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if (
            isinstance(target, ast.Name)
            and target.id in DICTIONARIES
            and isinstance(node.value, ast.Dict)
        ):
            found[target.id] = node.value
    if missing := sorted(set(DICTIONARIES) - found.keys()):
        raise ValueError(f"Dictionaries not found: {', '.join(missing)}")
    return found


def sorted_body(
    name: str, node: ast.Dict, lines: Sequence[str]
) -> tuple[str, ...]:
    """Return the lines between ``{`` and ``}`` with the entries sorted."""
    if not lines[node.lineno - 1].rstrip().endswith("{") or (
        lines[node.end_lineno - 1].strip() != "}"
    ):
        raise ValueError(f"{name}: braces must end and start their own lines")
    entries = []
    # Line indexes are 0-based; ast line numbers are 1-based.
    start = node.lineno
    for key, value in zip(node.keys, node.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            raise ValueError(
                f"{name}: line {value.lineno} has a non-string key"
            )
        if key.lineno - 1 < start:
            raise ValueError(
                f"{name}: line {key.lineno} does not start on its own line"
            )
        chunk = list(lines[start : value.end_lineno])
        if not chunk[-1].rstrip().endswith(","):
            if "#" in chunk[-1]:
                raise ValueError(
                    f"{name}: line {value.end_lineno} needs a trailing comma"
                )
            chunk[-1] = f"{chunk[-1].rstrip()},"
        entries.append((key.value, tuple(chunk)))
        start = value.end_lineno
    keys = [key for key, _ in entries]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"{name}: duplicate keys {duplicates}")
    trailing = tuple(lines[start : node.end_lineno - 1])
    return (
        *(line for _, chunk in sorted(entries) for line in chunk),
        *trailing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "plugin", type=Path, help="Path to pandas-testing-plugin.py"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not modify the file; exit with status 1 if it is not sorted",
    )
    args = parser.parse_args()
    source = args.plugin.read_text()
    lines = source.splitlines()
    try:
        dictionaries = find_dictionaries(ast.parse(source))
        # Replace from the bottom so earlier line numbers stay valid.
        for name, node in sorted(
            dictionaries.items(), key=lambda item: -item[1].lineno
        ):
            lines[node.lineno : node.end_lineno - 1] = sorted_body(
                name, node, lines
            )
    except ValueError as error:
        sys.exit(f"{args.plugin}: {error}")
    result = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    if result == source:
        return
    if args.check:
        sys.exit(f"{args.plugin}: dictionaries are not sorted")
    args.plugin.write_text(result)
    print(f"Sorted the dictionaries in {args.plugin}")  # noqa: T201


if __name__ == "__main__":
    main()
