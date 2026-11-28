# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption("--shard-coverage-output", type=Path, required=True)


def pytest_collection_finish(session):
    coverage = [
        (
            item.nodeid,
            [
                (mark.name, repr(mark.args), repr(mark.kwargs))
                for mark in item.iter_markers()
                if mark.name in {"skip", "skipif", "xfail"}
            ],
        )
        for item in session.items
    ]
    output = session.config.getoption("shard_coverage_output")
    output.write_text(json.dumps(coverage) + "\n")


def load_layout(paths):
    coverage = {}
    for path in paths:
        for nodeid, marks in json.loads(path.read_text()):
            if nodeid in coverage:
                raise ValueError(f"{path}: duplicate node ID {nodeid}")
            coverage[nodeid] = marks
    if not coverage:
        raise ValueError("No tests collected")
    return coverage


def main():
    parser = argparse.ArgumentParser(
        description="Compare node IDs and skip/xfail markers across two layouts."
    )
    parser.add_argument("--node", nargs="+", required=True, type=Path)
    parser.add_argument("--hybrid", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    try:
        node = load_layout(args.node)
        hybrid = load_layout(args.hybrid)
    except ValueError as error:
        parser.error(str(error))

    differences = {
        "Missing from hybrid": node.keys() - hybrid.keys(),
        "Only in hybrid": hybrid.keys() - node.keys(),
        "Different markers": {
            nodeid
            for nodeid in node.keys() & hybrid.keys()
            if node[nodeid] != hybrid[nodeid]
        },
    }
    for label, nodeids in differences.items():
        if nodeids:
            sys.stderr.write(
                f"{label} ({len(nodeids)}): {', '.join(sorted(nodeids)[:10])}\n"
            )
    if any(differences.values()):
        return 1

    sys.stdout.write(
        f"Collection and skip/xfail markers match for {len(node):,} tests.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
