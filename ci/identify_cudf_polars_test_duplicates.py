#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a review queue of local tests plausibly replaced by upstream tests.

Coverage is used only to find pairs.  The output deliberately records exact
pytest contexts and source-line evidence so reviewers can compare the public
assertions before deleting a local test.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from coverage.numbits import numbits_to_nums

SOURCE_MARKER = "/cudf_polars/"
EXCLUDED_LOCAL_PATHS = (
    "tests/containers/",
    "tests/dsl/",
    "tests/io/",
    "tests/streaming/",
    "tests/testing/",
    "tests/utils/",
)
EXCLUDED_TERMS = ("dtype", "decimal", "fallback", "unsupported", "scan")
MAX_UPSTREAM_CONTEXTS_PER_LINE = 100
GENERIC_NAME_TOKENS = frozenset({"test", "expr", "lazy", "frame", "polars"})


def _nodeid(context: str) -> str:
    return context.rsplit("|", 1)[0]


def _contexts(data: Path, prefix: str) -> dict[str, set[tuple[str, int]]]:
    connection = sqlite3.connect(data)
    rows = connection.execute(
        """
        SELECT file.path, context.context, line_bits.numbits
        FROM line_bits
        JOIN file ON file.id = line_bits.file_id
        JOIN context ON context.id = line_bits.context_id
        WHERE file.path LIKE ? AND context.context LIKE ?
        """,
        (f"%{SOURCE_MARKER}%", f"{prefix}%"),
    )
    result: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for filename, context, numbits in rows:
        result[_nodeid(context)].update(
            (filename, line) for line in numbits_to_nums(numbits)
        )
    connection.close()
    return result


def _eligible(nodeid: str) -> bool:
    return not nodeid.startswith(EXCLUDED_LOCAL_PATHS) and not any(
        term in nodeid.lower() for term in EXCLUDED_TERMS
    )


def _behavior_tokens(nodeid: str) -> set[str]:
    """Return meaningful public-behavior tokens from a pytest node ID."""
    return {
        token
        for token in re.split(r"[^a-z0-9]+", nodeid.lower())
        if len(token) > 2 and token not in GENERIC_NAME_TOKENS
    }


def build_queue(
    local: Path, upstream: Path, minimum_shared: int
) -> list[dict]:
    local_tests = _contexts(local, "tests/")
    upstream_tests = _contexts(upstream, "tests/")
    upstream_by_line: dict[tuple[str, int], set[str]] = defaultdict(set)
    for nodeid, lines in upstream_tests.items():
        for line in lines:
            upstream_by_line[line].add(nodeid)

    queue = []
    for nodeid, lines in local_tests.items():
        if not _eligible(nodeid) or not lines:
            continue
        matches = Counter(
            upstream_node
            for line in lines
            if len(upstream_by_line.get(line, ()))
            <= MAX_UPSTREAM_CONTEXTS_PER_LINE
            for upstream_node in upstream_by_line.get(line, ())
        )
        if not matches:
            continue
        local_tokens = _behavior_tokens(nodeid)
        matching_names = [
            (upstream_node, count)
            for upstream_node, count in matches.items()
            if local_tokens & _behavior_tokens(upstream_node)
        ]
        if not matching_names:
            continue
        upstream_node, shared = max(matching_names, key=lambda entry: entry[1])
        if shared < minimum_shared:
            continue
        queue.append(
            {
                "local_nodeid": nodeid,
                "upstream_nodeid": upstream_node,
                "shared_line_count": shared,
                "local_line_count": len(lines),
                "shared_fraction": round(shared / len(lines), 3),
                "other_upstream_matches": [
                    {"nodeid": match, "shared_line_count": count}
                    for match, count in matches.most_common(5)[1:]
                ],
            }
        )
    return sorted(
        queue,
        key=lambda row: (
            -row["shared_fraction"],
            -row["shared_line_count"],
            row["local_nodeid"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, required=True)
    parser.add_argument("--upstream-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-shared", type=int, default=4)
    args = parser.parse_args()
    queue = build_queue(
        args.local_data, args.upstream_data, args.minimum_shared
    )
    args.output.write_text(json.dumps({"candidates": queue}, indent=2) + "\n")
    print(f"Wrote {len(queue)} semantic-review candidates to {args.output}")


if __name__ == "__main__":
    main()
