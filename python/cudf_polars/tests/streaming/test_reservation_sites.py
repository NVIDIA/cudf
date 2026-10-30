# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep production memory-reservation telemetry sites attributed."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_RESERVATION_CALLS = {
    "AllGather",
    "AllGatherManager",
    "ShuffleManager",
    "bloom.apply",
    "bloom.build",
    "cardinality_estimator.estimate",
    "make_available_and_spill",
    "make_available_or_wait",
    "make_table_chunks_available_or_wait",
    "partition_and_pack",
    "py_partition_and_pack",
    "py_split_and_pack",
    "py_unpack_and_concat",
    "reserve",
    "reserve_device_memory_and_spill",
    "reserve_memory",
    "reserve_or_fail",
    "reserve_or_wait",
    "split_and_pack",
    "unpack_and_concat",
}

# Deliberately empty. Entries must be (package-relative path, line, call name)
# and carry a comment explaining why actor attribution is impossible. Site-only
# internal operations still pass attributes and do not belong here.
_UNATTRIBUTED_ALLOWLIST: set[tuple[str, int, str]] = set()
_CYTHON_CALL = re.compile(
    rf"\b({'|'.join(map(re.escape, sorted(_RESERVATION_CALLS, key=len, reverse=True)))})\s*\("
)


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        if (
            call.func.attr in {"apply", "build"}
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "bloom"
        ):
            return f"bloom.{call.func.attr}"
        if (
            call.func.attr == "estimate"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "cardinality_estimator"
        ):
            return "cardinality_estimator.estimate"
        return call.func.attr
    return None


def _cython_calls(path: Path) -> list[tuple[int, str, str]]:
    """Return reservation calls from Cython source with their argument text."""
    source = path.read_text()
    calls = []
    for match in _CYTHON_CALL.finditer(source):
        if (
            source.count('"""', 0, match.start()) % 2
            or source.count("'''", 0, match.start()) % 2
        ):
            continue
        line_start = source.rfind("\n", 0, match.start()) + 1
        prefix = source[line_start : match.start()].strip()
        first_token = prefix.split(maxsplit=1)[0] if prefix else ""
        if first_token in {"def", "async", "cdef", "cpdef"} or prefix.startswith(
            ("#", ">>>")
        ):
            continue
        depth = 1
        cursor = match.end()
        while cursor < len(source) and depth:
            depth += (source[cursor] == "(") - (source[cursor] == ")")
            cursor += 1
        calls.append(
            (
                source.count("\n", 0, match.start()) + 1,
                match.group(1),
                source[match.end() : cursor - 1],
            )
        )
    return calls


def test_production_reservation_calls_pass_attributes() -> None:
    """Require explicit attributes at every production reservation boundary."""
    package_root = Path(__file__).parents[2] / "cudf_polars"
    python_root = package_root.parents[1]
    missing: set[tuple[str, int, str]] = set()
    discovered_allowlist_entries: set[tuple[str, int, str]] = set()

    for path in package_root.rglob("*.py"):
        relative_path = path.relative_to(package_root).as_posix()
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in _RESERVATION_CALLS:
                continue
            entry = (relative_path, node.lineno, name)
            if any(keyword.arg == "attributes" for keyword in node.keywords):
                continue
            if entry in _UNATTRIBUTED_ALLOWLIST:
                discovered_allowlist_entries.add(entry)
            else:
                missing.add(entry)

    for path in (python_root / "cudf_streaming" / "cudf_streaming").rglob("*.pyx"):
        relative_path = path.relative_to(python_root).as_posix()
        for line, name, arguments in _cython_calls(path):
            entry = (relative_path, line, name)
            if "attributes=" in arguments:
                continue
            if entry in _UNATTRIBUTED_ALLOWLIST:
                discovered_allowlist_entries.add(entry)
            else:
                missing.add(entry)

    assert not missing, f"reservation calls missing attributes=: {sorted(missing)}"
    assert discovered_allowlist_entries == _UNATTRIBUTED_ALLOWLIST, (
        "remove stale reservation attribution allowlist entries: "
        f"{sorted(_UNATTRIBUTED_ALLOWLIST - discovered_allowlist_entries)}"
    )
