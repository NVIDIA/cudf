#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep reusable allocating test APIs on cudf::memory_resources."""

from __future__ import annotations

import re
import sys
from pathlib import Path

LEGACY_RESOURCE = re.compile(r"\brmm::device_async_resource_ref\b")
CURRENT_RESOURCE = "cudf::get_current_device_resource_ref()"
DEFAULT_ARGUMENT = re.compile(r"cudf::memory_resources\s+\w+\s*=\s*$")
DELEGATING_DEFAULT = re.compile(r"\(\)\s*:\s*\w+\s*\(\s*$")


def _mask_non_code(text: str) -> str:
    """Mask comments and literals while preserving offsets and newlines."""
    chars = list(text)
    state = "code"
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                chars[index] = chars[index + 1] = " "
                state = "code"
                index += 2
            else:
                if char != "\n":
                    chars[index] = " "
                index += 1
            continue
        if state in {"string", "character"}:
            quote = '"' if state == "string" else "'"
            if char == "\\":
                chars[index] = " "
                if index + 1 < len(text):
                    if text[index + 1] != "\n":
                        chars[index + 1] = " "
                    index += 2
                else:
                    index += 1
            else:
                if char != "\n":
                    chars[index] = " "
                if char == quote:
                    state = "code"
                index += 1
            continue

        if char == "/" and next_char == "/":
            chars[index] = chars[index + 1] = " "
            state = "line_comment"
            index += 2
        elif char == "/" and next_char == "*":
            chars[index] = chars[index + 1] = " "
            state = "block_comment"
            index += 2
        elif char == '"':
            chars[index] = " "
            state = "string"
            index += 1
        elif char == "'":
            chars[index] = " "
            state = "character"
            index += 1
        else:
            index += 1
    return "".join(chars)


def find_violations(text: str) -> list[tuple[int, str]]:
    """Return offsets and explanations for disallowed resource usage."""
    code = _mask_non_code(text)
    violations = [
        (
            match.start(),
            "use cudf::memory_resources instead of a resource-ref API",
        )
        for match in LEGACY_RESOURCE.finditer(code)
    ]

    offset = code.find(CURRENT_RESOURCE)
    while offset != -1:
        prefix = code[max(0, offset - 256) : offset]
        is_default_argument = DEFAULT_ARGUMENT.search(prefix) is not None
        is_delegating_default = DELEGATING_DEFAULT.search(prefix) is not None
        if not (is_default_argument or is_delegating_default):
            violations.append(
                (
                    offset,
                    "use a memory_resources accessor instead of the current resource",
                )
            )
        offset = code.find(CURRENT_RESOURCE, offset + len(CURRENT_RESOURCE))

    return sorted(violations)


def main(paths: list[str]) -> int:
    """Check each reusable test API path supplied by pre-commit."""
    failed = False
    for filename in paths:
        path = Path(filename)
        text = path.read_text(encoding="utf-8")
        for offset, explanation in find_violations(text):
            line_number = text.count("\n", 0, offset) + 1
            source_line = text.splitlines()[line_number - 1].strip()
            print(
                f"{filename}:{line_number}: {explanation}\n  {source_line}",
                file=sys.stderr,
            )
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
