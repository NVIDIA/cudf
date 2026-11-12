#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require a memory resource argument in every rmm::exec_policy_nosync call."""

from __future__ import annotations

import sys
from pathlib import Path

CALL = "rmm::exec_policy_nosync"


def _scan_call(text: str, open_paren: int) -> tuple[int, bool]:
    """Return the call end and whether a nonempty second argument was found."""
    paren_depth = 1
    bracket_depth = 0
    brace_depth = 0
    has_top_level_comma = False
    has_second_argument = False
    state = "code"
    index = open_paren + 1

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if state == "line_comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state in {"string", "character"}:
            quote = '"' if state == "string" else "'"
            if char == "\\":
                index += 2
            elif char == quote:
                state = "code"
                index += 1
            else:
                index += 1
            continue

        if char == "/" and next_char == "/":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue
        if char == '"':
            if (
                has_top_level_comma
                and paren_depth == 1
                and bracket_depth == 0
                and brace_depth == 0
            ):
                has_second_argument = True
            state = "string"
            index += 1
            continue
        if char == "'":
            if (
                has_top_level_comma
                and paren_depth == 1
                and bracket_depth == 0
                and brace_depth == 0
            ):
                has_second_argument = True
            state = "character"
            index += 1
            continue

        at_top_level = (
            paren_depth == 1 and bracket_depth == 0 and brace_depth == 0
        )
        if char == "," and at_top_level:
            has_top_level_comma = True
            has_second_argument = False
        elif (
            has_top_level_comma
            and at_top_level
            and not char.isspace()
            and char != ")"
        ):
            has_second_argument = True

        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
            if paren_depth == 0:
                return index, has_top_level_comma and has_second_argument
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        index += 1

    return len(text), False


def find_missing_resources(text: str) -> list[int]:
    """Return offsets of exec_policy_nosync calls without a second argument."""
    failures: list[int] = []
    state = "code"
    index = 0

    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if state == "line_comment":
            if char == "\n":
                state = "code"
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state in {"string", "character"}:
            quote = '"' if state == "string" else "'"
            if char == "\\":
                index += 2
            elif char == quote:
                state = "code"
                index += 1
            else:
                index += 1
            continue

        if char == "/" and next_char == "/":
            state = "line_comment"
            index += 2
            continue
        if char == "/" and next_char == "*":
            state = "block_comment"
            index += 2
            continue
        if char == '"':
            state = "string"
            index += 1
            continue
        if char == "'":
            state = "character"
            index += 1
            continue

        if text.startswith(CALL, index):
            open_paren = index + len(CALL)
            while open_paren < len(text) and text[open_paren].isspace():
                open_paren += 1
            if open_paren < len(text) and text[open_paren] == "(":
                end, has_resource = _scan_call(text, open_paren)
                if not has_resource:
                    failures.append(index)
                index = end + 1
                continue
        index += 1

    return failures


def main(paths: list[str]) -> int:
    """Check each path supplied by pre-commit."""
    failed = False
    for filename in paths:
        path = Path(filename)
        text = path.read_text(encoding="utf-8")
        for offset in find_missing_resources(text):
            line_number = text.count("\n", 0, offset) + 1
            source_line = text.splitlines()[line_number - 1].strip()
            print(
                f"{filename}:{line_number}: rmm::exec_policy_nosync requires "
                f"a memory resource as its second argument\n  {source_line}",
                file=sys.stderr,
            )
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
