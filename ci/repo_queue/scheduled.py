#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the explicit-source DM pilot identically from a scheduler or terminal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def command(selection, mode, output_dir):
    numbers = selection.get("numbers")
    if (
        not isinstance(numbers, list)
        or not 1 <= len(numbers) <= 10
        or any(type(number) is not int or number < 1 for number in numbers)
        or len(set(numbers)) != len(numbers)
    ):
        raise ValueError(
            "scheduled numbers must contain 1–10 distinct positive integers"
        )
    if mode not in {"preview", "refresh", "dm", "rehearsal"}:
        raise ValueError("unknown pilot mode")
    if mode == "rehearsal":
        return [
            sys.executable,
            str(ROOT / "rehearse.py"),
            "--output-dir",
            str(output_dir),
            "--send-dm",
        ]
    result = [
        sys.executable,
        str(ROOT / "pilot.py"),
        "--numbers",
        *map(str, numbers),
        "--output-dir",
        str(output_dir),
    ]
    if mode in {"refresh", "dm"}:
        result.append("--apply")
    if mode == "dm":
        result.append("--send-dm")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=ROOT / "scheduled.json"
    )
    parser.add_argument(
        "--mode",
        choices=["preview", "refresh", "dm", "rehearsal"],
        default="preview",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = command(
        json.loads(args.selection.read_text()), args.mode, args.output_dir
    )
    if args.mode in {"dm", "rehearsal"} and not os.environ.get(
        "SLACK_BOT_TOKEN"
    ):
        parser.error("SLACK_BOT_TOKEN is required for unattended DM delivery")
    if os.environ.get("GITHUB_ACTIONS") == "true" and args.mode != "rehearsal":
        if not os.environ.get("GH_TOKEN"):
            parser.error(
                "GH_TOKEN must provide access to the organization Project"
            )
    # Let the child retain its receipt files and diagnostics on failure.
    raise SystemExit(subprocess.run(selected, check=False).returncode)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        sys.exit(str(error))
