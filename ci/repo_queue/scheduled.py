#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run work tracking identically from a scheduler or terminal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from progress import read_stage, record_stage
from selection import validate_numbers
from slack_dm_test import slack_token

ROOT = Path(__file__).parent


def command(selection, mode, output_dir):
    if not isinstance(selection, dict):
        raise ValueError("scheduled selection must be an object")
    numbers = selection.get("numbers")
    validate_numbers(numbers)
    if mode not in {"preview", "refresh", "dm", "rehearsal"}:
        raise ValueError("unknown tracking mode")
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
        str(ROOT / "track.py"),
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
    try:
        selected = command(
            json.loads(args.selection.read_text()), args.mode, args.output_dir
        )
    except ValueError:
        parser.error("Invalid tracking selection configuration")
    if args.mode in {"dm", "rehearsal"}:
        try:
            slack_token(allow_prompt=False)
        except RuntimeError as error:
            parser.error(str(error))
    if os.environ.get("GITHUB_ACTIONS") == "true" and args.mode != "rehearsal":
        if not os.environ.get("GH_TOKEN"):
            parser.error(
                "GH_TOKEN must provide access to the organization Project"
            )
    # Private Project metadata and DM receipts must not become public Actions logs.
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.output_dir.chmod(0o700)
    record_stage(args.output_dir, "startup")
    with (args.output_dir / "diagnostics.log").open("a") as diagnostics:
        (args.output_dir / "diagnostics.log").chmod(0o600)
        result = subprocess.run(
            selected, stdout=diagnostics, stderr=diagnostics, check=False
        )
    print(
        "Work tracking completed."
        if result.returncode == 0
        else f"Work tracking failed at {read_stage(args.output_dir)} (exit {result.returncode})."
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    try:
        main()
    except OSError:
        sys.exit(
            "Work tracking startup failed: local I/O or subprocess launch."
        )
    except ValueError:
        sys.exit("Work tracking startup failed: invalid configuration.")
