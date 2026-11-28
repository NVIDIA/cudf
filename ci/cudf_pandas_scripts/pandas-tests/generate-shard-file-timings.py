# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Sum pytest report-log durations by pandas test file."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    timings = defaultdict(float)
    for path in args.reports:
        exit_status = None
        with path.open() as report:
            for line in report:
                event = json.loads(line)
                report_type = event.get("$report_type")
                if report_type == "TestReport":
                    file = event["nodeid"].split("::", 1)[0]
                    timings[file] += event["duration"]
                elif report_type == "SessionFinish":
                    exit_status = event["exitstatus"]
        if exit_status != 0:
            parser.error(f"{path} must contain a successful pytest session")

    if not timings or not all(
        math.isfinite(seconds) and seconds >= 0 for seconds in timings.values()
    ):
        parser.error("Expected finite, nonnegative durations for test files")
    if not any(timings.values()):
        parser.error("Expected at least one positive duration")

    args.output.write_text(
        json.dumps(timings, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
