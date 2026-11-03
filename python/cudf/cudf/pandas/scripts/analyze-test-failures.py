# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prints the most common test failures for the given tests.

Usage:
    python analyze-test-failures.py <path-to-test-log> <file-or-pattern>

Example:
-------
    python analyze-test-failures.py log.json frame/*
"""

import json
import sys
from collections import Counter
from fnmatch import fnmatch

from rich.console import Console
from rich.table import Table

PANDAS_TEST_PREFIX = "pandas-tests/"


def count_failures(log_file_name, pattern):
    counter = Counter()
    with open(log_file_name) as f:
        for line in f:
            try:
                line = json.loads(line)
            except Exception:
                continue
            if (
                "location" in line
                and line["when"] == "call"
                and line["outcome"] == "failed"
            ):
                line_module_name = line["location"][0].removeprefix(
                    PANDAS_TEST_PREFIX
                )
                if fnmatch(line_module_name, pattern):
                    if line.get("longrepr"):
                        if isinstance(line["longrepr"], (tuple, list)):
                            message = line["longrepr"][2].splitlines()[0]
                        elif isinstance(line["longrepr"], str):
                            message = line["longrepr"]
                        else:
                            message = line["longrepr"]["reprcrash"][
                                "message"
                            ].splitlines()[0]
                        counter[message] += 1
    return counter


def render_results(results, num_rows=20):
    table = Table()
    table.add_column("Failure message")
    table.add_column("Number of occurrences")

    for msg, num in results.most_common(20):
        table.add_row(msg, str(num))

    console = Console()
    console.print(table)


if __name__ == "__main__":
    log_file_name = sys.argv[1]
    pattern = sys.argv[2]
    render_results(count_failures(log_file_name, pattern), num_rows=20)
