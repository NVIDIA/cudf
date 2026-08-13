#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Parse a GitHub Actions job URL and emit shell-friendly variable assignments.
#
# Usage:
#    python parse-job-url.py "https://github.com/rapidsai/cudf/actions/runs/<run_id>/job/<job_id>?pr=<pr_number>"
#
# Output (eval-able):
#    RUN_ID=XXXXXXXXXX
#    JOB_ID=XXXXXXXXX
#    PR_NUMBER=XXXX

import re
import sys
from urllib.parse import parse_qs, urlparse

PATTERN = re.compile(r"/actions/runs/(?P<run_id>\d+)/job/(?P<job_id>\d+)")


def parse(url: str) -> dict[str, str]:
    """Extract run ID, job ID, and optional PR number from a GitHub Actions job URL.

    Parameters
    ----------
    url : str
        A GitHub Actions job URL of the form:
        https://github.com/<owner>/<repo>/actions/runs/<run_id>/job/<job_id>?pr=<N>

    Returns
    -------
    dict[str, str]
        Dictionary with keys RUN_ID, JOB_ID, and optionally PR_NUMBER.

    Raises
    ------
    ValueError
        If the URL does not match the expected pattern or contains an invalid PR number.
    """
    parsed = urlparse(url)
    match = PATTERN.search(parsed.path)
    if not match:
        raise ValueError(
            f"URL does not match expected GitHub Actions job pattern: {url}"
        )

    result = {
        "RUN_ID": match.group("run_id"),
        "JOB_ID": match.group("job_id"),
    }

    qs = parse_qs(parsed.query)
    if "pr" in qs:
        pr_number = qs["pr"][0]
        if not pr_number.isdigit():
            raise ValueError(
                f"Invalid PR number in URL (expected digits): {pr_number!r}"
            )
        result["PR_NUMBER"] = pr_number

    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(  # noqa: T201
            f"Usage: {sys.argv[0]} <github_actions_job_url>", file=sys.stderr
        )
        sys.exit(1)

    try:
        for key, value in parse(sys.argv[1]).items():
            print(f"{key}={value}")  # noqa: T201
    except ValueError as exc:
        print(str(exc), file=sys.stderr)  # noqa: T201
        sys.exit(1)
