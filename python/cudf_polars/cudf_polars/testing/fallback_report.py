# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect upstream Polars GPU execution diagnostics from pytest reports."""

from __future__ import annotations

import json
import os
import platform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

OUTCOME_PRIORITY = {
    "incomplete": 0,
    "passed": 1,
    "xpassed": 2,
    "xfailed": 3,
    "skipped": 4,
    "failed": 5,
    "error": 6,
}
OBSERVATION_PRIORITY = {"false": 0, "unknown": 1, "true": 2}


class FallbackReport:
    """Aggregate test phases on the controller, including xdist worker reports."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.tests: dict[str, dict[str, str]] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        """Retain the strongest outcome and every execution signal per item."""
        outcome = "incomplete"
        if report.failed:
            outcome = "failed" if report.when == "call" else "error"
        elif report.skipped:
            outcome = "xfailed" if hasattr(report, "wasxfail") else "skipped"
        elif report.when == "call" and report.passed:
            outcome = "xpassed" if hasattr(report, "wasxfail") else "passed"

        properties = dict(report.user_properties)
        observations = {}
        for name in ("fallback", "gpu_attempted", "gpu_executed"):
            value = properties.get(f"cudf_polars_{name}", "unknown")
            observations[name] = (
                value
                if isinstance(value, str) and value in OBSERVATION_PRIORITY
                else "unknown"
            )
        previous = self.tests.get(report.nodeid)
        # Setup, call, and teardown arrive separately; failures in later phases
        # must not erase earlier observations or turn one item into several tests.
        if previous is not None:
            outcome = max(
                previous["outcome"], outcome, key=OUTCOME_PRIORITY.__getitem__
            )
            for name, value in observations.items():
                observations[name] = max(
                    previous[name], value, key=OBSERVATION_PRIORITY.__getitem__
                )
        self.tests[report.nodeid] = {
            "nodeid": report.nodeid,
            "outcome": outcome,
            **observations,
        }

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Write one diagnostic file after all worker reports have arrived."""
        config = session.config
        report = {
            "engine": config.getoption("--inject-gpu-engine"),
            "blocksize": config.getoption("--inject-gpu-engine-blocksize"),
            "shard_id": config.getoption("cudf_polars_shard_id"),
            "num_shards": config.getoption("cudf_polars_num_shards"),
            "python": platform.python_version(),
            "arch": platform.machine(),
            "cuda": os.environ.get("RAPIDS_CUDA_VERSION"),
            "dependencies": os.environ.get("RAPIDS_DEPENDENCIES"),
            "exitstatus": int(exitstatus),
            "collected": session.testscollected,
            "tests": sorted(self.tests.values(), key=lambda test: test["nodeid"]),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
