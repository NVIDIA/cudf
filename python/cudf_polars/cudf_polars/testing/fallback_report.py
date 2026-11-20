# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect upstream Polars GPU execution diagnostics from pytest reports."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

import polars

if TYPE_CHECKING:
    from pathlib import Path


COLLECTION = pytest.StashKey[dict[str, object]]()

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
        self.tests: dict[str, dict[str, Any]] = {}
        self.collection: dict[str, object] = {}
        self.worker_failed = False

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        """Receive collection inventories from xdist workers."""
        workeroutput = getattr(node, "workeroutput", None)
        # Crashed workers do not send workeroutput. Keep their original failure
        # visible and prevent surviving workers from making the run look complete.
        if error is not None or workeroutput is None:
            self.worker_failed = True
            return
        inventory = workeroutput.get("cudf_polars_collection", {})
        if self.collection and inventory != self.collection:
            self.collection["consistent"] = False
        elif not self.collection:
            self.collection = inventory

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
            "complete": report.when == "teardown"
            or bool(previous and previous["complete"]),
            **observations,
        }

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Write one diagnostic file after all worker reports have arrived."""
        config = session.config
        collection = self.collection or config.stash.get(COLLECTION, {})
        if self.worker_failed:
            collection = {**collection, "consistent": False}
        upstream_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.rootpath,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        run_id = os.environ.get("GITHUB_RUN_ID")
        repository = os.environ.get("GITHUB_REPOSITORY")
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        report = {
            "polars_version": polars.__version__,
            "upstream_sha": upstream_sha,
            "cudf_sha": os.environ.get(
                "CUDF_POLARS_BASELINE_SHA", os.environ.get("GITHUB_SHA")
            ),
            "producer": os.environ.get("CUDF_POLARS_REPORT_PRODUCER"),
            "configuration": os.environ.get("CUDF_POLARS_REPORT_CONFIGURATION"),
            "workflow_run": (
                f"{server}/{repository}/actions/runs/{run_id}"
                if repository and run_id
                else None
            ),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            **collection,
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
