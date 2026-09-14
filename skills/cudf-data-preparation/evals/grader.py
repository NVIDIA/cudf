#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic companion grader for cuDF preparation evals."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

TRAJECTORY = Path(os.environ.get("HARBOR_ATIF_PATH", "/logs/agent/trajectory.json"))
ENTRY = Path(os.environ.get("HARBOR_ENTRY_JSON", "/tests/entry.json"))
REWARD_JSON = Path(os.environ.get("HARBOR_REWARD_JSON", "/logs/verifier/reward.json"))
REWARD_TXT = Path(os.environ.get("HARBOR_REWARD_TXT", "/logs/verifier/reward.txt"))
WORKSPACE = Path(os.environ.get("HARBOR_WORKSPACE", "/workspace"))


def trajectory_text():
    if not TRAJECTORY.exists():
        return ""
    data = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    return " ".join(json.dumps(step.get("message", "")) for step in data.get("steps", []) if step.get("source") == "agent").lower()


def find_fixture(filename):
    matches = list(WORKSPACE.rglob(filename))
    return matches[0].resolve() if matches else None


def main():
    entry = json.loads(ENTRY.read_text())
    case_id = entry["id"]
    text = trajectory_text()
    checks = {}

    if case_id == "cudf-prep-explicit-join-fanout":
        source = find_fixture("prepare_orders.py")
        smoke = find_fixture("smoke.py")
        source_text = source.read_text().lower() if source else ""
        trusted = bool(smoke and hashlib.sha256(smoke.read_bytes()).hexdigest() == "ac96bfe7cb7350090dc986c1b1f5cd3199fb499cd698982490f288ca575760a3")
        result = subprocess.run(["python3", str(smoke)], cwd=smoke.parent.parent, capture_output=True, text=True) if trusted else None
        checks = {
            "semantic_smoke": (0.6, bool(trusted and result and result.returncode == 0)),
            "no_silent_dedup": (0.2, "drop_duplicates" not in source_text),
            "execution_evidence": (0.2, "cpu" in text and ("gpu" in text or "cudf" in text)),
        }
    elif case_id == "cudf-prep-implicit-time-cutoff":
        checks = {
            "event_and_availability": (0.25, "event_time" in text and "available_at" in text),
            "boundary_rule": (0.25, any(term in text for term in ("strict", "inclusive", "< cutoff", "<= cutoff", "< score_at", "<= score_at"))),
            "boundary_tests": (0.25, ("below" in text and "at" in text and "above" in text) or ("just before" in text and "exactly at" in text and "just after" in text)),
            "keyed_snapshot": (0.25, "customer_id" in text and any(term in text for term in ("cutoff", "snapshot", "origin"))),
        }
    elif case_id == "cudf-prep-contextual-population":
        checks = {
            "population_spine": (0.25, "spine" in text and "left" in text),
            "zero_vs_unknown": (0.25, "zero" in text and any(term in text for term in ("unknown", "null", "missing"))),
            "grain_and_coverage": (0.25, "customer" in text and "month" in text and any(term in text for term in ("coverage", "every", "full population", "one row per"))),
            "no_global_fill": (0.25, any(term in text for term in ("not globally", "not blanket", "only fill", "only when", "do not fill"))),
        }
    else:
        checks = {
            "routes_to_cuml": (0.4, "cuml" in text or "machine learning" in text),
            "accepts_upstream_table": (0.3, "feature table" in text and not any(term in text for term in ("clean the feature", "rebuild the feature", "audit the feature"))),
            "avoids_cudf_audit": (0.3, "cudf" not in text and "data-preparation audit" not in text),
        }

    score = sum(weight for weight, passed in checks.values() if passed)
    details = {name: {"score": float(passed), "reason": "passed" if passed else "failed"} for name, (_, passed) in checks.items()}
    reward = {"overall": score, "custom_metrics": {"contract_correctness": score}, "details": details}
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps(reward, indent=2))
    REWARD_TXT.write_text(str(score))


if __name__ == "__main__":
    main()
