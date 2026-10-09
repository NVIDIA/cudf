# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the example runner without a GPU or installed libcudf."""

import collections
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_cudf_examples.sh"
PROGRAMS = {
    "basic": ["basic_example"],
    "nested_types": ["deduplication"],
    "hybrid_scan_io": [
        "hybrid_scan_io",
        "hybrid_scan_pipeline",
        "hybrid_scan_multifile_single_step",
        "hybrid_scan_multifile_two_step",
    ],
    "string_transformers": [
        "compute_checksum_jit",
        "extract_email_jit",
        "extract_email_precompiled",
        "format_phone_jit",
        "format_phone_precompiled",
        "localize_phone_jit",
        "localize_phone_precompiled",
        "url_log_transforms",
    ],
    "parquet_io": ["parquet_io", "parquet_io_multithreaded"],
    "strings": ["custom_optimized", "custom_prealloc", "custom_with_malloc"],
    "parquet_inspect": ["parquet_inspect"],
}
EXPECTED_COUNTS = {
    "basic": 1,
    "nested_types": 1,
    "hybrid_scan_io": 7,
    "string_transformers": 11,
    "parquet_io": 4,
    "strings": 3,
    "parquet_inspect": 1,
}
MOCK = r"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

name = Path(sys.argv[0]).name
if name == "nvidia-smi":
    print(os.environ.get("MOCK_COMPUTE_CAP", "7.0"))
    sys.exit(0)
if name == "compute-sanitizer":
    args = sys.argv[1:]
    options = dict(zip(args[:6:2], args[1:6:2]))
    assert options["--tool"] == "memcheck"
    assert options["--error-exitcode"] == "1"
    os.environ["MOCK_PORT"] = options["--port"]
    sys.exit(subprocess.call(args[6:]))

group = Path.cwd().name
def record(kind):
    with Path(os.environ["MOCK_EVENTS"]).open("a") as events:
        fcntl.flock(events, fcntl.LOCK_EX)
        events.write(json.dumps({"kind": kind, "group": group, "program": name,
                                 "args": sys.argv[1:], "port": os.environ.get("MOCK_PORT")}) + "\n")
record("start")
time.sleep(0.3 if group == os.environ.get("MOCK_SLOW_GROUP") else 0.04)
record("end")
sys.exit(1 if os.environ.get("MOCK_FAIL") == group + "/" + name else 0)
"""


class ExampleRunnerTests(unittest.TestCase):
    def run_examples(self, level="2", **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install = root / "install"
            binary_dir = root / "bin"
            binary_dir.mkdir()
            mock = binary_dir / "mock"
            mock.write_text(f"#!{sys.executable}\n" + MOCK)
            mock.chmod(0o755)
            for name in ("nvidia-smi", "compute-sanitizer"):
                (binary_dir / name).symlink_to(mock)
            for group, programs in PROGRAMS.items():
                group_dir = install / "bin/examples/libcudf" / group
                group_dir.mkdir(parents=True)
                for program in programs:
                    (group_dir / program).symlink_to(mock)
            events_path = root / "events"
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env={
                    **os.environ,
                    "PATH": f"{binary_dir}:{os.environ['PATH']}",
                    "INSTALL_PREFIX": str(install),
                    "EXAMPLES_PARALLEL_LEVEL": level,
                    "MOCK_EVENTS": str(events_path),
                    **overrides,
                },
                text=True,
                capture_output=True,
                timeout=30,
            )
            events = (
                [
                    json.loads(line)
                    for line in events_path.read_text().splitlines()
                ]
                if events_path.exists()
                else []
            )
            return result, events

    def check_schedule(self, events, expected_peak):
        active = set()
        peak = 0
        completed_serial = set()
        for event in events:
            group = event["group"]
            if event["kind"] == "start":
                self.assertNotIn(group, active)
                if group in ("basic", "nested_types"):
                    self.assertFalse(active)
                else:
                    self.assertEqual(
                        completed_serial, {"basic", "nested_types"}
                    )
                active.add(group)
                peak = max(peak, len(active))
            else:
                active.remove(group)
                if group in ("basic", "nested_types"):
                    completed_serial.add(group)
        self.assertFalse(active)
        self.assertEqual(peak, expected_peak)
        self.assertEqual(
            collections.Counter(
                e["group"] for e in events if e["kind"] == "start"
            ),
            EXPECTED_COUNTS,
        )

    def test_serial_and_parallel_schedules(self):
        for level in ("", "1", "2"):
            with self.subTest(level=level):
                result, events = self.run_examples(level)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.check_schedule(events, int(level or "2"))
                headings = [
                    line[len("::group::Examples: ") :]
                    for line in result.stdout.splitlines()
                    if line.startswith("::group::Examples: ")
                ]
                self.assertEqual(headings, list(PROGRAMS))
                self.assertIn("Example stage: elapsed=", result.stdout)
                for group in PROGRAMS:
                    self.assertIn(
                        f"{group}: status=0, elapsed=", result.stdout
                    )
                ports = collections.defaultdict(set)
                for event in events:
                    ports[event["group"]].add(event["port"])
                self.assertEqual(
                    dict(ports),
                    {
                        group: {str(49152 + i * 100)}
                        for i, group in enumerate(PROGRAMS)
                    },
                )

    def test_finished_slot_is_refilled(self):
        result, events = self.run_examples(MOCK_SLOW_GROUP="hybrid_scan_io")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.check_schedule(events, 2)
        parquet_start = next(
            i
            for i, event in enumerate(events)
            if event["group"] == "parquet_io" and event["kind"] == "start"
        )
        hybrid_end = max(
            i
            for i, event in enumerate(events)
            if event["group"] == "hybrid_scan_io" and event["kind"] == "end"
        )
        self.assertLess(parquet_start, hybrid_end)

    def test_early_failure_survives_later_success(self):
        for failure in (
            "basic/basic_example",
            "hybrid_scan_io/hybrid_scan_io",
            "parquet_inspect/parquet_inspect",
        ):
            with self.subTest(failure=failure):
                result, events = self.run_examples(MOCK_FAIL=failure)
                self.assertEqual(result.returncode, 1)
                self.check_schedule(events, 2)
                self.assertIn(
                    f"{failure.split('/')[0]}: status=1", result.stdout
                )
                self.assertEqual(
                    len([e for e in events if e["kind"] == "end"]), 28
                )

    def test_sm120_does_not_use_sanitizer(self):
        result, events = self.run_examples(MOCK_COMPUTE_CAP="12.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.check_schedule(events, 2)
        self.assertTrue(all(event["port"] is None for event in events))

    def test_invalid_parallel_level_runs_nothing(self):
        for level in ("0", "3", "invalid"):
            with self.subTest(level=level):
                result, events = self.run_examples(level)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(events, [])
                self.assertIn("must be 1 or 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
