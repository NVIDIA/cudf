# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary multi-process controls for the intermittent TRANSFORM_TEST crash."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.test_dir = args.test_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    discovery = subprocess.check_output(
        ["ctest", "--show-only=json-v1", "-R", "^TRANSFORM_TEST$"],
        cwd=args.test_dir,
        text=True,
    )
    (args.output / "ctest.json").write_text(discovery)
    test = next(
        t
        for t in json.loads(discovery)["tests"]
        if t["name"] == "TRANSFORM_TEST"
    )
    properties = {p["name"]: p["value"] for p in test["properties"]}
    command = next(
        arg.removeprefix("-Dcommand_to_run=")
        for arg in test["command"]
        if arg.startswith("-Dcommand_to_run=")
    )
    binary = Path(properties["WORKING_DIRECTORY"]) / command
    binary = binary.resolve(strict=True)
    overrides = dict(
        entry.split("=", 1) for entry in properties["ENVIRONMENT"]
    )
    prefix = (
        "AssertsTest.*:RuntimeSupportTest.*:UnaryOperationIntegrationTest.*:"
        "TernaryOperationTest.TransformWithScalar"
    )
    seed = args.output / "cache-seed"
    source_cache = os.environ.get("LIBCUDF_KERNEL_CACHE_PATH")
    if source_cache and Path(source_cache).is_dir():
        shutil.copytree(source_cache, seed)
    else:
        seed.mkdir()
    hardware = {
        "cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    for name in ("memory.max", "memory.peak", "memory.events", "cpu.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            hardware[name] = path.read_text()
    (args.output / "host.json").write_text(json.dumps(hardware, indent=2))

    records = []
    for label, disabled, no_pch in (
        ("shared-restored", False, False),
        ("shared-uncached", True, False),
        ("shared-uncached-no-pch", True, True),
    ):
        group = args.output / label
        group.mkdir()
        cache = group / "cache"
        # Preserve the real cross-process RTCX sharing within each group, but
        # prevent a preceding group from warming the next group's cache.
        shutil.copytree(seed, cache)
        jobs = []
        print(f"::group::{label}", flush=True)
        for worker in range(args.workers):
            trial = group / f"worker-{worker}"
            trial.mkdir()
            env = os.environ.copy()
            env.update(overrides)
            env.update(
                GTEST_CUDF_RMM_MODE="async",
                LIBCUDF_KERNEL_CACHE_PATH=str(cache),
                LIBCUDF_KERNEL_CACHE_DISABLED="1" if disabled else "0",
                LIBCUDF_JIT_VERBOSE="1",
            )
            env.pop("NVRTC_DISABLE_PCH", None)
            if no_pch:
                env["NVRTC_DISABLE_PCH"] = "1"
            test_command = [
                str(binary),
                f"--gtest_filter={prefix}",
                f"--gtest_repeat={args.repeats}",
                "--gtest_print_time=1",
                f"--gtest_output=xml:{trial / 'results.xml'}",
            ]
            debugger = worker % 2 == 0
            if debugger:
                # Leave half the processes uninstrumented so a timing-sensitive
                # failure can still manifest without debugger perturbation.
                env["TRANSFORM_GDB_BINARY"] = str(binary)
                test_command = [
                    "bash",
                    str(Path(__file__).with_name("transform_gdb.sh")),
                    *test_command[1:],
                ]
            test_command = [
                "timeout",
                "--kill-after=30s",
                "10m",
                *test_command,
            ]
            record = {
                "label": label,
                "worker": worker,
                "debugger": debugger,
                "command": test_command,
                "cwd": properties["WORKING_DIRECTORY"],
                "host_load": os.getloadavg(),
                "environment": {
                    k: v
                    for k, v in env.items()
                    if k in overrides
                    or k.startswith(("GTEST_", "LIBCUDF_", "NVRTC_", "CUDA_"))
                },
            }
            (trial / "manifest.json").write_text(json.dumps(record, indent=2))
            log = (trial / "run.log").open("w")
            process = subprocess.Popen(
                test_command,
                cwd=properties["WORKING_DIRECTORY"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            jobs.append((record, process, log, trial, time.monotonic()))
        for record, process, log, trial, start in jobs:
            record["returncode"] = process.wait()
            record["observed_seconds"] = time.monotonic() - start
            log.close()
            text = (trial / "run.log").read_text()
            record["gdb_sigsegv"] = "received signal SIGSEGV" in text
            record["sigsegv"] = (
                record["returncode"] in (-11, 139) or record["gdb_sigsegv"]
            )
            record["pch_incomplete"] = text.count("because it is incomplete")
            record["pch_delete_errors"] = text.count("error while deleting")
            record["gtest_pass_summaries"] = text.count(
                "[  PASSED  ] 13 tests."
            )
            (trial / "manifest.json").write_text(json.dumps(record, indent=2))
            if (
                trial / "results.xml"
            ).exists() and "RAPIDS_TESTS_DIR" in os.environ:
                shutil.copy2(
                    trial / "results.xml",
                    Path(os.environ["RAPIDS_TESTS_DIR"])
                    / f"{label}-{record['worker']}.xml",
                )
            records.append(record)
            (args.output / "summary.json").write_text(
                json.dumps(records, indent=2)
            )
            print(text, flush=True)
            print(json.dumps(record), flush=True)
        print("::endgroup::", flush=True)
    raise SystemExit(any(r["returncode"] != 0 for r in records))


if __name__ == "__main__":
    main()
