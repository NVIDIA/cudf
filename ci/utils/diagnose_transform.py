# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary allocator/cache controls for the intermittent TRANSFORM_TEST crash."""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    # Preserve the installed test's non-default stream and API-checking preload.
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
    overrides = dict(
        entry.split("=", 1) for entry in properties["ENVIRONMENT"]
    )
    prefix = (
        "AssertsTest.*:RuntimeSupportTest.*:UnaryOperationIntegrationTest.*:"
        "TernaryOperationTest.TransformWithScalar"
    )
    records = []

    def run(
        label, allocator, cache, disabled=False, debugger=False, full=False
    ):
        trial = args.output / label
        trial.mkdir()
        cache.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.pop("LIBCUDF_KERNEL_CACHE_DISABLED", None)
        env.update(overrides)
        env.update(
            GTEST_CUDF_RMM_MODE=allocator,
            LIBCUDF_KERNEL_CACHE_PATH=str(cache),
            LIBCUDF_JIT_VERBOSE="1",
        )
        if disabled:
            env["LIBCUDF_KERNEL_CACHE_DISABLED"] = "1"
        repeat = 1 if full else args.repeats
        if disabled:
            repeat = min(repeat, 3)
        test_command = [
            command,
            f"--gtest_filter={'*' if full else prefix}",
            f"--gtest_repeat={repeat}",
            "--gtest_break_on_failure",
            "--gtest_print_time=1",
            f"--gtest_output=xml:{trial / 'results.xml'}",
        ]
        process_env = env.copy()
        if debugger:
            # Preloading libcudf's stream subscriber into GDB itself would alter
            # the debugger; apply the test environment only to its inferior.
            process_env.pop("LD_PRELOAD", None)
            assignments = [
                f"{key}={value}"
                for key, value in env.items()
                if key in overrides or key.startswith(("GTEST_", "LIBCUDF_"))
            ]
            test_command = [
                "gdb",
                "--batch",
                "--return-child-result",
                "-ex",
                "set pagination off",
                "-ex",
                "set confirm off",
                "-ex",
                "set disable-randomization off",
                "-ex",
                "set debuginfod enabled off",
                "-ex",
                "set print elements 32",
                "-ex",
                "set exec-wrapper env " + shlex.join(assignments),
                "-ex",
                "run",
                "-ex",
                "thread apply all bt full",
                "-ex",
                "info sharedlibrary",
                "--args",
                *test_command,
            ]
        record = {
            "label": label,
            "allocator": allocator,
            "host_load": os.getloadavg(),
            "cache_disabled": disabled,
            "debugger": debugger,
            "command": test_command,
            "environment": {
                key: value
                for key, value in env.items()
                if key in overrides
                or key.startswith(("GTEST_", "LIBCUDF_", "CUDA_"))
            },
        }
        print(f"::group::{label}", flush=True)
        start = time.monotonic()
        with (trial / "run.log").open("w") as log:
            with subprocess.Popen(
                test_command,
                cwd=properties["WORKING_DIRECTORY"],
                env=process_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as process:
                try:
                    record["returncode"] = process.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    # Kill the inferior too, so it cannot interfere with the
                    # next allocator control after its debugger times out.
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    record["returncode"] = 124
        record["seconds"] = time.monotonic() - start
        (trial / "manifest.json").write_text(json.dumps(record, indent=2))
        if (trial / "results.xml").exists() and "RAPIDS_TESTS_DIR" in env:
            shutil.copyfile(
                trial / "results.xml",
                Path(env["RAPIDS_TESTS_DIR"]) / f"{label}.xml",
            )
        print((trial / "run.log").read_text(), flush=True)
        print(json.dumps(record), flush=True)
        print("::endgroup::", flush=True)
        records.append(record)
        # Persist partial results even if the runner later gets cancelled.
        (args.output / "summary.json").write_text(
            json.dumps(records, indent=2)
        )

    for allocator in ("async", "pool"):
        cache = args.output / f"cache-{allocator}"
        run(f"{allocator}-fresh", allocator, cache)
        run(f"{allocator}-reused", allocator, cache)
        run(f"{allocator}-disabled", allocator, cache, disabled=True)
        run(
            f"{allocator}-gdb-fresh",
            allocator,
            args.output / f"cache-gdb-{allocator}",
            debugger=True,
        )
        run(
            f"{allocator}-gdb-disabled",
            allocator,
            cache,
            disabled=True,
            debugger=True,
        )
        # A complete-binary run checks for effects from tests outside the prefix.
        run(
            f"{allocator}-gdb-full", allocator, cache, debugger=True, full=True
        )

    raise SystemExit(any(record["returncode"] != 0 for record in records))


if __name__ == "__main__":
    main()
