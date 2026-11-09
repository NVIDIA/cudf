# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary full-suite concurrency controls for the TRANSFORM_TEST crash."""

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
    parser.add_argument(
        "--ctest-filter", help="Local smoke-test selection only"
    )
    args = parser.parse_args()
    args.test_dir = args.test_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    discovery = subprocess.check_output(
        ["ctest", "--show-only=json-v1"], cwd=args.test_dir, text=True
    )
    (args.output / "ctest.json").write_text(discovery)
    test = next(
        t
        for t in json.loads(discovery)["tests"]
        if t["name"] == "TRANSFORM_TEST"
    )
    properties = {p["name"]: p["value"] for p in test["properties"]}
    binary = Path(
        next(
            arg.removeprefix("-Dcommand_to_run=")
            for arg in test["command"]
            if arg.startswith("-Dcommand_to_run=")
        )
    )
    if not binary.is_absolute():
        binary = Path(properties["WORKING_DIRECTORY"]) / binary
    binary = binary.resolve(strict=True)
    backup = binary.with_name(binary.name + ".diagnostic-real")
    if backup.exists():
        raise RuntimeError(f"Refusing to overwrite existing backup: {backup}")

    # Prevent the first pass's measured durations from changing the second
    # pass's scheduling order; CTest still manages the original GPU slots.
    cost_file = args.test_dir / "Testing/Temporary/CTestCostData.txt"
    initial_cost = cost_file.read_bytes() if cost_file.exists() else None
    if initial_cost is not None:
        (args.output / "initial-CTestCostData.txt").write_bytes(initial_cost)

    # Both passes start with the same restored cache, not a cache warmed by
    # the preceding full-suite run.
    seed = args.output / "cache-seed"
    source_cache = os.environ.get("LIBCUDF_KERNEL_CACHE_PATH")
    if source_cache and Path(source_cache).is_dir():
        shutil.copytree(source_cache, seed)
    else:
        seed.mkdir()
    records = []
    try:
        for label in ("concurrent-plain", "concurrent-gdb"):
            trial = args.output / label
            trial.mkdir()
            cache = trial / "cache"
            shutil.copytree(seed, cache)
            if initial_cost is not None:
                cost_file.write_bytes(initial_cost)
            else:
                cost_file.unlink(missing_ok=True)
            if label == "concurrent-gdb":
                # Keep CTest's command, stream-checking environment, working
                # directory and resource allocation intact for every suite.
                binary.rename(backup)
                wrapper = Path(__file__).with_name("transform_gdb.sh")
                shutil.copy2(wrapper, binary)
                binary.chmod(0o755)
            env = os.environ.copy()
            env.update(
                GTEST_CUDF_RMM_MODE="async",
                LIBCUDF_KERNEL_CACHE_PATH=str(cache),
                GTEST_OUTPUT=f"xml:{trial}/gtest-xml/",
            )
            (trial / "gtest-xml").mkdir()
            command = [
                "timeout",
                "--kill-after=30s",
                "30m",
                "ctest",
                "--output-on-failure",
                "--no-tests=error",
                "-j20",
                "--output-junit",
                str(trial / "ctest.xml"),
            ]
            if args.ctest_filter:
                command.extend(["-R", args.ctest_filter])
            record = {
                "label": label,
                "command": command,
                "host_load": os.getloadavg(),
                "allocator": env["GTEST_CUDF_RMM_MODE"],
                "cache": str(cache),
            }
            print(f"::group::{label}", flush=True)
            start = time.monotonic()
            with (trial / "run.log").open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=args.test_dir,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            record.update(
                returncode=result.returncode,
                seconds=time.monotonic() - start,
            )
            for name in (
                "LastTest.log",
                "LastTestsFailed.log",
                "CTestCostData.txt",
            ):
                path = cost_file.parent / name
                if path.exists():
                    shutil.copy2(path, trial / name)
            if (trial / "ctest.xml").exists() and "RAPIDS_TESTS_DIR" in env:
                shutil.copy2(
                    trial / "ctest.xml",
                    Path(env["RAPIDS_TESTS_DIR"]) / f"{label}.xml",
                )
            records.append(record)
            (args.output / "summary.json").write_text(
                json.dumps(records, indent=2)
            )
            print((trial / "run.log").read_text(), flush=True)
            print(json.dumps(record), flush=True)
            print("::endgroup::", flush=True)
    finally:
        if backup.exists():
            backup.replace(binary)
        if initial_cost is not None:
            cost_file.write_bytes(initial_cost)
        else:
            cost_file.unlink(missing_ok=True)

    raise SystemExit(any(record["returncode"] != 0 for record in records))


if __name__ == "__main__":
    main()
