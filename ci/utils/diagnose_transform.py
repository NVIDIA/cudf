# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeat normal concurrent CTest runs with a native SIGSEGV recorder."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def host_metrics():
    result = {
        "load": os.getloadavg(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    for name in (
        "memory.max",
        "memory.current",
        "memory.peak",
        "memory.events",
        "cpu.max",
        "cpu.stat",
    ):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result[name] = path.read_text()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crash-library", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument(
        "--ctest-filter", help="Local smoke-test selection only"
    )
    args = parser.parse_args()
    args.test_dir = args.test_dir.resolve()
    args.output = args.output.resolve()
    args.crash_library = args.crash_library.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    discovery = subprocess.check_output(
        ["ctest", "--show-only=json-v1", "-R", "^TRANSFORM_TEST$"],
        cwd=args.test_dir,
        text=True,
    )
    (args.output / "ctest-before.json").write_text(discovery)
    test = next(
        t
        for t in json.loads(discovery)["tests"]
        if t["name"] == "TRANSFORM_TEST"
    )
    properties = {p["name"]: p["value"] for p in test["properties"]}
    preload = next(
        v for v in properties["ENVIRONMENT"] if v.startswith("LD_PRELOAD=")
    )
    replacement = preload.replace(
        "LD_PRELOAD=", f"LD_PRELOAD={args.crash_library}:", 1
    )
    modification = f"LD_PRELOAD=path_list_prepend:{args.crash_library}"
    originals = {}
    cost = args.test_dir / "Testing/Temporary/CTestCostData.txt"
    initial_cost = cost.read_bytes() if cost.exists() else None
    seed = args.output / "cache-seed"
    source_cache = os.environ.get("LIBCUDF_KERNEL_CACHE_PATH")
    if source_cache and Path(source_cache).is_dir():
        shutil.copytree(source_cache, seed)
    else:
        seed.mkdir()
    records = []
    try:
        # Installed CTest definitions resolve paths at runtime, so their raw
        # preload strings need not match discovery's expanded paths.
        path = args.test_dir / "CTestTestfile.cmake"
        originals[path] = path.read_text()
        modifications = ";".join(
            [*properties.get("ENVIRONMENT_MODIFICATION", []), modification]
        )
        path.write_text(
            originals[path]
            + "\nset_tests_properties(TRANSFORM_TEST PROPERTIES "
            + f"ENVIRONMENT_MODIFICATION [==[{modifications}]==])\n"
        )
        discovery = subprocess.check_output(
            ["ctest", "--show-only=json-v1", "-R", "^TRANSFORM_TEST$"],
            cwd=args.test_dir,
            text=True,
        )
        (args.output / "ctest-instrumented.json").write_text(discovery)
        instrumented = next(
            t
            for t in json.loads(discovery)["tests"]
            if t["name"] == "TRANSFORM_TEST"
        )
        updated_properties = {
            p["name"]: p["value"] for p in instrumented["properties"]
        }
        assert modification in updated_properties["ENVIRONMENT_MODIFICATION"]
        assert updated_properties["ENVIRONMENT"] == properties["ENVIRONMENT"]
        assert instrumented["command"] == test["command"]
        for trial_number in range(1, args.trials + 1):
            trial = args.output / f"native-{trial_number}"
            trial.mkdir()
            cache = trial / "cache"
            shutil.copytree(seed, cache)
            if initial_cost is not None:
                cost.write_bytes(initial_cost)
            else:
                cost.unlink(missing_ok=True)
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
                "trial": trial_number,
                "command": command,
                "host_load": os.getloadavg(),
                "preload": replacement,
                "cache": str(cache),
                "host_before": host_metrics(),
            }
            print(f"::group::native-{trial_number}", flush=True)
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
                host_after=host_metrics(),
            )
            for name in (
                "LastTest.log",
                "LastTestsFailed.log",
                "CTestCostData.txt",
            ):
                path = cost.parent / name
                if path.exists():
                    shutil.copy2(path, trial / name)
            last_test = trial / "LastTest.log"
            record["sigsegv_recorded"] = result.returncode != 0 and (
                last_test.exists()
                and "CUDF_DIAGNOSTIC_SIGSEGV" in last_test.read_text()
            )
            if (trial / "ctest.xml").exists() and "RAPIDS_TESTS_DIR" in env:
                shutil.copy2(
                    trial / "ctest.xml",
                    Path(env["RAPIDS_TESTS_DIR"])
                    / f"native-{trial_number}.xml",
                )
            records.append(record)
            (args.output / "summary.json").write_text(
                json.dumps(records, indent=2)
            )
            print((trial / "run.log").read_text(), flush=True)
            print(json.dumps(record), flush=True)
            print("::endgroup::", flush=True)
            # A captured crash is the objective; stop spending GPU time once
            # its register state, mappings, and stack have been saved.
            if record["sigsegv_recorded"]:
                break
    finally:
        for path, text in originals.items():
            path.write_text(text)
        if initial_cost is not None:
            cost.write_bytes(initial_cost)
        else:
            cost.unlink(missing_ok=True)
    raise SystemExit(any(r["returncode"] != 0 for r in records))


if __name__ == "__main__":
    main()
