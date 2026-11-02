# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run CTest snapshots without changing installed scheduling metadata or source."""

import argparse
import fcntl
import fnmatch
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("--metadata", type=Path, required=True)
parser.add_argument("--config", type=Path)
parser.add_argument("--root", type=Path)
parser.add_argument("--jobs", type=int, default=20)
parser.add_argument("--gpu", default="6")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--cold", action="store_true")
parser.add_argument("--save-seed", action="store_true")
parser.add_argument(
    "--seed-regime", choices=["warm", "initial"], default="warm"
)
parser.add_argument("--jit-cache-path", type=Path)
args = parser.parse_args()
tools = Path(__file__).resolve().parent
root = args.root or tools.parent
root.mkdir(parents=True, exist_ok=True)
lock = (root / "measurement.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
trial = root / "trials" / args.name
trial.mkdir(parents=True, exist_ok=False)
(trial / "gtest").mkdir()
config = json.loads(args.config.read_text()) if args.config else {}
tests = json.loads(args.metadata.read_text())["tests"]
if config.get("only"):
    assert set(config["only"]) <= {test["name"] for test in tests}
    tests = [test for test in tests if test["name"] in config["only"]]
if config.get("trace") or config.get("case_events"):
    assert args.profile, (
        "Detailed tracing is diagnostic, never an acceptance run"
    )
uuid = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        args.gpu,
        "--query-gpu=uuid",
        "--format=csv,noheader",
    ],
    text=True,
).strip()
active = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader",
    ],
    text=True,
)
if any(line.startswith(uuid + ",") for line in active.splitlines()):
    raise SystemExit("Refusing measurement: selected GPU is occupied")


def quote(value):
    marker = "="
    while "]" + marker + "]" in str(value):
        marker += "="
    return "[" + marker + "[" + str(value) + "]" + marker + "]"


def cmake_value(value):
    if isinstance(value, list):
        return ";".join(str(part) for part in value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


lines = []
expected = []
original_properties = {}
for test in tests:
    if test["name"] == "generate_resource_spec":
        continue
    name = test["name"]
    props = {prop["name"]: prop["value"] for prop in test["properties"]}
    original_properties[name] = props
    assert props.get("DEPENDS", []) in [[], ["generate_resource_spec"]]
    assert props.get("FIXTURES_REQUIRED", []) in [[], ["resource_spec"]]
    groups = props["RESOURCE_GROUPS"]
    for part in config.get("partitions", {}).get(name, [dict(name=name)]):
        entry = part["name"]
        assert "/" not in entry and entry not in expected
        expected.append(entry)
        command = list(test["command"])
        if entry in config.get("trace", []):
            launch_index = next(
                i
                for i, value in enumerate(command)
                if value.startswith("-Dcommand_to_run=")
            )
            executable = command[launch_index].split("=", 1)[1]
            trace = [
                "nsys",
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--cuda-memory-usage=true",
                "--cuda-event-trace=false",
                "--output=" + str(trial / ("nsys-" + entry)),
                executable,
            ]
            # Leave the parent CMake process outside the diagnostic capture.
            command[launch_index] = "-Dcommand_to_run=" + ";".join(trace)
        if args.profile:
            command = [
                sys.executable,
                str(tools / "profile_one.py"),
                "--name",
                entry,
                "--directory",
                str(trial / "processes"),
                "--",
                *command,
            ]
        lines.append(
            "add_test("
            + " ".join(quote(value) for value in [entry, *command])
            + ")"
        )
        properties = dict(props)
        for generated in [
            "DEPENDS",
            "FIXTURES_REQUIRED",
            "FIXTURES_SETUP",
            "GENERATED_RESOURCE_SPEC_FILE",
        ]:
            properties.pop(generated, None)
        resource_groups = []
        for group in groups:
            resources = []
            for requirement in group["requirements"]:
                slots = requirement["slots"]
                if requirement[".type"] == "gpus":
                    slots = part.get(
                        "percent",
                        config.get("reservations", {}).get(name, slots),
                    )
                resources.append(f"{requirement['.type']}:{slots}")
            resource_groups.append(",".join(resources))
        properties["RESOURCE_GROUPS"] = ";".join(resource_groups)
        environment = [
            value
            for value in props.get("ENVIRONMENT", [])
            if not value.startswith(("GTEST_OUTPUT=", "GTEST_FILTER="))
        ]
        if entry in config.get("trace", []) and (
            config.get("trace_without_cupti_checker")
            or config.get("trace_checker_override")
        ):
            assert args.profile
            for value in environment:
                if value.startswith("LD_PRELOAD="):
                    assert (
                        "libcudf_identify_stream_usage" in value
                        and ":" not in value
                    )
            environment = [
                value
                for value in environment
                if not value.startswith("LD_PRELOAD=")
            ]
            if config.get("trace_checker_override"):
                environment.append(
                    "LD_PRELOAD=" + config["trace_checker_override"]
                )
        environment.append(
            "GTEST_OUTPUT=xml:" + str(trial / "gtest" / (entry + ".xml"))
        )
        if "filter" in part:
            environment.append("GTEST_FILTER=" + part["filter"])
        properties["ENVIRONMENT"] = environment
        for prop, field in [("PROCESSORS", "processors"), ("COST", "cost")]:
            if entry in config.get(field, {}):
                properties[prop] = config[field][entry]
        pairs = [
            token
            for key, value in properties.items()
            for token in [quote(key), quote(cmake_value(value))]
        ]
        lines.append(
            "set_tests_properties("
            + quote(entry)
            + " PROPERTIES "
            + " ".join(pairs)
            + ")"
        )
(trial / "CTestTestfile.cmake").write_text("\n".join(lines) + "\n")
(trial / "config.json").write_text(
    json.dumps(dict(config, jobs=args.jobs, profile=args.profile), indent=2)
)
(trial / "original-properties.json").write_text(
    json.dumps(original_properties, indent=2)
)
(trial / "resources.json").write_text(
    json.dumps(
        dict(
            version=dict(major=1, minor=0),
            local=[dict(gpus=[dict(id=args.gpu, slots=100)])],
        )
    )
)
cache = args.jit_cache_path or root / "jit-cache"
assert cache.is_absolute() or args.jit_cache_path is None
seed = root / "cache-seed"
driver_cache = root / "driver-cache"
cache_pairs = [(cache, seed), (driver_cache, root / "driver-cache-seed")]
archive_names = {cache: "jit-cache-after", driver_cache: "driver-cache-after"}
manifest = root / "cache-seed-manifest.json"
(trial / "cache-state.json").write_text(
    json.dumps(
        dict(
            regime="cold"
            if args.cold
            else (
                "seeded-warm"
                if args.seed_regime == "warm"
                else "restored-initial"
            ),
            seed_sha256=json.loads(manifest.read_text())["sha256"]
            if manifest.exists() and not args.cold
            else None,
            canonical_paths=[str(path) for path, initial in cache_pairs],
        ),
        indent=2,
    )
)
for current, initial in cache_pairs:
    if args.cold:
        assert not current.exists(), (
            "Cold runs require unused canonical cache paths"
        )
        current.mkdir()
    else:
        assert initial.is_dir() and not current.exists()
        shutil.copytree(initial, current)
env = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=args.gpu,
    LIBCUDF_KERNEL_CACHE_PATH=str(cache),
    CUDA_CACHE_PATH=str(driver_cache),
    TZ="UTC",
)
for key in [
    "GTEST_FILTER",
    "GTEST_OUTPUT",
    "GTEST_ALSO_RUN_DISABLED_TESTS",
    "GTEST_CUDF_MEMORY_PEAK",
    "LIBCUDF_NUM_HOST_WORKERS",
    "CUDF_TEST_PROFILE_DIR",
]:
    env.pop(key, None)
if args.profile:
    env["GTEST_CUDF_MEMORY_PEAK"] = "1"
if config.get("host_workers") is not None:
    env["LIBCUDF_NUM_HOST_WORKERS"] = str(config["host_workers"])
if config.get("case_events"):
    env["CUDF_TEST_PROFILE_DIR"] = str(trial / "case-events")
(trial / "environment.json").write_text(
    json.dumps(
        {
            key: env.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "LIBCUDF_KERNEL_CACHE_PATH",
                "CUDA_CACHE_PATH",
                "GTEST_CUDF_RMM_MODE",
                "LIBCUDF_NUM_HOST_WORKERS",
                "GTEST_CUDF_MEMORY_PEAK",
            ]
        },
        indent=2,
    )
)
memory_info = {
    key: int(value.strip().split()[0]) * 1024
    for key, value in (
        line.split(":", 1)
        for line in Path("/proc/meminfo").read_text().splitlines()
    )
    if value.strip().endswith("kB")
}
machine = dict(
    platform=platform.platform(),
    visible_cpus=os.cpu_count(),
    affinity=sorted(os.sched_getaffinity(0)),
    memory={
        key: memory_info[name]
        for key, name in [
            ("total", "MemTotal"),
            ("available", "MemAvailable"),
            ("free", "MemFree"),
        ]
    },
    cgroup={},
)
for name in [
    "/proc/self/cgroup",
    "/sys/fs/cgroup/cpu.max",
    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    "/sys/fs/cgroup/cpuset/cpuset.cpus",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
]:
    path = Path(name)
    if path.is_file():
        machine["cgroup"][name] = path.read_text().strip()
machine["gpu"] = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        args.gpu,
        "--query-gpu=name,uuid,driver_version,memory.total,pstate,temperature.gpu,clocks.sm,clocks.mem",
        "--format=csv",
    ],
    text=True,
).strip()
(trial / "machine.json").write_text(json.dumps(machine, indent=2))
with (
    (trial / "ctest.log").open("w") as log,
    (trial / "gpu.csv").open("w") as samples,
):
    sampler = None
    if args.profile:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                args.gpu,
                "--query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem",
                "--format=csv,nounits",
                "--loop-ms=100",
            ],
            env=env,
            stdout=samples,
            stderr=subprocess.DEVNULL,
        )
    start = time.monotonic_ns()
    command = [
        "timeout",
        "30m",
        "ctest",
        "-j" + str(args.jobs),
        "--output-on-failure",
        "--no-tests=error",
        "--resource-spec-file=" + str(trial / "resources.json"),
        "--output-junit=" + str(trial / "ctest.xml"),
    ]
    process = subprocess.Popen(
        command,
        cwd=trial,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )

    def cancel(signum, frame):
        # The new session isolates this trial's CTest tree from unrelated host jobs.
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, cancel)
    try:
        with (trial / "ctest-events.jsonl").open("w") as events:
            for line in process.stdout:
                now = time.monotonic_ns()
                log.write(line)
                log.flush()
                match = re.search(r"Start\s+\d+:\s+(\S+)", line)
                kind = "start"
                if not match:
                    match = re.search(r"Test\s+#\d+:\s+(\S+)\s+\.+\s+", line)
                    kind = "end"
                if match:
                    events.write(
                        json.dumps(
                            dict(monotonic_ns=now, event=kind, name=match[1])
                        )
                        + "\n"
                    )
                    events.flush()
        returncode = process.wait()
    finally:
        end = time.monotonic_ns()
        if sampler:
            sampler.terminate()
            sampler.wait()
        for current, initial in cache_pairs:
            current.rename(trial / archive_names[current])
cases = ET.parse(trial / "ctest.xml").findall(".//testcase")
assert {case.get("name") for case in cases} == set(expected)
outcomes = Counter()
gtest_failures = []
reports = sorted((trial / "gtest").glob("*.xml"))
missing = (
    set(expected)
    - {report.stem for report in reports}
    - {"STREAM_IDENTIFICATION_TEST"}
)
assert not missing, ("Missing GTest XML reports", missing)
for report in reports:
    for case in ET.parse(report).findall(".//testcase"):
        if case.get("status") == "notrun":
            continue
        outcomes[
            (
                case.get("classname"),
                case.get("name"),
                case.get("status"),
                case.get("result"),
                case.find("skipped") is not None,
            )
        ] += 1
        if case.find("failure") is not None:
            gtest_failures.append(
                dict(
                    binary=report.stem,
                    suite=case.get("classname"),
                    name=case.get("name"),
                )
            )
serialized = [
    dict(
        suite=key[0],
        name=key[1],
        status=key[2],
        result=key[3],
        skipped=key[4],
        count=count,
    )
    for key, count in sorted(outcomes.items())
]
(trial / "outcomes.json").write_text(json.dumps(serialized, indent=2))
reference = root / "baseline-outcomes.json"
coverage_matches = None
if reference.exists() or not args.cold:
    if config.get("only"):
        expected_outcomes = Counter()
        for test in tests:
            name = test["name"]
            if name == "STREAM_IDENTIFICATION_TEST":
                continue
            original = root / "trials/native-warm-0/gtest" / (name + ".xml")
            parts = config.get("partitions", {}).get(name, [dict(name=name)])
            for part in parts:
                positive, _, negative = part.get("filter", "*").partition("-")
                for case in ET.parse(original).findall(".//testcase"):
                    node = case.get("classname") + "." + case.get("name")
                    if case.get("status") == "notrun":
                        continue
                    if not any(
                        fnmatch.fnmatchcase(node, pattern)
                        for pattern in positive.split(":")
                    ):
                        continue
                    if negative and any(
                        fnmatch.fnmatchcase(node, pattern)
                        for pattern in negative.split(":")
                    ):
                        continue
                    expected_outcomes[
                        (
                            case.get("classname"),
                            case.get("name"),
                            case.get("status"),
                            case.get("result"),
                            case.find("skipped") is not None,
                        )
                    ] += 1
        coverage_matches = outcomes == expected_outcomes
    elif reference.exists():
        coverage_matches = serialized == json.loads(reference.read_text())
    elif returncode == 0 and not config:
        reference.write_text(json.dumps(serialized, indent=2))
        coverage_matches = True
if gtest_failures and returncode == 0:
    returncode = 1
summary = dict(
    name=args.name,
    returncode=returncode,
    seconds=(end - start) / 1e9,
    monotonic_start_ns=start,
    monotonic_end_ns=end,
    executables=len(cases),
    summed_executable_seconds=sum(
        float(case.get("time", 0)) for case in cases
    ),
    run_cases=sum(outcomes.values()),
    skipped=sum(count for key, count in outcomes.items() if key[-1]),
    coverage_matches=coverage_matches,
    gtest_failures=gtest_failures,
    failures=[
        case.get("name") for case in cases if case.find("failure") is not None
    ],
)
(trial / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
if args.save_seed:
    assert returncode == 0
    for current, initial in cache_pairs:
        assert not initial.exists()
        shutil.copytree(trial / (current.name + "-after"), initial)
assert coverage_matches is not False, "Case outcomes or skip coverage changed"
raise SystemExit(returncode)
