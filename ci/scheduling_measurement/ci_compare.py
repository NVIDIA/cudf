# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary matched CI measurements using one installed set of test binaries."""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from copy import deepcopy
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--test-dir", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--metadata", type=Path)
parser.add_argument("--baseline-metadata", type=Path)
parser.add_argument("--candidate-metadata", type=Path)
parser.add_argument("--gpu", default="0")
parser.add_argument("--pairs", type=int, choices=[1, 3], default=3)
parser.add_argument("--regime", choices=["warm", "initial"], default="warm")
args = parser.parse_args()
candidate = json.loads(args.candidate.read_text())
assert set(candidate) <= {"partitions", "reservations", "processors", "cost"}
assert not os.environ.get("GTEST_FILTER"), "The CI baseline must be unfiltered"
original_cache = (
    Path(os.environ["LIBCUDF_KERNEL_CACHE_PATH"]).absolute()
    if os.environ.get("LIBCUDF_KERNEL_CACHE_PATH")
    else None
)
if original_cache:
    workspace_cache = Path.cwd().resolve() / ".cache"
    resolved_cache = original_cache.resolve()
    assert resolved_cache != workspace_cache and resolved_cache.is_relative_to(
        workspace_cache
    ), (
        "Only a specific job-owned workspace .cache subdirectory may be temporarily replaced"
    )
    assert not original_cache.is_symlink()
    original_cache.parent.mkdir(parents=True, exist_ok=True)
root = Path(
    tempfile.mkdtemp(
        prefix="cudf-ci-scheduling-",
        dir=original_cache.parent if original_cache else None,
    )
)
cache_backup = root / "restored-original-cache"
cache_taken = False
tools = Path(__file__).resolve().parent
workers = int(
    os.environ.get("LIBCUDF_NUM_HOST_WORKERS")
    or min(16, (os.cpu_count() or 1) // 4)
)
configs = {
    "baseline": dict(host_workers=workers),
    "candidate": dict(candidate, host_workers=workers),
}
metadata = (
    json.loads(args.metadata.read_text())
    if args.metadata
    else json.loads(
        subprocess.check_output(
            ["ctest", "--show-only=json-v1"], cwd=args.test_dir, text=True
        )
    )
)
matrices = {
    kind: json.loads(source.read_text()) if source else deepcopy(metadata)
    for kind, source in [
        ("baseline", args.baseline_metadata),
        ("candidate", args.candidate_metadata),
    ]
}
for matrix in [metadata, *matrices.values()]:
    for test in matrix["tests"]:
        if test["name"] == "generate_resource_spec":
            continue
        if not any(
            prop["name"] == "WORKING_DIRECTORY" for prop in test["properties"]
        ):
            test["properties"].append(
                dict(
                    name="WORKING_DIRECTORY",
                    value=str(args.test_dir.resolve()),
                )
            )
original_names = {
    test["name"]
    for test in matrices["candidate"]["tests"]
    if test["name"] != "generate_resource_spec"
}
assert set(candidate.get("partitions", {})) <= original_names
effective_names = set()
for name in original_names:
    effective_names.update(
        part["name"]
        for part in candidate.get("partitions", {}).get(
            name, [dict(name=name)]
        )
    )
assert set(candidate.get("reservations", {})) <= original_names
for field in ["cost", "processors"]:
    assert set(candidate.get(field, {})) <= effective_names
(root / "tests.json").write_text(json.dumps(metadata, indent=2))
for name, config in configs.items():
    (root / (name + ".json")).write_text(json.dumps(config, indent=2))
    (root / (name + "-tests.json")).write_text(
        json.dumps(matrices[name], indent=2)
    )
state = dict(process=None, cancelled=None, registered=None)


def cancel(signum, frame):
    state["cancelled"] = signum
    process = state["process"]
    if process is not None and process.poll() is None:
        process.send_signal(signum)


for signum in [signal.SIGTERM, signal.SIGINT]:
    signal.signal(signum, cancel)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seed(source, destination):
    if source and source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.mkdir()


def run(name, kind):
    if state["cancelled"]:
        raise SystemExit(128 + state["cancelled"])
    command = [
        sys.executable,
        str(tools / "run_suite.py"),
        name,
        "--metadata",
        str(root / (kind + "-tests.json")),
        "--root",
        str(root),
        "--gpu",
        args.gpu,
        "--config",
        str(root / (kind + ".json")),
        "--seed-regime",
        "initial" if name == "warmup" or args.regime == "initial" else "warm",
    ]
    if original_cache:
        command += ["--jit-cache-path", str(original_cache)]
    print("CUDF_SCHEDULING_TRIAL=" + name, flush=True)
    process = subprocess.Popen(command)
    state["process"] = process
    if state["cancelled"]:
        process.send_signal(state["cancelled"])
    try:
        returncode = process.wait()
    finally:
        state["process"] = None
    if state["cancelled"]:
        raise SystemExit(128 + state["cancelled"])
    path = root / "trials" / name
    if returncode:
        log = path / "ctest.log"
        if log.exists():
            print(log.read_text(), flush=True)
        raise subprocess.CalledProcessError(returncode, command)
    result = json.loads((path / "summary.json").read_text())
    assert (
        result["returncode"] == 0
        and not result["failures"]
        and not result.get("gtest_failures")
    )
    registered = Counter(
        (
            case.get("classname"),
            case.get("name"),
            case.get("status") == "notrun",
        )
        for report in (path / "gtest").glob("*.xml")
        for case in ET.parse(report).findall(".//testcase")
    )
    if state["registered"] is None:
        state["registered"] = registered
        (root / "registered-oracle.json").write_text(
            json.dumps(
                [
                    dict(suite=key[0], name=key[1], notrun=key[2], count=count)
                    for key, count in sorted(registered.items())
                ],
                indent=2,
            )
        )
    assert registered == state["registered"], (
        "Registered/disabled GTest inventory changed"
    )
    result["registered_cases"] = sum(registered.values())
    result["disabled_cases"] = sum(
        count for key, count in registered.items() if key[2]
    )
    return result


def binary_identity():
    files = set()
    for test in [
        *metadata["tests"],
        *matrices["baseline"]["tests"],
        *matrices["candidate"]["tests"],
    ]:
        if test["name"] == "generate_resource_spec":
            continue
        working_directory = Path(
            next(
                prop["value"]
                for prop in test["properties"]
                if prop["name"] == "WORKING_DIRECTORY"
            )
        )
        executable = next(
            (
                argument.split("=", 1)[1]
                for argument in test["command"]
                if argument.startswith("-Dcommand_to_run=")
            ),
            test["command"][0],
        )
        if executable == "bash":
            arguments = next(
                argument.split("=", 1)[1]
                for argument in test["command"]
                if argument.startswith("-Dcommand_args=")
            ).split(";")
            assert arguments[0] == "./run_cudf_test.sh" and len(arguments) >= 3
            files.add((working_directory / arguments[0]).resolve())
            executable = arguments[1]
            if arguments[2] != "-":
                checker = Path(arguments[2])
                if not checker.is_absolute():
                    checker = working_directory / checker
                files.add(checker.resolve())
        path = Path(executable)
        if not path.is_absolute():
            path = working_directory / path
        files.add(path.resolve())
        for prop in test["properties"]:
            if prop["name"] == "ENVIRONMENT":
                for entry in prop["value"]:
                    if entry.startswith("LD_PRELOAD="):
                        for name in entry.split("=", 1)[1].split(":"):
                            path = Path(name)
                            if not path.is_absolute():
                                path = working_directory / path
                            files.add(path.resolve())
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        files.update(
            path.resolve()
            for path in (Path(prefix) / "lib").glob("libcudf.so*")
        )
    return {
        str(path): dict(bytes=path.stat().st_size, sha256=digest(path))
        for path in sorted(files)
    }


def cache_identity():
    entries = []
    for name in ["cache-seed", "driver-cache-seed"]:
        for path in sorted((root / name).rglob("*")):
            if path.is_file():
                entries.append(
                    dict(
                        path=str(path.relative_to(root)),
                        bytes=path.stat().st_size,
                        sha256=digest(path),
                    )
                )
    encoded = json.dumps(entries, sort_keys=True).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), files=entries)


def export():
    # Compiled cache archives are intentionally private to this job, not uploaded artifacts.
    shutil.copytree(
        root,
        args.output,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "cache-seed",
            "driver-cache-seed",
            "initial-cache-seed",
            "initial-driver-cache-seed",
            "restored-original-cache",
            "interrupted-canonical-cache",
            "*-after",
            "jit-cache",
            "driver-cache",
            "measurement.lock",
        ),
    )


try:
    binaries = binary_identity()
    (root / "binary-identities.json").write_text(
        json.dumps(binaries, indent=2)
    )
    (root / "protocol.json").write_text(
        json.dumps(
            dict(
                regime=args.regime,
                pairs=args.pairs,
                effective_host_workers=workers,
                visible_cpu_count=os.cpu_count(),
                affinity=sorted(os.sched_getaffinity(0)),
                caveat="Initial and warm cache regimes are distinct; no cross-regime speedup claim.",
            ),
            indent=2,
        )
    )
    if original_cache:
        if original_cache.exists():
            original_cache.rename(cache_backup)
        cache_taken = True
    seed(cache_backup if cache_backup.exists() else None, root / "cache-seed")
    seed(
        Path(os.environ["CUDA_CACHE_PATH"]).resolve()
        if os.environ.get("CUDA_CACHE_PATH")
        else None,
        root / "driver-cache-seed",
    )
    warmup = run("warmup", "baseline")
    outcomes = root / "trials/warmup/outcomes.json"
    shutil.copy2(outcomes, root / "baseline-outcomes.json")
    if args.regime == "warm":
        for name in ["cache-seed", "driver-cache-seed"]:
            (root / name).rename(root / ("initial-" + name))
        shutil.copytree(
            root / "trials/warmup/jit-cache-after", root / "cache-seed"
        )
        shutil.copytree(
            root / "trials/warmup/driver-cache-after",
            root / "driver-cache-seed",
        )
    identity = cache_identity()
    (root / "cache-seed-manifest.json").write_text(
        json.dumps(identity, indent=2)
    )
    records = []
    for index in range(args.pairs):
        pair = {}
        for kind in (
            ["baseline", "candidate"]
            if index % 2 == 0
            else ["candidate", "baseline"]
        ):
            result = run(f"pair-{index + 1}-{kind}", kind)
            assert result["coverage_matches"] is True
            pair[kind] = result
        pair["saved_seconds"] = (
            pair["baseline"]["seconds"] - pair["candidate"]["seconds"]
        )
        records.append(pair)
        (root / "pairs.json").write_text(json.dumps(records, indent=2))
    assert cache_identity() == identity, "The frozen CI cache seed changed"
    assert binary_identity() == binaries, (
        "Installed binaries changed during comparison"
    )
    differences = [pair["saved_seconds"] for pair in records]
    mean = statistics.mean(differences)
    margin = (
        4.302652729911275 * statistics.stdev(differences) / math.sqrt(3)
        if len(records) == 3
        else None
    )
    baseline = statistics.median(
        pair["baseline"]["seconds"] for pair in records
    )
    candidate_time = statistics.median(
        pair["candidate"]["seconds"] for pair in records
    )
    summary = dict(
        regime=args.regime,
        pairs=records,
        warmup=warmup,
        mean_saved_seconds=mean,
        paired_t_95_ci_seconds=[mean - margin, mean + margin]
        if margin is not None
        else None,
        baseline_median_seconds=baseline,
        candidate_median_seconds=candidate_time,
        improvement_percent=100 * (baseline - candidate_time) / baseline,
        regression_over_three_percent=candidate_time > baseline * 1.03,
        cache_seed_sha256=identity["sha256"],
    )
    (root / "comparison.json").write_text(json.dumps(summary, indent=2))
    print("CUDF_SCHEDULING_COMPARISON=" + json.dumps(summary), flush=True)
    report_dir = os.environ.get("RAPIDS_TESTS_DIR")
    if report_dir:
        last_candidate = root / "trials" / f"pair-{args.pairs}-candidate/gtest"
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        for report in last_candidate.glob("*.xml"):
            shutil.copy2(report, Path(report_dir) / report.name)
finally:
    if cache_taken:
        if original_cache.exists():
            original_cache.rename(root / "interrupted-canonical-cache")
        if cache_backup.exists():
            cache_backup.rename(original_cache)
    export()
