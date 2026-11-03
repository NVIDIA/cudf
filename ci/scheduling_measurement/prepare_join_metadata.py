# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select unsplit/split discovery from one installed measurement build."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("metadata", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
original = json.loads(args.metadata.read_text())
tests = {test["name"]: test for test in original["tests"]}
assert len(tests) == len(original["tests"]), "Duplicate CTest names"
required = {
    "JOIN_TEST",
    "MIXED_JOIN_TEST",
    "JOIN_UNSPLIT_BASELINE_TEST",
    "DATA_CHUNK_SOURCE_TEST",
}
assert required <= tests.keys(), (
    "The measurement build needs both combined and split JOIN targets"
)
matrices = {
    kind: dict(deepcopy(original), tests=[])
    for kind in ["baseline", "candidate"]
}
for test in original["tests"]:
    name = test["name"]
    if name == "JOIN_UNSPLIT_BASELINE_TEST":
        continue
    matrices["candidate"]["tests"].append(deepcopy(test))
    if name == "MIXED_JOIN_TEST":
        continue
    if name == "JOIN_TEST":
        baseline = deepcopy(tests["JOIN_UNSPLIT_BASELINE_TEST"])
        baseline["name"] = "JOIN_TEST"
        baseline["properties"] = [
            prop
            for prop in baseline["properties"]
            if prop["name"] not in ["DISABLED", "COST"]
        ]
    else:
        baseline = deepcopy(test)
    command = baseline.get("command", [])
    launch_index = next(
        (
            i
            for i, value in enumerate(command)
            if value.startswith("-Dcommand_to_run=")
        ),
        None,
    )
    if (
        launch_index is not None
        and command[launch_index] == "-Dcommand_to_run=bash"
    ):
        argument_index = next(
            i
            for i, value in enumerate(command)
            if value.startswith("-Dcommand_args=")
        )
        arguments = command[argument_index].split("=", 1)[1].split(";")
        assert arguments[0] == "./run_cudf_test.sh" and len(arguments) >= 3
        command[launch_index] = "-Dcommand_to_run=" + arguments[1]
        command[argument_index] = "-Dcommand_args=" + ";".join(arguments[3:])
        if arguments[2] != "-":
            environment = next(
                (
                    prop
                    for prop in baseline["properties"]
                    if prop["name"] == "ENVIRONMENT"
                ),
                None,
            )
            if environment is None:
                environment = dict(name="ENVIRONMENT", value=[])
                baseline["properties"].append(environment)
            environment["value"].append("LD_PRELOAD=" + arguments[2])
    if name == "DATA_CHUNK_SOURCE_TEST":
        groups = next(
            prop["value"]
            for prop in baseline["properties"]
            if prop["name"] == "RESOURCE_GROUPS"
        )
        for group in groups:
            for requirement in group["requirements"]:
                if requirement[".type"] == "gpus":
                    requirement["slots"] = 100
    matrices["baseline"]["tests"].append(baseline)
assert len(matrices["baseline"]["tests"]) == len(original["tests"]) - 2
assert len(matrices["candidate"]["tests"]) == len(original["tests"]) - 1
args.output.mkdir(parents=True, exist_ok=True)
for kind, metadata in matrices.items():
    destination = args.output / (kind + "-tests.json")
    assert not destination.exists()
    destination.write_text(json.dumps(metadata, indent=2))
