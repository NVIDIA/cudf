# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partition a resolved PR matrix between internal and upstream test suites."""

import json
import os
import subprocess
from pathlib import Path

# https://github.com/NVIDIA/cudf/issues/23498
SUPPORTED_GPUS = 'map(select(.GPU != "gb300" and .GPU != "gh200"))'
LATEST_PER_MAJOR = (
    'group_by(.CUDA_VER|split(".")|map(tonumber)|.[0]) | '
    'map(max_by([(.PY_VER|split(".")|map(tonumber)), '
    '(.CUDA_VER|split(".")|map(tonumber))]))'
)
# These are the pre-existing job selectors, applied before coverage is traded.
JOB_FILTERS = {
    "wheel-tests-cudf": ("wheels", "pandas", SUPPORTED_GPUS),
    "conda-python-cudf-tests": ("conda", "pandas", SUPPORTED_GPUS),
    "unit-tests-cudf-pandas": (
        "wheels",
        "pandas",
        'group_by([.ARCH, (.CUDA_VER|split(".")|map(tonumber)|.[0])]) | '
        'map(max_by([(.PY_VER|split(".")|map(tonumber)), '
        '(.CUDA_VER|split(".")|map(tonumber))])) | ' + SUPPORTED_GPUS,
    ),
    "wheel-tests-cudf-polars": (
        "wheels",
        "polars",
        'map(select(.ARCH == "amd64")) | ' + LATEST_PER_MAJOR,
    ),
    "conda-python-cudf-polars-tests": ("conda", "polars", SUPPORTED_GPUS),
}


def version(value):
    return tuple(int(component) for component in value.split("."))


def select_existing(matrix, expression):
    result = subprocess.run(
        ["jq", "-ce", ".include | " + expression],
        input=json.dumps(matrix),
        text=True,
        capture_output=True,
        check=True,
    )
    entries = json.loads(result.stdout)
    if not entries:
        raise ValueError("Existing job selector produced an empty matrix")
    return entries


def coverage_filter(expression, cuda_major, enabled):
    if not enabled:
        return expression
    # Consumers may resolve newer definitions: enforce safeguards on their matrix,
    # rather than freezing the entries observed by the planning job.
    return (
        f"({expression}) as $original | "
        '($original | map(select(.ARCH != "amd64" or '
        '.DEPENDENCIES == "oldest" or '
        f'(.CUDA_VER|split(".")|map(tonumber)|.[0]) != {cuda_major}))) '
        'as $retained | if any($retained[]; .ARCH == "amd64") '
        "then $retained else $original end"
    )


def plan_coverage(wheels, conda, *, run_pandas, run_polars, rapids_version):
    candidates = [
        entry
        for entry in wheels["include"]
        if entry["ARCH"] == "amd64"
        and entry["DRIVER"] == "latest"
        and entry["DEPENDENCIES"] == "latest"
    ]
    if not candidates:
        raise ValueError(
            "No amd64/latest-driver/latest-dependencies wheel environment "
            "is available for upstream tests"
        )
    selected = max(
        candidates,
        key=lambda entry: (
            version(entry["CUDA_VER"]),
            version(entry["PY_VER"]),
            json.dumps(entry, sort_keys=True),
        ),
    )
    outputs = {
        "run-pandas": str(run_pandas).lower(),
        "run-polars": str(run_polars).lower(),
        "upstream-container": (
            f"rapidsai/citestwheel:{rapids_version}"
            f"-cuda{selected['CUDA_VER']}-{selected['LINUX_VER']}"
            f"-py{selected['PY_VER']}"
        ),
    }
    matrices = {"wheels": wheels, "conda": conda}
    enabled = {"pandas": run_pandas, "polars": run_polars}
    decisions = {}
    for job, (source, suite, expression) in JOB_FILTERS.items():
        entries = select_existing(matrices[source], expression)
        expression = coverage_filter(
            expression, version(selected["CUDA_VER"])[0], enabled[suite]
        )
        retained = select_existing(matrices[source], expression)
        removed = [entry for entry in entries if entry not in retained]
        outputs[job] = expression
        decisions[job] = {"retained": retained, "removed": removed}
    return outputs, selected, decisions


def summary(outputs, selected, decisions):
    lines = [
        "## PR Python CUDA coverage",
        "",
        f"Upstream environment: `{outputs['upstream-container']}`",
        "",
        f"CUDA: {selected['CUDA_VER']}; Python: {selected['PY_VER']}; "
        "amd64, RTX Pro, latest dependencies.",
        "",
        f"Upstream pandas scheduled: {outputs['run-pandas']}; "
        f"upstream Polars scheduled: {outputs['run-polars']}.",
        "",
        "Oldest-dependency and ARM entries remain internal; nightly is unchanged.",
        "",
        "Retained/removed entries below preview the matrices resolved for planning. "
        "Test jobs independently resolve shared-workflows@main and apply the same "
        "policy; upstream matrix updates may change their final entries.",
    ]
    for job, decision in decisions.items():
        lines.extend(["", f"### {job}", ""])
        for disposition, entries in decision.items():
            lines.append(f"{disposition.title()} ({len(entries)}):")
            lines.extend(
                f"- `{json.dumps(entry, sort_keys=True)}`" for entry in entries
            )
    return "\n".join(lines) + "\n"


def main():
    outputs, selected, decisions = plan_coverage(
        json.loads(os.environ["WHEELS_MATRIX"]),
        json.loads(os.environ["CONDA_MATRIX"]),
        run_pandas=os.environ["RUN_PANDAS"] == "true",
        run_polars=os.environ["RUN_POLARS"] == "true",
        rapids_version=".".join(
            Path("VERSION").read_text().strip().split(".")[:2]
        ),
    )
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        for key, value in outputs.items():
            print(f"{key}={value}", file=output)
    report = summary(outputs, selected, decisions)
    print(report)
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as step_summary:
        step_summary.write(report)


if __name__ == "__main__":
    main()
