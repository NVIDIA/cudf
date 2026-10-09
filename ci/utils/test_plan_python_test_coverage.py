# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import unittest
from pathlib import Path

from plan_python_test_coverage import (
    JOB_FILTERS,
    plan_coverage,
    select_existing,
)

RAPIDS_VERSION = ".".join(
    (Path(__file__).resolve().parents[2] / "VERSION")
    .read_text()
    .strip()
    .split(".")[:2]
)


def entry(cuda, *, arch="amd64", py="3.14", deps="latest", driver="latest"):
    return {
        "ARCH": arch,
        "CUDA_VER": cuda,
        "PY_VER": py,
        "LINUX_VER": "ubuntu24.04",
        "GPU": "rtxpro6000" if arch == "amd64" else "l4",
        "DRIVER": driver,
        "DEPENDENCIES": deps,
    }


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.wheels = {
            "include": [
                entry("12.9.2", py="3.12", deps="oldest"),
                entry("13.3.0"),
                entry("13.3.0", arch="arm64"),
            ]
        }
        self.conda = {
            "include": [
                entry("12.2.2", py="3.12", deps="oldest", driver="earliest"),
                entry("13.0.3"),
                entry("13.3.0", arch="arm64", py="3.13"),
            ]
        }

    def plan(self, **overrides):
        return plan_coverage(
            self.wheels,
            self.conda,
            rapids_version=RAPIDS_VERSION,
            **{"run_pandas": True, "run_polars": True, **overrides},
        )

    def consumer_entries(self, job, outputs):
        source = JOB_FILTERS[job][0]
        matrix = self.wheels if source == "wheels" else self.conda
        return select_existing(matrix, outputs[job])

    def test_complement_preserves_oldest_and_arm(self):
        outputs, selected, decisions = self.plan()
        self.assertEqual(selected["CUDA_VER"], "13.3.0")
        self.assertEqual(
            outputs["upstream-container"],
            f"rapidsai/citestwheel:{RAPIDS_VERSION}-cuda13.3.0-ubuntu24.04-py3.14",
        )
        for job, decision in decisions.items():
            with self.subTest(job=job):
                self.assertEqual(len(decision["removed"]), 1)
                self.assertTrue(
                    all(
                        e["ARCH"] == "amd64" and e["DEPENDENCIES"] == "latest"
                        for e in decision["removed"]
                    )
                )
                self.assertTrue(
                    any(
                        e["DEPENDENCIES"] == "oldest"
                        for e in decision["retained"]
                    )
                )
                self.assertEqual(
                    self.consumer_entries(job, outputs), decision["retained"]
                )
        self.assertEqual(len(decisions["wheel-tests-cudf"]["retained"]), 2)

    def test_absent_upstream_suite_keeps_original_selector(self):
        for pandas, polars in ((False, False), (True, False), (False, True)):
            outputs, _, decisions = self.plan(
                run_pandas=pandas, run_polars=polars
            )
            for job, (source, suite, expression) in JOB_FILTERS.items():
                enabled = pandas if suite == "pandas" else polars
                if not enabled:
                    original = select_existing(
                        self.wheels if source == "wheels" else self.conda,
                        expression,
                    )
                    self.assertEqual(
                        self.consumer_entries(job, outputs), original
                    )
                    self.assertEqual(decisions[job]["removed"], [])

    def test_added_major_numeric_selection_and_python_tie(self):
        self.wheels["include"] += [
            entry("14.1.0", py="3.9"),
            entry("14.1.0", py="3.14"),
        ]
        outputs, selected, decisions = self.plan()
        self.assertEqual(selected["CUDA_VER"], "14.1.0")
        self.assertEqual(selected["PY_VER"], "3.14")
        self.assertIn("cuda14.1.0", outputs["upstream-container"])
        self.assertEqual(decisions["conda-python-cudf-tests"]["removed"], [])
        self.assertTrue(
            any(
                e["CUDA_VER"] == "13.3.0"
                for e in decisions["wheel-tests-cudf"]["retained"]
            )
        )

    def test_minor_bump_and_reordering(self):
        self.wheels["include"] += [entry("13.10.0"), entry("13.9.0")]
        expected = self.plan()
        self.wheels["include"].reverse()
        actual = self.plan()
        self.assertEqual(expected[1], actual[1])
        self.assertEqual(actual[1]["CUDA_VER"], "13.10.0")
        for decision in actual[2].values():
            self.assertTrue(
                all(
                    e["CUDA_VER"].startswith("13.")
                    for e in decision["removed"]
                )
            )

    def test_removed_major_and_single_major_fallback(self):
        self.wheels["include"] = [
            entry("12.9.2"),
            entry("12.9.2", arch="arm64"),
        ]
        outputs, selected, decisions = self.plan()
        self.assertEqual(selected["CUDA_VER"], "12.9.2")
        for job in (
            "wheel-tests-cudf",
            "wheel-tests-cudf-polars",
            "unit-tests-cudf-pandas",
        ):
            self.assertEqual(decisions[job]["removed"], [])
            self.assertTrue(
                any(
                    e["ARCH"] == "amd64"
                    for e in self.consumer_entries(job, outputs)
                )
            )

    def test_oldest_on_selected_major_is_not_removed(self):
        self.wheels["include"].append(
            entry("13.3.0", deps="oldest", py="3.12")
        )
        _, _, decisions = self.plan()
        retained = decisions["wheel-tests-cudf"]["retained"]
        self.assertIn(self.wheels["include"][-1], retained)

    def test_no_suitable_upstream_environment(self):
        self.wheels["include"] = [
            entry("13.3.0", deps="oldest"),
            entry("13.3.0", arch="arm64"),
        ]
        with self.assertRaisesRegex(ValueError, "No amd64"):
            self.plan()

    def test_inputs_are_not_modified(self):
        original = copy.deepcopy((self.wheels, self.conda))
        self.plan()
        self.assertEqual((self.wheels, self.conda), original)

    def test_consumers_apply_policy_to_updated_matrices(self):
        outputs, _, _ = self.plan()
        for job, (source, _, expression) in JOB_FILTERS.items():
            with self.subTest(job=job):
                matrix = self.wheels if source == "wheels" else self.conda
                changed = copy.deepcopy(matrix)
                changed["include"] += [
                    entry("14.1.0"),
                    entry("13.10.0"),
                    entry("13.10.0", arch="arm64"),
                    entry("13.10.0", deps="oldest", py="3.12"),
                ]
                original = select_existing(changed, expression)
                retained = select_existing(changed, outputs[job])
                self.assertIn(entry("14.1.0"), retained)
                for e in original:
                    if e["ARCH"] != "amd64" or e["DEPENDENCIES"] == "oldest":
                        self.assertIn(e, retained)
                    elif e["CUDA_VER"].startswith("13."):
                        self.assertNotIn(e, retained)

    def test_updated_single_major_matrix_falls_back_at_execution(self):
        outputs, _, _ = self.plan()
        changed = {"include": [entry("13.9.0"), entry("13.9.0", arch="arm64")]}
        for job, (_, _, expression) in JOB_FILTERS.items():
            with self.subTest(job=job):
                self.assertEqual(
                    select_existing(changed, outputs[job]),
                    select_existing(changed, expression),
                )

    def test_absent_upstream_suite_keeps_updated_matrix(self):
        outputs, _, _ = self.plan(run_pandas=False, run_polars=False)
        changed = {"include": [entry("13.9.0"), entry("14.1.0")]}
        for job, (_, _, expression) in JOB_FILTERS.items():
            with self.subTest(job=job):
                self.assertEqual(
                    select_existing(changed, outputs[job]),
                    select_existing(changed, expression),
                )


if __name__ == "__main__":
    unittest.main()
