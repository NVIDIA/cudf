# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level registration contracts; run with python3 test_registration.py.

NVBench's state_iterator advances the first registered axis fastest. The local
fixture cache retains only one scale, so scale_factor must be registered last.
Inspect the real registrations rather than simulating an independent axis loop.
No GPU, NVBench build, or third-party Python packages are needed for this test.
"""

import re
import unittest
from pathlib import Path


QUERIES = (1, 5, 6, 9, 10)


def registrations(query):
    source = Path(__file__).with_name(f"q{query:02}.cpp").read_text()
    return {
        name: re.sub(r"\s+", "", body)
        for name, body in re.findall(
            r"NVBENCH_BENCH\((\w+)\)([^;]+);", source
        )
    }


def registration(name, axes):
    return re.sub(r"\s+", "", f'.set_name("{name}")' + "".join(axes))


class RegistrationTest(unittest.TestCase):
    def test_local_scale_last_and_defaults_unchanged(self):
        for query in QUERIES:
            with self.subTest(query=query):
                name = f"ndsh_q{query}_local"
                axes = [
                    '.add_string_axis("format", {"parquet", "vortex"})',
                    f'.add_string_axis("workload", {{"read", "q{query}"}})',
                ]
                if query == 9:
                    axes.append(
                        '.add_string_axis("engine", {"binaryop", "ast", "transform"})'
                    )
                axes.extend(
                    [
                        '.add_string_axis("cache", {"warm", "cold"})',
                        '.add_string_axis("io", {"buffered"})',
                        '.add_float64_axis("scale_factor", {0.01, 0.1, 1, 10})',
                    ]
                )
                self.assertEqual(
                    registrations(query)[name], registration(name, axes)
                )

    def test_legacy_registrations_unchanged(self):
        for query in QUERIES:
            with self.subTest(query=query):
                axes = []
                if query == 1:
                    axes.append('.add_string_axis("filename", {""})')
                axes.append('.add_float64_axis("scale_factor", {0.01, 0.1, 1})')
                suffixes = ("",)
                if query == 9:
                    axes.append(
                        '.add_string_axis("engine", {"binaryop", "ast", "transform"})'
                    )
                    suffixes = ("", "_noio", "_amount")
                actual = registrations(query)
                del actual[f"ndsh_q{query}_local"]
                expected = {
                    f"ndsh_q{query}{suffix}": registration(
                        f"ndsh_q{query}{suffix}", axes
                    )
                    for suffix in suffixes
                }
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
