# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check pagination and sampling without GitHub access."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from collect import GitHub, collect_item, sample


class CollectorTests(unittest.TestCase):
    def test_paginated_arrays_and_nested_check_runs(self):
        for pages, key in [
            ([[{"number": 1}], [{"number": 2}]], None),
            ([{"check_runs": [1]}, {"check_runs": [2]}], "check_runs"),
        ]:
            with self.subTest(key=key), patch("collect.subprocess.run") as run:
                run.return_value = SimpleNamespace(
                    returncode=0, stdout=json.dumps(pages)
                )
                rows = GitHub().get("endpoint", paginate=True, key=key)
                self.assertEqual(len(rows), 2)
                command = run.call_args[0][0]
                self.assertIn("--paginate", command)
                self.assertIn("--slurp", command)
                self.assertEqual(command[3], "GET")

    def test_api_failure_does_not_turn_into_empty_results(self):
        with patch("collect.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=1, stderr="HTTP 401: Bad credentials"
            )
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                GitHub().get("endpoint", paginate=True)

    def test_sample_deduplicates_and_fills_short_strata(self):
        class FakeGitHub:
            def search(self, repository, query, count):
                numbers = [1, 2, 3] if query == "is:pr" else [1]
                return [{"number": number} for number in numbers]

        numbers, queries = sample(FakeGitHub(), "NVIDIA/cudf", 3, "triage")
        self.assertEqual(numbers, [1, 2, 3])
        self.assertEqual(len(queries), 5)

    def test_incomplete_search_is_rejected(self):
        with patch.object(GitHub, "get") as get:
            get.return_value = {"incomplete_results": True, "items": []}
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                GitHub().search("NVIDIA/cudf", "is:pr", 5)

    def test_scope_label_applies_to_every_sample_query_and_fallback(self):
        github = Mock()
        github.search.return_value = []
        sample(github, "NVIDIA/cudf", 25, "triage", "Python")
        queries = [call[0][1] for call in github.search.call_args_list]
        self.assertEqual(len(queries), 6)
        self.assertEqual(queries[-1], 'is:pr label:"Python"')
        for query in queries[:5]:
            self.assertIn('label:"Python"', query)

    def test_graphql_errors_are_not_treated_as_success(self):
        with patch("collect.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=0, stdout='{"data": null, "errors": ["denied"]}'
            )
            with self.assertRaisesRegex(RuntimeError, "denied"):
                GitHub().graphql("query { viewer { login } }")

    def test_collector_includes_paginated_inline_replies(self):
        github = GitHub()
        issue = {"number": 1, "pull_request": {}}
        pr = {"head": {"sha": "head"}}
        reply = {"user": {"login": "author"}}

        def response(endpoint, **kwargs):
            if endpoint == "repos/NVIDIA/cudf/issues/1":
                return issue
            if endpoint == "repos/NVIDIA/cudf/pulls/1":
                return pr
            if endpoint == "repos/NVIDIA/cudf/commits/head":
                return {"author": {"login": "author"}}
            if endpoint == "repos/NVIDIA/cudf/pulls/1/comments?per_page=100":
                self.assertTrue(kwargs["paginate"])
                return [reply]
            return []

        with patch.object(github, "get", side_effect=response):
            result = collect_item(github, "NVIDIA/cudf", 1)
        self.assertEqual(result["review_comments"], [reply])
        self.assertEqual(result["head_commit_author"], "author")
        self.assertEqual(result["warnings"], [])


if __name__ == "__main__":
    unittest.main()
