# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the issue event entry point without sending Slack messages."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import call, patch

from triage import classify, main

REPOSITORY = "NVIDIA/cudf"


class IssueEventTests(unittest.TestCase):
    def run_event(
        self, repository, issue, *, roster_error=None, linked_responses=()
    ):
        with tempfile.TemporaryDirectory() as tmp:
            event = Path(tmp) / "event.json"
            event.write_text(
                json.dumps(
                    {
                        "action": "opened",
                        "repository": {"full_name": repository},
                        "issue": {"number": 42, "body": "<!channel>"},
                    }
                )
            )
            with ExitStack() as stack:
                stack.enter_context(
                    patch("sys.argv", ["triage.py", "--event", str(event)])
                )
                github = stack.enter_context(patch("triage.GitHub"))
                stack.enter_context(
                    patch("triage.slack_token", return_value="test-token")
                )
                post = stack.enter_context(patch("triage.post"))
                get = github.return_value.get
                get.side_effect = [
                    issue,
                    roster_error
                    or [
                        {"login": "maintainer", "type": "User"},
                        {"login": "reporter", "type": "User"},
                    ],
                    *linked_responses,
                ]
                main()
                return get, post

    def test_event_only_sends_to_primary_contact_and_uses_plain_text(self):
        get, post = self.run_event(
            REPOSITORY,
            {
                "number": 42,
                "html_url": f"https://github.com/{REPOSITORY}/issues/42",
                "state": "open",
                "user": {"login": "reporter"},
                "title": "<!here>",
                "body": "import cudf",
                "labels": [],
            },
        )
        self.assertEqual(
            get.call_args_list,
            [
                call(f"repos/{REPOSITORY}/issues/42", timeout=30),
                call(
                    "orgs/NVIDIA/teams/cudf-python-codeowners/members?per_page=100",
                    paginate=True,
                    timeout=30,
                ),
            ],
        )
        post.assert_called_once()
        token, message = post.call_args.args
        self.assertEqual(token, "test-token")
        self.assertEqual(message["channel"], "U01PXPJ5741")
        self.assertFalse(message["mrkdwn"])
        self.assertFalse(message["link_names"])
        self.assertIn("Proposed label: Python", message["text"])
        self.assertIn("Suggested first contact: maintainer", message["text"])
        self.assertNotIn("<!", message["text"])

    def test_roster_timeout_still_sends_component_preview(self):
        _, post = self.run_event(
            REPOSITORY,
            {
                "number": 42,
                "state": "open",
                "user": {"login": "reporter"},
                "title": "import cudf fails",
                "labels": [],
            },
            roster_error=TimeoutExpired("gh", 30),
        )
        post.assert_called_once()
        _, message = post.call_args.args
        self.assertEqual(message["channel"], "U01PXPJ5741")
        self.assertIn("Proposed label: Python", message["text"])
        self.assertIn("Suggested first contact: None", message["text"])
        self.assertIn(
            "Unable to read the codeowner team roster", message["text"]
        )

    def test_labeled_and_assigned_issue_skips_dm_and_roster_lookup(self):
        get, post = self.run_event(
            REPOSITORY,
            {
                "number": 42,
                "state": "open",
                "labels": [{"name": "Python"}],
                "assignees": [{"login": "maintainer"}],
            },
        )
        get.assert_called_once()
        post.assert_not_called()

    def test_linked_pr_reviewer_beats_rotation_and_excludes_reporter(self):
        _, post = self.run_event(
            REPOSITORY,
            {
                "number": 42,
                "html_url": f"https://github.com/{REPOSITORY}/issues/42",
                "state": "open",
                "user": {"login": "reporter"},
                "title": "import cudf fails",
                "body": f"Follow-up to https://github.com/{REPOSITORY}/pull/123",
                "labels": [],
            },
            linked_responses=(
                {"user": {"login": "reporter"}},
                [{"state": "APPROVED", "user": {"login": "maintainer"}}],
            ),
        )
        post.assert_called_once()
        _, message = post.call_args.args
        self.assertIn("Suggested first contact: maintainer", message["text"])
        self.assertIn("Eligible reviewer on linked PR #123", message["text"])

    def test_issue_closed_before_fetch_does_not_send(self):
        _, post = self.run_event(REPOSITORY, {"state": "closed"})
        post.assert_not_called()


class ComponentEvidenceTests(unittest.TestCase):
    def test_java_api_title_beats_native_implementation_reference(self):
        components, evidence = classify(
            {
                "labels": [],
                "title": "Java API for multi-column transform",
                "body": "The underlying implementation uses cudf::transform.",
            },
            {"Java": "java", "libcudf": "cpp"},
        )
        self.assertEqual(components, ["Java"])
        self.assertEqual(evidence["signal_source"], "title")

    def test_build_problem_beats_signals_in_reproduction_details(self):
        components, evidence = classify(
            {
                "labels": [],
                "title": "Build fails on aarch64",
                "body": (
                    "## Describe the bug\nBuilding libcudf fails when linking.\n"
                    "## Steps/Code to reproduce bug\n"
                    "```text\nUpdate cudf-polars test run list\nCMakeLists.txt\n```"
                ),
            },
            {"cudf-polars": "polars", "libcudf": "cpp", "CMake": "cmake"},
        )
        self.assertEqual(components, ["libcudf"])
        self.assertEqual(
            evidence["signal_source"], "problem or requested behavior"
        )
