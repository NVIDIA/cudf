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

from triage import main

REPOSITORY = "NVIDIA/cudf"


class IssueEventTests(unittest.TestCase):
    def run_event(self, repository, issue, *, roster_error=None):
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
                    roster_error or [{"login": "maintainer", "type": "User"}],
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
                call(f"repos/{REPOSITORY}/issues/42"),
                call(
                    "orgs/NVIDIA/teams/cudf-python-codeowners/members?per_page=100",
                    paginate=True,
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
            roster_error=TimeoutExpired("gh", 180),
        )
        post.assert_called_once()
        _, message = post.call_args.args
        self.assertEqual(message["channel"], "U01PXPJ5741")
        self.assertIn("Proposed label: Python", message["text"])
        self.assertIn("Suggested first contact: None", message["text"])
        self.assertIn(
            "Unable to read the codeowner team roster", message["text"]
        )

    def test_issue_closed_before_fetch_does_not_send(self):
        _, post = self.run_event(REPOSITORY, {"state": "closed"})
        post.assert_not_called()
