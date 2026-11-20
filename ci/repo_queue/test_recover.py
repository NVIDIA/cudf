# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify manual recovery preserves history and never posts another Slack message."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from actions import preview
from recover import inspect_pending, main, message_receipt, reconcile_pending
from sync import ProjectCards
from test_actions import CONFIG, python_item, snapshot
from test_report import NOW, POLICY, request
from test_sync import ROUTE, STATE, FakeGitHub


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.github = FakeGitHub()
        self.config = deepcopy(CONFIG)
        data = python_item()
        request(data, "reviewer")
        self.cards = ProjectCards(self.github, self.config, STATE, True)
        plan = preview(snapshot(data), POLICY, self.config)
        self.cards.reconcile(plan, snapshot(data))
        self.key = plan["actions"][0]["key"]
        self.pending = {
            "id": "pending-test",
            "kind": "reminder",
            "prepared_at": NOW,
        }
        data = deepcopy(self.cards.cards[self.key]["data"])
        self.history = {
            "first_notified_at": NOW,
            "last_notified_at": NOW,
            "thread_ts": "1794830400.000001",
            "channel": "D123",
        }
        data["deliveries"][ROUTE] = {**self.history, "pending": self.pending}
        self.cards.save(self.key, data, {})
        self.receipt = message_receipt(
            self.config["slack"]["workspace_url"]
            + "/archives/D123/p1794916800123456",
            self.config["slack"]["workspace_url"],
        )
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.journal = Path(self.directory.name) / "dm-receipts.json"
        post = patch(
            "track.post",
            side_effect=AssertionError("Recovery must never post to Slack"),
        )
        post.start()
        self.addCleanup(post.stop)

    def test_inspection_and_preview_do_not_mutate_project_or_journal(self):
        count = len(self.github.mutations)
        info = inspect_pending(self.cards, self.key)
        self.assertEqual(info["pending"]["id"], self.pending["id"])
        for receipt in (None, self.receipt):
            result = reconcile_pending(
                self.cards, self.key, self.pending["id"], receipt, self.journal
            )
            self.assertFalse(result["applied"])
        self.assertFalse(self.journal.exists())
        self.assertEqual(len(self.github.mutations), count)

    def test_confirmed_non_send_removes_only_pending_intent(self):
        action = deepcopy(self.cards.cards[self.key]["data"]["action"])
        reconcile_pending(
            self.cards, self.key, self.pending["id"], None, self.journal, True
        )
        self.assertEqual(
            self.cards.cards[self.key]["data"]["deliveries"][ROUTE],
            self.history,
        )
        self.assertEqual(self.cards.cards[self.key]["data"]["action"], action)

    def test_confirmed_send_keeps_existing_history_and_thread(self):
        reconcile_pending(
            self.cards,
            self.key,
            self.pending["id"],
            self.receipt,
            self.journal,
            True,
        )
        record = self.cards.cards[self.key]["data"]["deliveries"][ROUTE]
        self.assertNotIn("pending", record)
        self.assertEqual(record["first_notified_at"], NOW)
        self.assertEqual(record["thread_ts"], self.history["thread_ts"])
        self.assertEqual(record["last_notified_at"], self.receipt["sent_at"])
        self.assertTrue(
            self.cards.cards[self.key]["values"]["Slack thread"].endswith(
                "p1794830400000001"
            )
        )
        self.assertEqual(
            json.loads(self.journal.read_text())[self.pending["id"]],
            self.receipt,
        )

    def test_receipt_survives_failed_acknowledgement_and_blocks_contradiction(
        self,
    ):
        with patch(
            "recover.acknowledge_with_retry",
            side_effect=RuntimeError("Project unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                reconcile_pending(
                    self.cards,
                    self.key,
                    self.pending["id"],
                    self.receipt,
                    self.journal,
                    True,
                )
        self.assertEqual(
            json.loads(self.journal.read_text())[self.pending["id"]],
            self.receipt,
        )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            reconcile_pending(
                self.cards,
                self.key,
                self.pending["id"],
                None,
                self.journal,
                True,
            )
        reconcile_pending(
            self.cards,
            self.key,
            self.pending["id"],
            self.receipt,
            self.journal,
            True,
        )

    def test_wrong_intent_wrong_dm_and_archival_stop_recovery(self):
        count = len(self.github.mutations)
        with self.assertRaisesRegex(ValueError, "does not match"):
            reconcile_pending(
                self.cards, self.key, "wrong-id", None, self.journal, True
            )
        with self.assertRaisesRegex(ValueError, "different DM"):
            reconcile_pending(
                self.cards,
                self.key,
                self.pending["id"],
                {**self.receipt, "channel": "DOTHER"},
                self.journal,
                True,
            )
        self.cards.cards[self.key]["item"]["isArchived"] = True
        self.assertTrue(inspect_pending(self.cards, self.key)["archived"])
        with self.assertRaisesRegex(RuntimeError, "Restore archived"):
            reconcile_pending(
                self.cards,
                self.key,
                self.pending["id"],
                None,
                self.journal,
                True,
            )
        self.assertEqual(len(self.github.mutations), count)

    def test_parent_thread_cannot_acknowledge_a_new_reminder(self):
        receipt = message_receipt(
            self.config["slack"]["workspace_url"]
            + "/archives/D123/p1794830400000001",
            self.config["slack"]["workspace_url"],
        )
        with self.assertRaisesRegex(ValueError, "parent thread"):
            reconcile_pending(
                self.cards,
                self.key,
                self.pending["id"],
                receipt,
                self.journal,
                True,
            )

    def test_documented_cli_inspection_preview_and_apply(self):
        for sent in (False, True):
            with self.subTest(sent=sent):
                self.setUp()
                common = [
                    "recover.py",
                    "--key",
                    self.key,
                    "--output-dir",
                    self.directory.name,
                ]
                outcome = (
                    [
                        "--sent-message-url",
                        self.config["slack"]["workspace_url"]
                        + "/archives/D123/p1794916800123456",
                    ]
                    if sent
                    else ["--not-sent"]
                )
                commands = [
                    [*common, "--inspect"],
                    [*common, "--pending-id", self.pending["id"], *outcome],
                    [
                        *common,
                        "--pending-id",
                        self.pending["id"],
                        *outcome,
                        "--apply",
                    ],
                ]
                for args in commands:
                    with ExitStack() as stack:
                        stack.enter_context(patch("sys.argv", args))
                        stack.enter_context(patch("sys.stdout", StringIO()))
                        stack.enter_context(
                            patch("recover.GitHub", return_value=self.github)
                        )
                        stack.enter_context(
                            patch("recover.read_project", return_value=STATE)
                        )
                        main()
                result = json.loads(
                    (Path(self.directory.name) / "recovery.json").read_text()
                )
                self.assertTrue(result["applied"])
                self.assertEqual(
                    result["outcome"], "sent" if sent else "not-sent"
                )

    def test_permalink_requires_configured_workspace_and_dm_message(self):
        workspace = self.config["slack"]["workspace_url"]
        self.assertEqual(self.receipt["ts"], "1794916800.123456")
        for url in (
            workspace + "/archives/C123/p1794916800123456",
            "https://example.com/archives/D123/p1794916800123456",
            workspace + "/archives/D123",
            workspace.replace("https:", "http:")
            + "/archives/D123/p1794916800123456",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                message_receipt(url, workspace)
