# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay Project synchronization, GitHub handoffs, and interrupted DM deliveries."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from actions import preview
from pilot import delivery_history, recover_receipts, send_notifications
from sync import ProjectCards, metadata
from test_actions import CONFIG, python_item, snapshot
from test_report import NOW, POLICY, REQUESTED, request, review

SCHEMA = json.loads(
    Path(__file__).with_name("project_schema.json").read_text()
)
STATE = {
    "fields": [
        {
            "id": field["name"],
            "name": field["name"],
            "dataType": field["data_type"],
            "options": [
                {"id": option["name"], "name": option["name"]}
                for option in field.get("options", [])
            ],
        }
        for field in SCHEMA["fields"]
    ]
}
ROUTE = "dm-test:U01PXPJ5741"


class FakeGitHub:
    def __init__(self):
        self.items = {}
        self.mutations = []

    def graphql(self, query, variables):
        if "items(first:" in query:
            return {
                "node": {
                    "items": {
                        "nodes": deepcopy(list(self.items.values())),
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            }
        if "user(login:" in query:
            return {"user": {"id": "UID:" + variables["login"]}}
        self.mutations.append(query)
        value = variables["input"]
        if "addProjectV2DraftIssue(" in query:
            item_id = str(len(self.items) + 1)
            self.items[item_id] = {
                "id": item_id,
                "fullDatabaseId": item_id,
                "isArchived": False,
                "content": self.content(value, "draft:" + item_id),
                "fieldValues": {
                    "nodes": [],
                    "pageInfo": {"hasNextPage": False},
                },
            }
            return {
                "addProjectV2DraftIssue": {
                    "projectItem": deepcopy(self.items[item_id])
                }
            }
        if "updateProjectV2DraftIssue(" in query:
            item = next(
                item
                for item in self.items.values()
                if item["content"]["id"] == value["draftIssueId"]
            )
            item["content"] = self.content(value, value["draftIssueId"])
            return {
                "updateProjectV2DraftIssue": {
                    "draftIssue": {"id": value["draftIssueId"]}
                }
            }
        item = self.items[value["itemId"]]
        nodes = item["fieldValues"]["nodes"]
        nodes[:] = [
            field
            for field in nodes
            if field["field"]["name"] != value["fieldId"]
        ]
        if "updateProjectV2ItemFieldValue(" in query:
            encoded = value["value"]
            field_type = next(
                field["dataType"]
                for field in STATE["fields"]
                if field["id"] == value["fieldId"]
            )
            key = {"TEXT": "text", "DATE": "date", "SINGLE_SELECT": "name"}[
                field_type
            ]
            nodes.append(
                {
                    "field": {"name": value["fieldId"]},
                    key: next(iter(encoded.values())),
                }
            )
            method = "updateProjectV2ItemFieldValue"
        else:
            method = "clearProjectV2ItemFieldValue"
        return {method: {"projectV2Item": {"id": item["id"]}}}

    @staticmethod
    def content(value, draft_id):
        return {
            "id": draft_id,
            "title": value["title"],
            "body": value["body"],
            "assignees": {
                "nodes": [
                    {"id": user, "login": user[4:]}
                    for user in value["assigneeIds"]
                ],
                "pageInfo": {"hasNextPage": False},
            },
        }


class SyncTests(unittest.TestCase):
    def setUp(self):
        clock = patch("pilot.timestamp", return_value=NOW)
        clock.start()
        self.addCleanup(clock.stop)
        self.github = FakeGitHub()
        self.config = deepcopy(CONFIG)
        self.config["slack"]["users"].update(first="U111", second="U222")
        self.data = python_item()
        request(self.data, "first", REQUESTED)
        request(self.data, "second", REQUESTED)

    def cards(self, apply=True):
        return ProjectCards(self.github, self.config, STATE, apply)

    def sync(self):
        cards = self.cards()
        plan = preview(
            snapshot(self.data),
            POLICY,
            self.config,
            routing_history=cards.routing_history(),
        )
        cards.reconcile(plan, snapshot(self.data))
        return cards, plan

    def test_author_conflict_supersedes_fallback_and_survives_unknown(self):
        self.data["pull_request"]["requested_reviewers"] = []
        review(self.data, "APPROVED")
        cards, plan = self.sync()
        original = plan["actions"][0]["key"]
        self.assertEqual(plan["actions"][0]["owner"], "vyasr")
        self.data["pull_request"]["mergeable"] = False
        cards, plan = self.sync()
        conflict = plan["actions"][0]["key"]
        self.assertEqual(cards.cards[original]["values"]["Status"], "Complete")
        self.assertIn(
            "Superseded", cards.cards[original]["data"]["resolution"]
        )
        self.assertEqual(
            cards.cards[conflict]["data"]["action"]["owner"], "author"
        )
        self.data["pull_request"]["mergeable"] = None
        cards, plan = self.sync()
        self.assertEqual(plan["actions"][0]["key"], conflict)
        self.assertEqual(
            cards.cards[conflict]["values"]["Status"], "Awaiting action"
        )
        self.data["pull_request"]["mergeable"] = True
        cards, plan = self.sync()
        self.assertEqual(cards.cards[conflict]["values"]["Status"], "Complete")
        self.assertIn(
            "without conflicts", cards.cards[conflict]["data"]["resolution"]
        )

    def test_new_conflict_after_resolution_starts_new_episode(self):
        self.data["pull_request"]["mergeable"] = False
        cards, plan = self.sync()
        old = next(
            a for a in plan["actions"] if a["kind"] == "Resolve conflicts"
        )
        self.data["pull_request"]["mergeable"] = True
        self.sync()
        self.data["pull_request"]["mergeable"] = False
        saved = snapshot(self.data)
        saved["collected_at"] = "2026-11-25T12:00:00Z"
        cards = self.cards()
        plan = preview(
            saved, POLICY, self.config, routing_history=cards.routing_history()
        )
        new = next(
            a for a in plan["actions"] if a["kind"] == "Resolve conflicts"
        )
        self.assertNotEqual(old["key"], new["key"])
        self.assertEqual(new["since"], saved["collected_at"])

    def test_resumed_inspection_has_new_stable_handoff_and_delivery_clock(
        self,
    ):
        self.data["pull_request"]["requested_reviewers"] = []
        review(self.data, "APPROVED")
        cards, plan = self.sync()
        original = plan["actions"][0]["key"]
        self.data["pull_request"]["mergeable"] = False
        self.sync()
        self.data["pull_request"]["mergeable"] = True
        resolved = "2026-11-20T12:00:00Z"
        saved = snapshot(self.data)
        saved["collected_at"] = resolved
        cards = self.cards()
        plan = preview(
            saved, POLICY, self.config, routing_history=cards.routing_history()
        )
        cards.reconcile(plan, saved)
        resumed = plan["actions"][0]
        self.assertEqual(resumed["kind"], "Inspect blockers")
        self.assertNotEqual(resumed["key"], original)
        self.assertEqual(resumed["since"], resolved)
        self.assertEqual(
            cards.cards[resumed["key"]]["values"]["Status"], "Awaiting action"
        )
        for now in [resolved, "2026-11-23T12:00:00Z"]:
            saved["collected_at"] = now
            cards = self.cards()
            plan = preview(
                saved,
                POLICY,
                self.config,
                routing_history=cards.routing_history(),
            )
            self.assertEqual(plan["actions"][0]["key"], resumed["key"])
            cards.reconcile(plan, saved)

    def test_dry_run_then_create_then_rerun_without_mutations(self):
        plan = preview(snapshot(self.data), POLICY, self.config)
        dry_run = self.cards(False)
        dry_run.reconcile(plan, snapshot(self.data))
        self.assertEqual(len(dry_run.changes), 2)
        self.assertFalse(self.github.mutations)
        cards, _ = self.sync()
        self.assertEqual(len(cards.cards), 2)
        count = len(self.github.mutations)
        again, _ = self.sync()
        self.assertEqual(again.changes, [])
        self.assertEqual(len(self.github.mutations), count)
        self.assertEqual(
            {
                card["item"]["content"]["assignees"]["nodes"][0]["login"]
                for card in cards.cards.values()
            },
            {"first", "second"},
        )

    def test_submitted_review_completes_only_that_reviewers_card(self):
        cards, _ = self.sync()
        keys = {
            card["data"]["action"]["owner"]: key
            for key, card in cards.cards.items()
        }
        review(self.data, "APPROVED", who="first")
        self.data["pull_request"]["requested_reviewers"] = [
            {"login": "second"}
        ]
        cards, _ = self.sync()
        self.assertEqual(
            cards.cards[keys["first"]]["values"]["Status"], "Complete"
        )
        self.assertIn(
            "review submitted",
            cards.cards[keys["first"]]["data"]["resolution"],
        )
        self.assertEqual(
            cards.cards[keys["second"]]["values"]["Status"], "Awaiting action"
        )

    def test_rerequest_supersedes_old_episode_and_preserves_other(self):
        cards, _ = self.sync()
        old = {
            card["data"]["action"]["owner"]: key
            for key, card in cards.cards.items()
        }
        self.data["timeline"][0]["created_at"] = NOW
        cards, _ = self.sync()
        self.assertEqual(len(cards.cards), 3)
        self.assertEqual(
            cards.cards[old["first"]]["values"]["Status"], "Complete"
        )
        self.assertIn(
            "Superseded", cards.cards[old["first"]]["data"]["resolution"]
        )
        self.assertEqual(
            cards.cards[old["second"]]["values"]["Status"], "Awaiting action"
        )

    def test_close_and_draft_are_distinct_and_partial_scope_is_preserved(self):
        self.sync()
        other = python_item()
        other["issue"].update(
            number=2, html_url="https://github.com/NVIDIA/cudf/pull/2"
        )
        request(other, "first")
        plan = preview(snapshot(other), POLICY, self.config)
        cards = self.cards()
        cards.reconcile(plan, snapshot(other))
        self.data["pull_request"]["draft"] = True
        cards, _ = self.sync()
        self.assertEqual(
            [
                card["values"]["Status"]
                for card in cards.cards.values()
                if card["data"]["action"]["number"] == 1
            ],
            ["Deferred", "Deferred"],
        )
        self.data["issue"]["state"] = "closed"
        cards, _ = self.sync()
        for card in cards.cards.values():
            self.assertEqual(
                card["values"]["Status"],
                "Complete"
                if card["data"]["action"]["number"] == 1
                else "Awaiting action",
            )

    def test_label_loss_does_not_claim_completion(self):
        self.sync()
        self.data["issue"]["labels"] = []
        cards, _ = self.sync()
        self.assertTrue(
            all(
                card["values"]["Status"] == "Awaiting action"
                for card in cards.cards.values()
            )
        )

    def test_archived_cards_are_not_changed(self):
        self.sync()
        self.github.items["1"]["isArchived"] = True
        archived = deepcopy(self.github.items["1"])
        self.data["pull_request"]["head"]["sha"] = "new-head"
        self.sync()
        self.assertEqual(self.github.items["1"], archived)

    def test_duplicate_managed_keys_abort_before_mutation(self):
        self.sync()
        self.github.items["duplicate"] = deepcopy(self.github.items["1"])
        count = len(self.github.mutations)
        with self.assertRaisesRegex(ValueError, "Duplicate managed action"):
            self.cards()
        self.assertEqual(len(self.github.mutations), count)

    def test_manual_notes_and_status_are_preserved(self):
        self.sync()
        item = self.github.items["1"]
        item["content"]["body"] += "Keep this maintainer note.\n"
        status = next(
            field
            for field in item["fieldValues"]["nodes"]
            if field["field"]["name"] == "Status"
        )
        status["name"] = "In progress"
        self.data["pull_request"]["head"]["sha"] = "new-head"
        self.sync()
        self.assertTrue(
            item["content"]["body"].endswith("Keep this maintainer note.\n")
        )
        self.assertEqual(
            metadata(item["content"]["body"])["action"]["source_sha"],
            "new-head",
        )
        self.assertEqual(status["name"], "In progress")

    def test_dm_send_rerun_and_separate_production_clocks(self):
        cards, _ = self.sync()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.json"
            journal = {}
            dm_plan = preview(
                snapshot(self.data),
                POLICY,
                self.config,
                history=delivery_history(cards, ROUTE),
            )
            with patch(
                "pilot.post",
                side_effect=[
                    {"channel": "D123", "ts": "1.0"},
                    {"channel": "D123", "ts": "2.0"},
                ],
            ) as post:
                send_notifications(
                    cards,
                    dm_plan,
                    ROUTE,
                    "U01PXPJ5741",
                    "secret",
                    journal,
                    path,
                )
            self.assertEqual(post.call_count, 2)
            self.assertTrue(
                all(
                    call.args[1]["channel"] == "U01PXPJ5741"
                    for call in post.call_args_list
                )
            )
            cards, _ = self.sync()
            again = preview(
                snapshot(self.data),
                POLICY,
                self.config,
                history=delivery_history(cards, ROUTE),
            )
            self.assertTrue(
                all(
                    action["notification"]["kind"] == "none"
                    for action in again["actions"]
                )
            )
            for card in cards.cards.values():
                self.assertIsNone(card["values"].get("Escalate on"))
                self.assertIsNone(card["values"].get("Follow-up"))
                self.assertTrue(
                    card["data"]["deliveries"][ROUTE]["first_notified_at"]
                )
            self.assertEqual(len(json.loads(path.read_text())), 2)

    def test_receipt_repairs_send_followed_by_failed_project_ack(self):
        cards, plan = self.sync()
        plan["actions"] = plan["actions"][:1]
        key = plan["actions"][0]["key"]
        original_save = cards.save
        calls = []

        def fail_ack(*args):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("Project unavailable")
            return original_save(*args)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.json"
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(cards, "save", side_effect=fail_ack)
                )
                post = stack.enter_context(
                    patch(
                        "pilot.post",
                        return_value={"channel": "D123", "ts": "1.0"},
                    )
                )
                with self.assertRaisesRegex(
                    RuntimeError, "Project unavailable"
                ):
                    send_notifications(
                        cards, plan, ROUTE, "U01PXPJ5741", "secret", {}, path
                    )
            self.assertEqual(post.call_count, 1)
            cards = self.cards()
            self.assertEqual(
                recover_receipts(cards, ROUTE, json.loads(path.read_text())),
                [],
            )
            record = cards.cards[key]["data"]["deliveries"][ROUTE]
            self.assertNotIn("pending", record)
            self.assertEqual(record["thread_ts"], "1.0")

    def test_reminder_and_escalation_use_the_initial_dm_thread(self):
        self.sync()
        with tempfile.TemporaryDirectory() as directory:
            journal = {}
            path = Path(directory) / "receipts.json"
            for at, kind in [
                (NOW, "initial"),
                ("2026-11-19T12:00:00Z", "reminder"),
                ("2026-11-23T12:00:00Z", "escalation"),
            ]:
                cards = self.cards()
                saved = snapshot(self.data)
                saved["collected_at"] = at
                plan = preview(
                    saved,
                    POLICY,
                    self.config,
                    history=delivery_history(cards, ROUTE),
                )
                plan["actions"] = plan["actions"][:1]
                self.assertEqual(
                    plan["actions"][0]["notification"]["kind"], kind
                )
                with ExitStack() as stack:
                    stack.enter_context(
                        patch("pilot.timestamp", return_value=at)
                    )
                    post = stack.enter_context(
                        patch(
                            "pilot.post",
                            return_value={
                                "channel": "D123",
                                "ts": "1.0" if kind == "initial" else "2.0",
                            },
                        )
                    )
                    send_notifications(
                        cards,
                        plan,
                        ROUTE,
                        "U01PXPJ5741",
                        "secret",
                        journal,
                        path,
                    )
                message = post.call_args[0][1]
                if kind == "initial":
                    self.assertNotIn("thread_ts", message)
                else:
                    self.assertEqual(message["channel"], "U01PXPJ5741")
                    self.assertEqual(message["thread_ts"], "1.0")
                if kind == "escalation":
                    self.assertIn("<@U01PXPJ5741>", message["text"])
            cards = self.cards()
            plan = preview(
                saved,
                POLICY,
                self.config,
                history=delivery_history(cards, ROUTE),
            )
            self.assertEqual(
                plan["actions"][0]["notification"]["kind"], "none"
            )

    def test_ambiguous_send_remains_pending_and_is_not_retried(self):
        cards, plan = self.sync()
        plan["actions"] = plan["actions"][:1]
        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            post = stack.enter_context(
                patch("pilot.post", side_effect=RuntimeError("timeout"))
            )
            with self.assertRaisesRegex(RuntimeError, "timeout"):
                send_notifications(
                    cards,
                    plan,
                    ROUTE,
                    "U01PXPJ5741",
                    "secret",
                    {},
                    Path(directory) / "receipts.json",
                )
            self.assertEqual(post.call_count, 1)
        cards = self.cards()
        self.assertEqual(
            recover_receipts(cards, ROUTE, {}), [plan["actions"][0]["key"]]
        )


if __name__ == "__main__":
    unittest.main()
