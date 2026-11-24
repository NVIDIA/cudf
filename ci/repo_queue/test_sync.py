# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay Project synchronization, GitHub handoffs, and interrupted DM deliveries."""

import base64
import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from actions import preview
from slack_dm_test import SlackRejected
from sync import END, MARKER, START, ProjectCards, managed_body, metadata
from test_actions import CONFIG, python_item, snapshot
from test_report import NOW, POLICY, REQUESTED, request, review
from track import delivery_history, recover_receipts, send_notifications

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
        if "... on DraftIssue{id title body" in query:
            return {
                "node": deepcopy(
                    next(
                        item["content"]
                        for item in self.items.values()
                        if item["content"]["id"] == variables["id"]
                    )
                )
            }
        if "... on ProjectV2Item{fieldValueByName" in query:
            nodes = self.items[variables["id"]]["fieldValues"]["nodes"]
            status = next(
                (f["name"] for f in nodes if f["field"]["name"] == "Status"),
                None,
            )
            return {
                "node": {
                    "fieldValueByName": {"name": status} if status else None
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
    def test_automatic_dispositions_resume_but_manual_statuses_survive(self):
        for transition in ("draft", "closed", "parked"):
            with self.subTest(transition=transition):
                self.setUp()
                self.data["pull_request"]["requested_reviewers"] = []
                review(self.data, "CHANGES_REQUESTED", who="unsolicited")
                cards, plan = self.sync()
                key = plan["actions"][0]["key"]
                if transition == "draft":
                    self.data["pull_request"]["draft"] = True
                elif transition == "closed":
                    self.data["issue"]["state"] = "closed"
                else:
                    self.data["issue"]["labels"].append({"name": "parked"})
                with patch.dict(POLICY, {"parked_labels": ["parked"]}):
                    self.sync()
                self.data["pull_request"]["draft"] = False
                self.data["issue"]["state"] = "open"
                self.data["issue"]["labels"] = [{"name": "Python"}]
                cards, _ = self.sync()
                self.assertEqual(
                    cards.cards[key]["values"]["Status"], "Awaiting action"
                )
                self.assertNotIn("resolution", cards.cards[key]["data"])
                self.assertNotIn("resolved_at", cards.cards[key]["data"])
                for manual in ("Deferred", "Complete", "In progress"):
                    native = self.github.items[cards.cards[key]["item"]["id"]]
                    status = next(
                        f
                        for f in native["fieldValues"]["nodes"]
                        if f["field"]["name"] == "Status"
                    )
                    status["name"] = manual
                    cards, _ = self.sync()
                    self.assertEqual(
                        cards.cards[key]["values"]["Status"], manual
                    )
                    self.data["pull_request"]["draft"] = True
                    self.sync()
                    self.data["pull_request"]["draft"] = False
                    cards, _ = self.sync()
                    self.assertEqual(
                        cards.cards[key]["values"]["Status"], manual
                    )

    def test_legacy_dispositions_resume_without_losing_manual_overrides(self):
        for status, expected in (
            ("Deferred", "Awaiting action"),
            ("Complete", "Complete"),
        ):
            with self.subTest(status=status):
                self.setUp()
                cards, plan = self.sync()
                key = plan["actions"][0]["key"]
                data = deepcopy(cards.cards[key]["data"])
                data.pop("written_status")
                data["resolution"] = "Source is draft."
                cards.save(key, data, {"Status": status})
                cards, _ = self.sync()
                self.assertEqual(
                    cards.cards[key]["values"]["Status"], expected
                )
                self.assertNotIn("resolution", cards.cards[key]["data"])

    def test_damaged_managed_identity_aborts_before_duplicate_or_mutation(
        self,
    ):
        for damage in ("marker", "all_body", "corrupt_marker"):
            with self.subTest(damage=damage):
                self.setUp()
                self.sync()
                native = self.github.items["1"]
                if damage == "all_body":
                    native["content"]["body"] = (
                        "Accidentally replaced the body"
                    )
                elif damage == "corrupt_marker":
                    native["content"]["body"] = native["content"][
                        "body"
                    ].replace(
                        "cudf-work-tracking:v1:", "cudf-work-tracking:v1:!"
                    )
                else:
                    native["content"]["body"] = MARKER.sub(
                        "", native["content"]["body"]
                    )
                count = len(self.github.mutations)
                with self.assertRaisesRegex(ValueError, "metadata"):
                    self.sync()
                self.assertEqual(len(self.github.mutations), count)
                self.assertEqual(len(self.github.items), 2)

    def test_author_activity_updates_clock_without_completing_or_reassigning_card(
        self,
    ):
        self.data["pull_request"]["requested_reviewers"] = []
        review(self.data, "CHANGES_REQUESTED", who="unsolicited")
        cards, plan = self.sync()
        key = plan["actions"][0]["key"]
        self.data["timeline"].append(
            {
                "event": "commented",
                "created_at": NOW,
                "user": {"login": "author"},
            }
        )
        cards, plan = self.sync()
        self.assertEqual(len(cards.cards), 1)
        self.assertEqual(plan["actions"][0]["key"], key)
        self.assertEqual(
            cards.cards[key]["data"]["action"]["last_author_activity_at"], NOW
        )
        self.assertEqual(cards.cards[key]["data"]["action"]["owner"], "author")
        self.assertEqual(
            cards.cards[key]["values"]["Status"], "Awaiting action"
        )
        request(self.data, "unsolicited", NOW)
        cards, plan = self.sync()
        self.assertEqual(cards.cards[key]["values"]["Status"], "Complete")
        self.assertEqual(
            [(a["kind"], a["owner"]) for a in plan["actions"]],
            [("Review", "unsolicited")],
        )

    def setUp(self):
        clock = patch("track.timestamp", return_value=NOW)
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
        persisted = next(
            value
            for value in item["fieldValues"]["nodes"]
            if value["field"]["name"] == "Status"
        )
        self.assertEqual(persisted["name"], "In progress")

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
                "track.post",
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
            if len(calls) >= 2:
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
                        "track.post",
                        return_value={"channel": "D123", "ts": "1.0"},
                    )
                )
                stack.enter_context(patch("track.time.sleep"))
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

    def test_transient_ack_failure_retries_without_another_slack_send(self):
        cards, plan = self.sync()
        plan["actions"] = plan["actions"][:1]
        original = cards.save
        calls = []

        def save(*args):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("Temporary Project failure")
            return original(*args)

        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(patch.object(cards, "save", side_effect=save))
            stack.enter_context(patch("track.time.sleep"))
            post = stack.enter_context(
                patch(
                    "track.post", return_value={"channel": "D123", "ts": "1.0"}
                )
            )
            sent = send_notifications(
                cards,
                plan,
                ROUTE,
                "U01PXPJ5741",
                "secret",
                {},
                Path(directory) / "receipts.json",
            )
            self.assertEqual(sent, [plan["actions"][0]["key"]])
            self.assertEqual(post.call_count, 1)
        self.assertEqual(recover_receipts(self.cards(), ROUTE, {}), [])

    def test_definite_slack_rejection_clears_pending_and_can_be_retried(self):
        cards, plan = self.sync()
        plan["actions"] = plan["actions"][:1]
        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(
                patch(
                    "track.post",
                    side_effect=SlackRejected("Slack API: invalid_auth"),
                )
            )
            with self.assertRaises(SlackRejected):
                send_notifications(
                    cards,
                    plan,
                    ROUTE,
                    "U01PXPJ5741",
                    "secret",
                    {},
                    Path(directory) / "receipts.json",
                )
        self.assertEqual(recover_receipts(self.cards(), ROUTE, {}), [])
        self.assertNotIn(
            "first_notified_at",
            self.cards().cards[plan["actions"][0]["key"]]["data"][
                "deliveries"
            ][ROUTE],
        )

    def test_metadata_must_be_inside_block_and_structurally_complete(self):
        self.sync()
        initial = deepcopy(self.github.items)
        for damage in (
            "outside",
            "deliveries",
            "record",
            "owner",
            "since",
            "transition",
            "status",
            "timestamp",
        ):
            with self.subTest(damage=damage):
                self.github.items = deepcopy(initial)
                native = self.github.items["1"]
                body = native["content"]["body"]
                data = metadata(body)
                if damage == "outside":
                    marker = MARKER.search(body).group()
                    native["content"]["body"] = (
                        START + "\n" + END + "\n" + marker
                    )
                else:
                    if damage == "deliveries":
                        data.pop("deliveries")
                    elif damage == "record":
                        data["deliveries"][ROUTE] = []
                    elif damage == "transition":
                        data["status_transition"] = {"to": "Complete"}
                    elif damage == "status":
                        data["manual_status"] = "Invalid option"
                    elif damage == "timestamp":
                        data["action"]["since"] = "0001-01-01T00:00:00+01:00"
                    else:
                        data["action"].pop(damage)
                    encoded = base64.b64encode(
                        json.dumps(data).encode()
                    ).decode()
                    native["content"]["body"] = MARKER.sub(
                        "<!-- cudf-work-tracking:v1:" + encoded + " -->", body
                    )
                count = len(self.github.mutations)
                with self.assertRaisesRegex(ValueError, "Project item 1"):
                    self.cards()
                self.assertEqual(len(self.github.mutations), count)

    def test_fresh_notes_survive_and_concurrent_managed_updates_abort(self):
        cards, plan = self.sync()
        action = plan["actions"][0]
        native = self.github.items[cards.cards[action["key"]]["item"]["id"]]
        native["content"]["body"] += "Fresh maintainer note\n"
        updated = deepcopy(cards.cards[action["key"]]["data"])
        updated["resolution"] = "Updated reason"
        cards.save(action["key"], updated, {})
        self.assertIn("Fresh maintainer note", native["content"]["body"])
        concurrent = {**updated, "resolution": "Another writer"}
        native["content"]["body"] = managed_body(
            concurrent, native["content"]["body"]
        )
        with self.assertRaisesRegex(RuntimeError, "fresh read"):
            cards.save(
                action["key"], {**updated, "resolution": "Stale overwrite"}, {}
            )
        self.assertEqual(
            metadata(native["content"]["body"])["resolution"], "Another writer"
        )

    def test_partial_status_writes_resume_without_creating_manual_override(
        self,
    ):
        for fail_after_status in (False, True):
            with self.subTest(fail_after_status=fail_after_status):
                self.setUp()
                cards, plan = self.sync()
                key = plan["actions"][0]["key"]
                self.data["pull_request"]["draft"] = True
                original = self.github.graphql

                def fail(query, variables):
                    if (
                        not fail_after_status
                        and "updateProjectV2ItemFieldValue(" in query
                        and variables["input"]["fieldId"] == "Status"
                    ):
                        raise RuntimeError("Field write failed")
                    if (
                        fail_after_status
                        and "updateProjectV2DraftIssue(" in query
                        and "status_transition"
                        not in json.dumps(metadata(variables["input"]["body"]))
                    ):
                        raise RuntimeError("Finalization failed")
                    return original(query, variables)

                with patch.object(self.github, "graphql", side_effect=fail):
                    with self.assertRaises(RuntimeError):
                        self.sync()
                self.assertIn(
                    "status_transition", self.cards().cards[key]["data"]
                )
                cards, _ = self.sync()
                self.assertEqual(
                    cards.cards[key]["values"]["Status"], "Deferred"
                )
                self.assertNotIn("manual_status", cards.cards[key]["data"])
                self.assertNotIn("status_transition", cards.cards[key]["data"])
                self.data["pull_request"]["draft"] = False
                cards, _ = self.sync()
                self.assertEqual(
                    cards.cards[key]["values"]["Status"], "Awaiting action"
                )

    def test_conflict_closure_is_not_clearance_and_reopen_preserves_manual_status(
        self,
    ):
        self.data["pull_request"]["requested_reviewers"] = []
        review(self.data, "APPROVED")
        self.data["pull_request"]["mergeable"] = False
        cards, plan = self.sync()
        key = plan["actions"][0]["key"]
        native = self.github.items[cards.cards[key]["item"]["id"]]
        next(
            f
            for f in native["fieldValues"]["nodes"]
            if f["field"]["name"] == "Status"
        )["name"] = "In progress"
        self.data["issue"]["state"] = "closed"
        cards, _ = self.sync()
        self.assertNotIn("resolved_at", cards.cards[key]["data"])
        self.data["issue"]["state"] = "open"
        cards, plan = self.sync()
        self.assertEqual(plan["actions"][0]["key"], key)
        self.assertEqual(cards.cards[key]["values"]["Status"], "In progress")

    def test_manually_completed_conflict_records_clearance_once(self):
        self.data["pull_request"]["requested_reviewers"] = []
        review(self.data, "APPROVED")
        self.data["pull_request"]["mergeable"] = False
        cards, plan = self.sync()
        key = plan["actions"][0]["key"]
        native = self.github.items[cards.cards[key]["item"]["id"]]
        next(
            f
            for f in native["fieldValues"]["nodes"]
            if f["field"]["name"] == "Status"
        )["name"] = "Complete"
        self.data["pull_request"]["mergeable"] = True
        cards, plan = self.sync()
        self.assertEqual(cards.cards[key]["data"]["resolved_at"], NOW)
        again, second = self.sync()
        self.assertEqual(
            [a["key"] for a in plan["actions"]],
            [a["key"] for a in second["actions"]],
        )
        self.assertEqual(again.changes, [])

    def test_failed_thread_field_write_retains_intent_until_receipt_recovery(
        self,
    ):
        cards, plan = self.sync()
        plan["actions"] = plan["actions"][:1]
        key = plan["actions"][0]["key"]
        original = self.github.graphql

        def fail_link(query, variables):
            if (
                "updateProjectV2ItemFieldValue(" in query
                and variables["input"]["fieldId"] == "Slack thread"
            ):
                raise RuntimeError("Thread field unavailable")
            return original(query, variables)

        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            path = Path(directory) / "receipts.json"
            stack.enter_context(patch("track.time.sleep"))
            post = stack.enter_context(
                patch(
                    "track.post", return_value={"channel": "D123", "ts": "1.0"}
                )
            )
            with patch.object(self.github, "graphql", side_effect=fail_link):
                with self.assertRaisesRegex(
                    RuntimeError, "Thread field unavailable"
                ):
                    send_notifications(
                        cards, plan, ROUTE, "U01PXPJ5741", "secret", {}, path
                    )
            saved = self.cards()
            record = saved.cards[key]["data"]["deliveries"][ROUTE]
            self.assertIn("pending", record)
            self.assertNotIn("first_notified_at", record)
            self.assertEqual(
                recover_receipts(saved, ROUTE, json.loads(path.read_text())),
                [],
            )
            self.assertIn(
                "/archives/D123/p10",
                saved.cards[key]["values"]["Slack thread"],
            )
            self.assertNotIn(
                "pending", saved.cards[key]["data"]["deliveries"][ROUTE]
            )
            self.assertEqual(post.call_count, 1)

    def test_rejection_cleanup_retries_and_reads_back_uncertain_write(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                self.setUp()
                cards, plan = self.sync()
                plan["actions"] = plan["actions"][:1]
                original = cards.save
                calls = []

                def save(*args):
                    calls.append(args)
                    if len(calls) == 2:
                        if committed:
                            original(*args)
                        raise RuntimeError("Cleanup write timed out")
                    return original(*args)

                with ExitStack() as stack:
                    directory = stack.enter_context(
                        tempfile.TemporaryDirectory()
                    )
                    stack.enter_context(
                        patch.object(cards, "save", side_effect=save)
                    )
                    refresh = stack.enter_context(
                        patch.object(cards, "refresh", wraps=cards.refresh)
                    )
                    stack.enter_context(patch("track.time.sleep"))
                    post = stack.enter_context(
                        patch(
                            "track.post",
                            side_effect=SlackRejected("invalid_auth"),
                        )
                    )
                    with self.assertRaises(SlackRejected):
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
                    self.assertEqual(refresh.call_count, 1)
                self.assertEqual(recover_receipts(self.cards(), ROUTE, {}), [])

    def test_legacy_closure_stays_distinct_from_confirmed_conflict_clearance(
        self,
    ):
        for distinguished in (False, True):
            with self.subTest(distinguished=distinguished):
                self.setUp()
                self.data["pull_request"]["requested_reviewers"] = []
                self.data["pull_request"]["mergeable"] = False
                cards, plan = self.sync()
                key = plan["actions"][0]["key"]
                native = self.github.items[cards.cards[key]["item"]["id"]]
                data = deepcopy(cards.cards[key]["data"])
                data.update(
                    resolved_at=NOW,
                    resolution="Source closed or merged: "
                    + self.data["issue"]["html_url"],
                    written_status="Complete",
                )
                if distinguished:
                    data["closed_at"] = NOW
                else:
                    data.pop("written_status")
                native["content"]["body"] = managed_body(
                    data, native["content"]["body"]
                )
                next(
                    f
                    for f in native["fieldValues"]["nodes"]
                    if f["field"]["name"] == "Status"
                )["name"] = "Deferred"
                self.data["issue"]["state"] = "closed"
                cards, _ = self.sync()
                self.assertEqual(
                    cards.cards[key]["data"].get("resolved_at"),
                    NOW if distinguished else None,
                )
                self.assertEqual(
                    next(
                        a for a in cards.routing_history() if a["key"] == key
                    )["resolved_at"],
                    NOW if distinguished else None,
                )
                self.assertEqual(cards.cards[key]["data"]["closed_at"], NOW)

    def test_status_edit_after_collection_aborts_then_reconciles_fresh_intent(
        self,
    ):
        self.sync()
        self.data["pull_request"]["draft"] = True
        cards, _ = self.sync()
        self.data["pull_request"]["draft"] = False
        cards = self.cards()
        plan = preview(
            snapshot(self.data),
            POLICY,
            self.config,
            routing_history=cards.routing_history(),
        )
        key = plan["actions"][0]["key"]
        native = self.github.items[cards.cards[key]["item"]["id"]]
        next(
            f
            for f in native["fieldValues"]["nodes"]
            if f["field"]["name"] == "Status"
        )["name"] = "In progress"
        with self.assertRaisesRegex(RuntimeError, "Status changed"):
            cards.reconcile(plan, snapshot(self.data))
        self.assertEqual(
            next(
                f
                for f in native["fieldValues"]["nodes"]
                if f["field"]["name"] == "Status"
            )["name"],
            "In progress",
        )
        cards, _ = self.sync()
        self.assertEqual(cards.cards[key]["values"]["Status"], "In progress")
        self.assertEqual(
            cards.cards[key]["data"]["manual_status"], "In progress"
        )

    def test_manual_status_matching_transition_target_is_preserved(self):
        for during_write in (False, True):
            with self.subTest(during_write=during_write):
                self.setUp()
                cards, _ = self.sync()
                self.data["pull_request"]["draft"] = True
                cards = self.cards()
                key = next(iter(cards.cards))
                native = self.github.items[cards.cards[key]["item"]["id"]]
                original_data = deepcopy(cards.cards[key]["data"])
                status = next(
                    f
                    for f in native["fieldValues"]["nodes"]
                    if f["field"]["name"] == "Status"
                )
                original = self.github.graphql
                changed = []

                def manual_edit(query, variables):
                    result = original(query, variables)
                    if (
                        during_write
                        and "updateProjectV2DraftIssue(" in query
                        and not changed
                    ):
                        changed.append(True)
                        status["name"] = "Deferred"
                    return result

                if not during_write:
                    status["name"] = "Deferred"
                plan = preview(
                    snapshot(self.data),
                    POLICY,
                    self.config,
                    routing_history=cards.routing_history(),
                )
                with patch.object(
                    self.github, "graphql", side_effect=manual_edit
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Status changed"
                    ):
                        cards.reconcile(plan, snapshot(self.data))
                self.assertEqual(
                    metadata(native["content"]["body"]), original_data
                )
                cards, _ = self.sync()
                self.data["pull_request"]["draft"] = False
                cards, _ = self.sync()
                self.assertEqual(
                    cards.cards[key]["values"]["Status"], "Deferred"
                )
                self.assertEqual(
                    cards.cards[key]["data"]["manual_status"], "Deferred"
                )

    def test_archived_pending_delivery_requires_restoration_before_recovery(
        self,
    ):
        cards, plan = self.sync()
        plan["actions"] = plan["actions"][:1]
        key = plan["actions"][0]["key"]
        native = self.github.items[cards.cards[key]["item"]["id"]]
        original = self.github.graphql

        def archive_during_ack(query, variables):
            if (
                "updateProjectV2ItemFieldValue(" in query
                and variables["input"]["fieldId"] == "Slack thread"
            ):
                native["isArchived"] = True
                raise RuntimeError("Write failed while card was archived")
            return original(query, variables)

        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            path = Path(directory) / "receipts.json"
            stack.enter_context(patch("track.time.sleep"))
            post = stack.enter_context(
                patch(
                    "track.post", return_value={"channel": "D123", "ts": "1.0"}
                )
            )
            with patch.object(
                self.github, "graphql", side_effect=archive_during_ack
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "Restore archived Project card"
                ):
                    send_notifications(
                        cards, plan, ROUTE, "U01PXPJ5741", "secret", {}, path
                    )
            journal = json.loads(path.read_text())
            for receipts in ({}, journal):
                with self.assertRaisesRegex(
                    RuntimeError, "Restore archived Project card"
                ):
                    recover_receipts(self.cards(), ROUTE, receipts)
            self.assertIn(
                "pending",
                metadata(native["content"]["body"])["deliveries"][ROUTE],
            )
            native["isArchived"] = False
            restored = self.cards()
            self.assertEqual(recover_receipts(restored, ROUTE, journal), [])
            self.assertNotIn(
                "pending", restored.cards[key]["data"]["deliveries"][ROUTE]
            )
            self.assertEqual(post.call_count, 1)

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
                        patch("track.timestamp", return_value=at)
                    )
                    post = stack.enter_context(
                        patch(
                            "track.post",
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
                patch("track.post", side_effect=RuntimeError("timeout"))
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
