# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise independent ownership and notification clocks without sending."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from actions import add_business_days, notification_state, preview, slack_text
from test_report import NOW, POLICY, REQUESTED, REVIEWED, item, request, review

CONFIG = json.loads(Path(__file__).with_name("pilot.json").read_text())


def python_item():
    result = item()
    result["issue"]["labels"].append({"name": "Python"})
    return result


def snapshot(data):
    return {
        "schema_version": 1,
        "repository": "NVIDIA/cudf",
        "collected_at": NOW,
        "selection": {"mode": "sample"},
        "items": [data],
    }


class ActionTests(unittest.TestCase):
    def test_confirmed_conflict_routes_concrete_action_to_author(self):
        data = python_item()
        review(data, "APPROVED")
        data["pull_request"]["mergeable"] = False
        actions = preview(snapshot(data), POLICY, CONFIG)["actions"]
        self.assertEqual(
            [(a["kind"], a["owner"]) for a in actions],
            [("Resolve conflicts", "author")],
        )
        self.assertEqual(actions[0]["since"], NOW)
        self.assertFalse(actions[0]["overdue"])

    def test_conflict_does_not_hide_independent_requested_reviews(self):
        data = python_item()
        request(data, "other")
        data["pull_request"]["mergeable"] = False
        actions = preview(snapshot(data), POLICY, CONFIG)["actions"]
        self.assertEqual(
            {(a["kind"], a["owner"]) for a in actions},
            {("Resolve conflicts", "author"), ("Review", "other")},
        )

    def test_failed_ci_and_unknown_mergeability_do_not_assign_author_work(
        self,
    ):
        for mergeable in [True, None]:
            data = python_item()
            review(data, "APPROVED")
            data["pull_request"]["mergeable"] = mergeable
            data["check_runs"] = [
                {
                    "name": "build",
                    "conclusion": "failure",
                    "status": "completed",
                }
            ]
            actions = preview(snapshot(data), POLICY, CONFIG)["actions"]
            self.assertEqual(
                [(a["kind"], a["owner"]) for a in actions],
                [("Inspect blockers", "vyasr")],
            )

    def test_bot_author_conflict_requires_human_routing(self):
        data = python_item()
        review(data, "APPROVED")
        data["issue"]["user"] = {"login": "automation[bot]", "type": "Bot"}
        data["pull_request"]["mergeable"] = False
        actions = preview(snapshot(data), POLICY, CONFIG)["actions"]
        self.assertEqual(
            [(a["kind"], a["owner"]) for a in actions],
            [("Inspect blockers", "vyasr")],
        )

    def test_conflict_clock_survives_refresh_and_unknown_mergeability(self):
        data = python_item()
        review(data, "APPROVED")
        data["pull_request"]["mergeable"] = False
        first = preview(snapshot(data), POLICY, CONFIG)["actions"]
        later = snapshot(data)
        later["collected_at"] = "2026-11-25T12:00:00Z"
        for mergeable in [False, None]:
            later["items"][0]["pull_request"]["mergeable"] = mergeable
            actions = preview(later, POLICY, CONFIG, routing_history=first)[
                "actions"
            ]
            self.assertEqual(actions[0]["key"], first[0]["key"])
            self.assertEqual(actions[0]["since"], NOW)
            self.assertTrue(actions[0]["overdue"])
            self.assertEqual(actions[0]["kind"], "Resolve conflicts")

    def test_reviewers_have_independent_cards_and_clocks(self):
        data = python_item()
        request(data, "first", REQUESTED)
        request(data, "second", REVIEWED)
        plan = preview(snapshot(data), POLICY, CONFIG)
        actions = {action["owner"]: action for action in plan["actions"]}
        self.assertEqual(set(actions), {"first", "second"})
        self.assertEqual(actions["first"]["since"], REQUESTED)
        self.assertEqual(actions["second"]["since"], REVIEWED)
        self.assertNotEqual(actions["first"]["key"], actions["second"]["key"])

    def test_author_action_does_not_hide_another_reviewers_request(self):
        data = python_item()
        review(data, "CHANGES_REQUESTED")
        request(data, "other", NOW)
        actions = preview(snapshot(data), POLICY, CONFIG)["actions"]
        self.assertEqual(
            {(action["kind"], action["owner"]) for action in actions},
            {("Respond", "author"), ("Review", "other")},
        )

    def test_team_only_request_routes_to_primary_for_named_reviewer(self):
        data = python_item()
        data["pull_request"]["requested_teams"] = [
            {"slug": "cudf-python-codeowners"}
        ]
        data["timeline"] = [
            {
                "event": "review_requested",
                "created_at": REQUESTED,
                "requested_team": {"slug": "cudf-python-codeowners"},
            }
        ]
        actions = preview(snapshot(data), POLICY, CONFIG)["actions"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["owner"], "vyasr")
        self.assertEqual(actions[0]["kind"], "Assign reviewer")
        request(data, "named")
        actions = preview(snapshot(data), POLICY, CONFIG)["actions"]
        self.assertEqual([action["owner"] for action in actions], ["named"])

    def test_unlabeled_issue_and_cpp_only_work_are_not_silently_routed_to_python(
        self,
    ):
        data = item()
        request(data)
        result = preview(snapshot(data), POLICY, CONFIG)
        self.assertEqual(result["actions"], [])
        self.assertIn("No confirmed Python", result["omitted"][0]["reason"])

    def test_drafts_do_not_create_routine_actions(self):
        data = python_item()
        data["pull_request"]["draft"] = True
        self.assertEqual(
            preview(snapshot(data), POLICY, CONFIG)["actions"], []
        )

    def test_unknown_request_time_does_not_start_reminders(self):
        data = python_item()
        request(data)
        data["timeline"] = []
        action = preview(snapshot(data), POLICY, CONFIG)["actions"][0]
        self.assertEqual(action["attention"], "Unknown age")
        self.assertEqual(action["notification"]["kind"], "none")

    def test_importing_an_old_wait_never_creates_an_immediate_escalation(self):
        data = python_item()
        request(data)
        action = preview(snapshot(data), POLICY, CONFIG)["actions"][0]
        self.assertTrue(action["overdue"])
        self.assertEqual(action["notification"]["kind"], "initial")
        self.assertIsNone(action["notification"]["escalate_on"])
        self.assertIsNone(action["project_fields"]["Escalate on"])

    def test_reminder_uses_successful_delivery_and_business_time(self):
        action = {"key": "key", "overdue": True}
        history = {"key": {"first_notified_at": "2026-11-13T12:00:00Z"}}
        result = notification_state(action, CONFIG, history, NOW)
        self.assertEqual(result["kind"], "none")
        self.assertEqual(result["follow_up"], "2026-11-18T12:00:00+00:00")
        result = notification_state(
            action, CONFIG, history, "2026-11-18T12:00:00Z"
        )
        self.assertEqual(result["kind"], "reminder")

    def test_weekend_start_uses_same_business_clock_as_aging(self):
        self.assertEqual(
            add_business_days("2026-11-14T12:00:00Z", 3),
            "2026-11-19T00:00:00+00:00",
        )

    def test_escalation_after_one_week_does_not_repeat_and_consolidates_mentions(
        self,
    ):
        data = python_item()
        review(data, "APPROVED")
        plan = preview(snapshot(data), POLICY, CONFIG)
        action = plan["actions"][0]
        history = {
            action["key"]: {"first_notified_at": "2026-11-09T12:00:00Z"}
        }
        action = preview(snapshot(data), POLICY, CONFIG, history=history)[
            "actions"
        ][0]
        self.assertEqual(action["notification"]["kind"], "escalation")
        self.assertEqual(
            slack_text(action, CONFIG, None).count("<@U01PXPJ5741>"), 1
        )
        history[action["key"]].update(escalated_at=NOW, last_notified_at=NOW)
        action = preview(snapshot(data), POLICY, CONFIG, history=history)[
            "actions"
        ][0]
        self.assertEqual(action["notification"]["kind"], "none")

    def test_new_revision_preserves_request_identity_but_rerequest_does_not(
        self,
    ):
        data = python_item()
        request(data)
        original = preview(snapshot(data), POLICY, CONFIG)["actions"][0]["key"]
        data["pull_request"]["head"]["sha"] = "updated-head"
        self.assertEqual(
            preview(snapshot(data), POLICY, CONFIG)["actions"][0]["key"],
            original,
        )
        data["timeline"][0]["created_at"] = REVIEWED
        self.assertNotEqual(
            preview(snapshot(data), POLICY, CONFIG)["actions"][0]["key"],
            original,
        )

    def test_previews_are_repeatable_and_expose_unmapped_recipients(self):
        data = python_item()
        request(data)
        saved = snapshot(data)
        before = deepcopy(saved)
        first = preview(saved, POLICY, CONFIG)
        self.assertEqual(first, preview(saved, POLICY, CONFIG))
        self.assertEqual(saved, before)
        self.assertEqual(first["missing_slack_mappings"], ["reviewer"])
        self.assertIn(
            "Slack mapping needed", first["actions"][0]["slack_preview"]
        )


if __name__ == "__main__":
    unittest.main()
