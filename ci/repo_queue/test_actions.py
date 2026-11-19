# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise independent ownership and notification clocks without sending."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from actions import (
    add_business_days,
    notification_recipients,
    notification_state,
    preview,
    slack_text,
)
from report import make_report
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
    def test_bot_changes_requests_route_to_a_human(self):
        data = python_item()
        data["issue"]["user"] = {"login": "dependabot[bot]", "type": "Bot"}
        review(data, "CHANGES_REQUESTED", who="human")
        action = preview(snapshot(data), POLICY, CONFIG)["actions"][0]
        self.assertEqual(
            (action["kind"], action["owner"]),
            ("Clarify handoff", CONFIG["primary_contact"]),
        )
        self.assertIn("dependabot[bot]", action["description"])
        self.assertNotIn("author_inactivity_days", action)

    def test_report_and_board_share_history_aware_author_clock(self):
        data = python_item()
        review(data, "CHANGES_REQUESTED", who="reviewer")
        first = preview(snapshot(data), POLICY, CONFIG)["actions"][0]
        data["pull_request"]["head"]["sha"] = "new-head"
        data["head_commit_author"] = "author"
        saved = snapshot(data)
        saved["collected_at"] = "2026-11-23T12:00:00Z"
        data["observed_at"] = "2026-11-23T11:59:00Z"
        history = [first]
        action = preview(saved, POLICY, CONFIG, routing_history=history)[
            "actions"
        ][0]
        report = make_report(saved, POLICY, routing_history=history)["items"][
            0
        ]
        self.assertEqual(
            action["last_author_activity_at"], data["observed_at"]
        )
        for field in ("last_author_activity_at", "age_days", "overdue"):
            self.assertEqual(action[field], report[field])
        later = make_report(saved, POLICY, "2026-11-24T12:00:00Z", history)[
            "items"
        ][0]
        self.assertAlmostEqual(later["age_days"], 1 + 60 / 86400)
        standalone = make_report(saved, POLICY)["items"][0]
        self.assertTrue(
            any(
                "No routing history" in message
                for message in standalone["uncertainty"]
            )
        )

    def test_unrequested_changes_reviewer_keeps_author_owner_after_reply(self):
        data = python_item()
        review(data, "CHANGES_REQUESTED", who="unsolicited")
        first = preview(snapshot(data), POLICY, CONFIG)["actions"][0]
        data["timeline"].append(
            {
                "event": "commented",
                "created_at": NOW,
                "user": {"login": "author"},
            }
        )
        action = preview(
            snapshot(data), POLICY, CONFIG, routing_history=[first]
        )["actions"][0]
        self.assertEqual(
            (action["kind"], action["owner"]), ("Respond", "author")
        )
        self.assertEqual(action["key"], first["key"])
        self.assertEqual(action["last_author_activity_at"], NOW)
        self.assertEqual(action["notification"]["kind"], "none")

    def test_author_inactivity_escalates_after_strictly_more_than_fourteen_days(
        self,
    ):
        action = {
            "key": "key",
            "last_author_activity_at": "2026-11-01T12:00:00Z",
            "author_inactivity_days": 14,
        }
        self.assertEqual(
            notification_state(action, CONFIG, {}, "2026-11-15T12:00:00Z")[
                "kind"
            ],
            "none",
        )
        result = notification_state(action, CONFIG, {}, "2026-11-15T12:00:01Z")
        self.assertEqual(result["kind"], "escalation")
        history = {
            "key": {
                "first_notified_at": "2026-11-15T12:00:01Z",
                "last_notified_at": "2026-11-15T12:00:01Z",
                "escalated_at": "2026-11-15T12:00:01Z",
            }
        }
        self.assertEqual(
            notification_state(
                action, CONFIG, history, "2026-11-16T12:00:00Z"
            )["kind"],
            "none",
        )
        self.assertEqual(
            notification_state(
                action, CONFIG, history, "2026-11-19T12:00:00Z"
            )["kind"],
            "reminder",
        )
        action["last_author_activity_at"] = "2026-11-20T12:00:00Z"
        self.assertEqual(
            notification_state(
                action, CONFIG, history, "2026-11-25T12:00:00Z"
            )["kind"],
            "none",
        )
        self.assertEqual(
            notification_state(
                action, CONFIG, history, "2026-12-05T12:00:00Z"
            )["kind"],
            "escalation",
        )

    def test_reviewer_and_bot_activity_do_not_reset_author_inactivity(self):
        data = python_item()
        review(data, "CHANGES_REQUESTED")
        for who in ["reviewer", "robot[bot]"]:
            data["timeline"].append(
                {
                    "event": "commented",
                    "created_at": NOW,
                    "user": {"login": who},
                }
            )
        saved = snapshot(data)
        saved["collected_at"] = "2026-11-21T12:00:00Z"
        action = preview(saved, POLICY, CONFIG)["actions"][0]
        self.assertEqual(action["last_author_activity_at"], REVIEWED)
        self.assertEqual(action["notification"]["kind"], "escalation")
        self.assertIn("<@U01PXPJ5741>", action["slack_preview"])
        self.assertIn("author retains ownership", action["slack_preview"])
        self.assertEqual(notification_recipients(action, CONFIG), ["vyasr"])
        self.assertIn(
            "<https://github.com/author|@author>", action["slack_preview"]
        )

    def test_new_author_head_is_activity_once_at_first_observation(self):
        data = python_item()
        review(data, "CHANGES_REQUESTED")
        first = preview(snapshot(data), POLICY, CONFIG)["actions"][0]
        data["pull_request"]["head"]["sha"] = "new-head"
        data["head_commit_author"] = "author"
        saved = snapshot(data)
        saved["collected_at"] = "2026-11-21T12:00:00Z"
        action = preview(saved, POLICY, CONFIG, routing_history=[first])[
            "actions"
        ][0]
        self.assertEqual(action["key"], first["key"])
        self.assertEqual(
            action["last_author_activity_at"], saved["collected_at"]
        )
        later = saved["collected_at"]
        saved["collected_at"] = "2026-11-25T12:00:00Z"
        again = preview(saved, POLICY, CONFIG, routing_history=[action])[
            "actions"
        ][0]
        self.assertEqual(again["last_author_activity_at"], later)
        self.assertFalse(again["overdue"])
        data["head_commit_author"] = "robot[bot]"
        other = preview(saved, POLICY, CONFIG, routing_history=[first])[
            "actions"
        ][0]
        self.assertEqual(other["last_author_activity_at"], REVIEWED)

    def test_incomplete_reply_history_preserves_owner_but_prevents_escalation(
        self,
    ):
        data = python_item()
        review(data, "CHANGES_REQUESTED")
        del data["review_comments"]
        saved = snapshot(data)
        saved["collected_at"] = "2026-12-05T12:00:00Z"
        action = preview(saved, POLICY, CONFIG)["actions"][0]
        self.assertEqual(action["owner"], "author")
        self.assertIsNone(action["overdue"])
        self.assertEqual(action["notification"]["kind"], "none")

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
        later["collected_at"] = "2026-12-02T12:00:00Z"
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
