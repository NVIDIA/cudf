# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression cases for ambiguous handoffs and misleading activity clocks."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from report import classify, elapsed_days, make_report, render_markdown

POLICY = json.loads(Path(__file__).with_name("policy.json").read_text())
NOW = "2026-11-16T12:00:00Z"
CREATED = "2026-11-02T12:00:00Z"
REQUESTED = "2026-11-05T12:00:00Z"
REVIEWED = "2026-11-06T12:00:00Z"


def item():
    return {
        "issue": {
            "number": 1,
            "html_url": "https://github.com/NVIDIA/cudf/pull/1",
            "title": "Example",
            "created_at": CREATED,
            "state": "open",
            "updated_at": NOW,
            "user": {"login": "author"},
            "labels": [{"name": "libcudf"}],
        },
        "pull_request": {
            "state": "open",
            "draft": False,
            "requested_reviewers": [],
            "requested_teams": [],
            "head": {"sha": "head"},
            "base": {"repo": {"owner": {"login": "NVIDIA"}}},
            "mergeable": True,
        },
        "timeline": [],
        "reviews": [],
        "review_comments": [],
        "check_runs": [],
        "statuses": [],
        "warnings": [],
    }


def request(data, who="reviewer", at=REQUESTED):
    data["pull_request"]["requested_reviewers"].append({"login": who})
    data["timeline"].append(
        {
            "event": "review_requested",
            "created_at": at,
            "requested_reviewer": {"login": who},
        }
    )


def review(data, state, who="reviewer", at=REVIEWED):
    data["reviews"].append(
        {
            "state": state,
            "user": {"login": who},
            "submitted_at": at,
            "html_url": "https://github.com/NVIDIA/cudf/pull/1#review",
            "commit_id": "head",
        }
    )


class ReportTests(unittest.TestCase):
    def classify(self, data, policy=POLICY):
        return classify(data, policy, NOW)

    def test_bot_activity_does_not_refresh_wait(self):
        data = item()
        request(data)
        review(data, "APPROVED", "coderabbitai[bot]", NOW)
        data["timeline"].append(
            {
                "event": "commented",
                "created_at": NOW,
                "user": {"login": "rapids-bot"},
            }
        )
        row = self.classify(data)
        self.assertEqual(row["queue"], "Waiting for first review")
        self.assertEqual(row["since"], REQUESTED)
        self.assertTrue(row["overdue"])

    def test_rerequest_starts_new_wait(self):
        data = item()
        review(data, "CHANGES_REQUESTED")
        request(data, at="2026-11-16T11:00:00Z")
        row = self.classify(data)
        self.assertEqual(row["queue"], "Waiting for another review")
        self.assertFalse(row["overdue"])

    def test_old_removed_request_is_not_reused(self):
        data = item()
        request(data)
        data["timeline"].append(
            {
                "event": "review_request_removed",
                "created_at": REVIEWED,
                "requested_reviewer": {"login": "reviewer"},
            }
        )
        row = self.classify(data)
        self.assertIsNone(row["since"])
        self.assertIsNone(row["overdue"])

    def test_multiple_reviewers_keep_individual_clocks(self):
        data = item()
        request(data)
        request(data, "other", NOW)
        row = self.classify(data)
        self.assertEqual(row["since"], REQUESTED)
        self.assertEqual(len(row["review_requests"]), 2)
        data["timeline"].pop()
        self.assertIsNone(self.classify(data)["age_days"])

    def test_team_review_request(self):
        data = item()
        data["pull_request"]["requested_teams"] = [
            {"slug": "cudf-cpp-codeowners"}
        ]
        data["timeline"] = [
            {
                "event": "review_requested",
                "created_at": REQUESTED,
                "requested_team": {"slug": "cudf-cpp-codeowners"},
            }
        ]
        row = self.classify(data)
        self.assertEqual(row["next_actor"], "@NVIDIA/cudf-cpp-codeowners")
        self.assertEqual(row["since"], REQUESTED)

    def test_changes_requested_is_author_handoff(self):
        data = item()
        review(data, "CHANGES_REQUESTED")
        row = self.classify(data)
        self.assertEqual(row["queue"], "Waiting for author")
        self.assertEqual(row["next_actor"], "@author")

    def test_rerequest_to_other_reviewer_does_not_clear_author_handoff(self):
        data = item()
        review(data, "CHANGES_REQUESTED")
        request(data, "other", NOW)
        self.assertEqual(self.classify(data)["queue"], "Waiting for author")

    def test_author_response_retains_ownership_and_resets_inactivity(self):
        data = item()
        review(data, "CHANGES_REQUESTED")
        data["timeline"].append(
            {
                "event": "commented",
                "created_at": NOW,
                "user": {"login": "author"},
            }
        )
        row = self.classify(data)
        self.assertEqual(row["queue"], "Waiting for author")
        self.assertEqual(row["next_actor"], "@author")
        self.assertEqual(row["since"], REVIEWED)
        self.assertEqual(row["last_author_activity_at"], NOW)
        self.assertEqual(row["age_days"], 0)

    def test_inline_author_response_retains_ownership(self):
        data = item()
        review(data, "CHANGES_REQUESTED")
        data["review_comments"] = [
            {"created_at": NOW, "user": {"login": "author"}}
        ]
        self.assertEqual(self.classify(data)["queue"], "Waiting for author")
        self.assertEqual(self.classify(data)["last_author_activity_at"], NOW)
        del data["review_comments"]
        row = self.classify(data)
        self.assertEqual(row["queue"], "Waiting for author")
        self.assertTrue(row["uncertainty"])

    def test_later_commit_does_not_establish_push_or_rereview_clock(self):
        data = item()
        review(data, "CHANGES_REQUESTED")
        data["timeline"].append(
            {
                "event": "committed",
                "committer": {"date": NOW},
            }
        )
        row = self.classify(data)
        self.assertEqual(row["queue"], "Waiting for author")
        self.assertEqual(row["last_author_activity_at"], REVIEWED)

    def test_comment_does_not_revoke_approval_but_dismissal_does(self):
        data = item()
        review(data, "APPROVED")
        review(data, "COMMENTED", at=NOW)
        self.assertEqual(
            self.classify(data)["queue"], "Has approvals; inspect blockers"
        )
        data["reviews"][0]["state"] = "DISMISSED"
        self.assertEqual(self.classify(data)["queue"], "Unclear handoff")

    def test_approval_is_not_assumed_valid_for_new_head(self):
        data = item()
        review(data, "APPROVED")
        data["pull_request"]["head"]["sha"] = "new-head"
        self.assertIn(
            "Some approvals reference a different head commit",
            self.classify(data)["uncertainty"],
        )

    def test_draft_is_separate_from_deferred(self):
        data = item()
        data["pull_request"]["draft"] = True
        self.assertEqual(self.classify(data)["queue"], "Draft")
        self.assertIsNone(self.classify(data)["overdue"])

    def test_ready_event_sets_assignment_clock(self):
        data = item()
        data["timeline"] = [
            {"event": "convert_to_draft", "created_at": REQUESTED},
            {"event": "ready_for_review", "created_at": NOW},
        ]
        row = self.classify(data)
        self.assertEqual(row["queue"], "Needs reviewer assignment")
        self.assertEqual(row["since"], NOW)

    def test_draft_time_is_excluded_from_outstanding_review_wait(self):
        data = item()
        request(data)
        data["timeline"].extend(
            [
                {"event": "convert_to_draft", "created_at": REVIEWED},
                {"event": "ready_for_review", "created_at": NOW},
            ]
        )
        row = self.classify(data)
        self.assertEqual(row["since"], NOW)
        self.assertFalse(row["overdue"])

    def test_triage_reapplied_label_uses_latest_episode(self):
        data = item()
        del data["pull_request"]
        data["issue"]["labels"] = [{"name": POLICY["triage_label"]}]
        data["timeline"] = [
            {
                "event": kind,
                "created_at": at,
                "label": {"name": POLICY["triage_label"]},
            }
            for kind, at in [
                ("labeled", CREATED),
                ("unlabeled", REQUESTED),
                ("labeled", NOW),
            ]
        ]
        row = self.classify(data)
        self.assertEqual(row["since"], NOW)
        self.assertFalse(row["overdue"])
        self.assertEqual(row["next_actor"], "Triage owner unassigned")

    def test_missing_label_event_does_not_use_updated_or_created(self):
        data = item()
        del data["pull_request"]
        data["issue"]["labels"] = [{"name": POLICY["triage_label"]}]
        self.assertIsNone(self.classify(data)["since"])

    def test_blockers_do_not_replace_lifecycle(self):
        data = item()
        request(data)
        data["pull_request"]["mergeable"] = False
        data["statuses"] = [
            {"context": "CI", "state": "pending", "created_at": NOW},
            {"context": "CI", "state": "failure", "created_at": CREATED},
        ]
        row = self.classify(data)
        self.assertEqual(row["queue"], "Waiting for first review")
        self.assertEqual(
            row["blockers"],
            [
                "Merge conflict reported by GitHub",
                "Observed CI pending: CI",
            ],
        )

    def test_business_time_excludes_weekend_and_keeps_fraction(self):
        self.assertEqual(elapsed_days("2026-11-13T12:00:00Z", NOW, True), 1.0)
        self.assertEqual(elapsed_days("2026-11-13T12:00:00Z", NOW), 3.0)

    def test_policy_overrides_and_input_are_not_mutated(self):
        data = item()
        policy = deepcopy(POLICY)
        policy["waiting_for_author_labels"] = ["author-action"]
        data["issue"]["labels"].append({"name": "author-action"})
        before, before_policy = deepcopy(data), deepcopy(policy)
        row = self.classify(data, policy)
        self.assertEqual(row["queue"], "Waiting for author")
        self.assertEqual(data, before)
        self.assertEqual(policy, before_policy)

    def test_replay_is_deterministic_and_escapes_titles(self):
        data = item()
        data["issue"]["title"] = "A | B\n<script>x</script> [link](bad)"
        snapshot = {
            "schema_version": 1,
            "repository": "NVIDIA/cudf",
            "collected_at": NOW,
            "selection": {"mode": "sample"},
            "items": [data],
        }
        first = make_report(snapshot, POLICY)
        self.assertEqual(first, make_report(snapshot, POLICY))
        output = render_markdown(first)
        self.assertNotIn("<script>", output)
        self.assertIn("A \\| B", output)
        with self.assertRaises(ValueError):
            make_report(snapshot, POLICY, CREATED)


if __name__ == "__main__":
    unittest.main()
