# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect field identities and make Project setup safe to rerun."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

from project import (
    apply_setup,
    connection,
    field_input,
    plan_setup,
    view_for,
)

SCHEMA = json.loads(
    Path(__file__).with_name("project_schema.json").read_text()
)


def configured_state():
    fields = [
        {"id": "title", "name": "Title", "dataType": "TITLE"},
        {"id": "assignees", "name": "Assignees", "dataType": "ASSIGNEES"},
    ]
    for index, spec in enumerate(SCHEMA["fields"]):
        value = field_input(spec)
        fields.append(
            {
                "id": str(index),
                "name": spec["name"],
                "dataType": spec["data_type"],
                "options": [
                    {**option, "id": f"{index}-{number}"}
                    for number, option in enumerate(
                        value.get("singleSelectOptions", [])
                    )
                ],
            }
        )
    by_name = {field["name"]: field for field in fields}
    visible = [by_name[name]["id"] for name in SCHEMA["visible_fields"]]
    views = [
        {
            **spec,
            "id": spec["name"],
            "configuration": {
                "visibleFields": {
                    "nodes": [{"id": value} for value in visible],
                    "pageInfo": {"hasNextPage": False},
                }
            },
        }
        for spec in SCHEMA["views"]
    ]
    return {
        "viewerCanUpdate": True,
        "fields": fields,
        "views": views,
        "readme": "readme",
        "shortDescription": SCHEMA["short_description"],
    }


class ProjectTests(unittest.TestCase):
    def test_renaming_status_preserves_native_option_ids(self):
        old = {
            "dataType": "SINGLE_SELECT",
            "options": [
                {
                    "id": "todo",
                    "name": "Todo",
                    "color": "GRAY",
                    "description": "",
                },
                {
                    "id": "done",
                    "name": "Done",
                    "color": "GREEN",
                    "description": "",
                },
            ],
        }
        value = field_input(SCHEMA["fields"][0], old)
        by_name = {
            option["name"]: option for option in value["singleSelectOptions"]
        }
        self.assertEqual(by_name["Awaiting action"]["id"], "todo")
        self.assertEqual(by_name["Complete"]["id"], "done")

    def test_custom_options_are_preserved_on_update(self):
        old = {
            "dataType": "SINGLE_SELECT",
            "options": [
                {
                    "id": "keep",
                    "name": "Custom state",
                    "color": "BLUE",
                    "description": "Keep this",
                }
            ],
        }
        value = field_input(SCHEMA["fields"][0], old)
        self.assertIn(old["options"][0], value["singleSelectOptions"])

    def test_configured_project_is_noop_without_any_mutation(self):
        state = configured_state()
        self.assertEqual(plan_setup(state, SCHEMA, "readme"), [])
        github = Mock()
        apply_setup(github, {"id": "project"}, state, SCHEMA, "readme")
        github.graphql.assert_not_called()

    def test_incompatible_field_type_is_rejected_during_planning(self):
        state = configured_state()
        state["fields"][2]["dataType"] = "TEXT"
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            plan_setup(state, SCHEMA, "readme")

    def test_missing_field_and_view_are_planned(self):
        state = configured_state()
        state["fields"] = [f for f in state["fields"] if f["name"] != "Source"]
        state["views"].pop()
        changes = plan_setup(state, SCHEMA, "readme")
        self.assertIn({"kind": "field", "name": "Source"}, changes)
        self.assertIn({"kind": "view", "name": "Action board"}, changes)

    def test_adopt_only_an_unfiltered_default_view(self):
        spec = SCHEMA["views"][0]
        default = {"name": "View 1", "filter": ""}
        self.assertIs(view_for(spec, [default]), default)
        default["filter"] = "assignee:someone"
        self.assertIsNone(view_for(spec, [default]))

    def test_schema_connections_are_paginated(self):
        github = Mock()
        github.graphql.side_effect = [
            {
                "node": {
                    "fields": {
                        "nodes": [1],
                        "pageInfo": {"hasNextPage": True, "endCursor": "next"},
                    }
                }
            },
            {
                "node": {
                    "fields": {
                        "nodes": [2],
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            },
        ]
        self.assertEqual(connection(github, "project", "fields", "id"), [1, 2])
        self.assertEqual(github.graphql.call_args[0][1]["after"], "next")

    def test_planning_does_not_modify_live_snapshot(self):
        state = configured_state()
        before = deepcopy(state)
        plan_setup(state, SCHEMA, "new readme")
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
