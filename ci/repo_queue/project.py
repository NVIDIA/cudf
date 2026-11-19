#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan or apply the configured GitHub Project fields and saved views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from collect import GitHub

ROOT = Path(__file__).parent
FIELD_SELECTION = """
... on ProjectV2FieldCommon { id name dataType }
... on ProjectV2SingleSelectField {
  options { id name color description }
}
"""
VIEW_SELECTION = """
id name number layout filter
configuration {
  visibleFields(first: 100) {
    nodes { ... on ProjectV2FieldCommon { id } }
    pageInfo { hasNextPage }
  }
}
"""


def connection(github, project_id, name, selection):
    cursor = None
    result = []
    while True:
        data = github.graphql(
            "query($id: ID!, $after: String) { node(id: $id) { "
            "... on ProjectV2 { " + name + "(first: 100, after: $after) { "
            "nodes { " + selection + " } "
            "pageInfo { hasNextPage endCursor } } } } }",
            {"id": project_id, "after": cursor},
        )["node"][name]
        result.extend(data["nodes"])
        if not data["pageInfo"]["hasNextPage"]:
            return result
        cursor = data["pageInfo"]["endCursor"]


def read_project(github, project):
    state = github.graphql(
        """query($id: ID!) { node(id: $id) { ... on ProjectV2 {
          id title url public shortDescription readme viewerCanUpdate
          owner { ... on Organization { login } }
        } } }""",
        {"id": project["id"]},
    )["node"]
    if not state or state["owner"]["login"] != project["owner"]:
        raise ValueError("Project identity does not match configured owner")
    if state["title"] != project["title"]:
        raise ValueError("Project title does not match configuration")
    state["fields"] = connection(
        github, project["id"], "fields", FIELD_SELECTION
    )
    state["views"] = connection(github, project["id"], "views", VIEW_SELECTION)
    return state


def field_input(spec, existing=None):
    result = {"name": spec["name"]}
    if existing and existing["dataType"] != spec["data_type"]:
        raise ValueError(f"Incompatible existing field: {spec['name']}")
    if not existing:
        result["dataType"] = spec["data_type"]
    if spec["data_type"] == "SINGLE_SELECT":
        old = {
            option["name"]: option
            for option in (existing or {}).get("options", [])
        }
        options, used = [], set()
        for option in spec["options"]:
            previous = old.get(option["name"]) or old.get(
                option.get("previous_name")
            )
            value = {
                "name": option["name"],
                "color": option["color"],
                "description": option.get("description", ""),
            }
            if previous:
                value["id"] = previous["id"]
                used.add(previous["id"])
            options.append(value)
        # Retaining IDs and additional options preserves existing card values.
        options.extend(
            option for option in old.values() if option["id"] not in used
        )
        result["singleSelectOptions"] = options
    return result


def field_matches(spec, existing):
    if not existing:
        return False
    desired = field_input(spec, existing)
    if spec["data_type"] != "SINGLE_SELECT":
        return True
    actual = {option["id"]: option for option in existing.get("options", [])}
    return all(
        "id" in option and actual.get(option["id"]) == option
        for option in desired["singleSelectOptions"]
    )


def view_for(spec, views):
    matches = [view for view in views if view["name"] == spec["name"]]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous duplicate view: {spec['name']}")
    if matches:
        return matches[0]
    return next(
        (
            view
            for view in views
            if view["name"] == spec.get("adopt_default") and not view["filter"]
        ),
        None,
    )


def view_matches(spec, view, field_ids):
    if not view or any(
        view[key] != spec[key] for key in ["name", "layout", "filter"]
    ):
        return False
    visible = view["configuration"]["visibleFields"]
    return (
        not visible["pageInfo"]["hasNextPage"]
        and [field["id"] for field in visible["nodes"]] == field_ids
    )


def plan_setup(state, schema, readme):
    changes = []
    if (
        state["readme"] != readme
        or state["shortDescription"] != schema["short_description"]
    ):
        changes.append(
            {"kind": "metadata", "name": "Project description and README"}
        )
    fields = {field["name"]: field for field in state["fields"]}
    for spec in schema["fields"]:
        if not field_matches(spec, fields.get(spec["name"])):
            changes.append({"kind": "field", "name": spec["name"]})
    visible = [
        fields.get(name, {}).get("id") for name in schema["visible_fields"]
    ]
    for spec in schema["views"]:
        if not view_matches(spec, view_for(spec, state["views"]), visible):
            changes.append({"kind": "view", "name": spec["name"]})
    return changes


def apply_setup(github, project, state, schema, readme):
    if not state["viewerCanUpdate"]:
        raise ValueError(
            "The current GitHub account cannot update this Project"
        )
    if (
        state["readme"] != readme
        or state["shortDescription"] != schema["short_description"]
    ):
        github.graphql(
            """mutation($input: UpdateProjectV2Input!) {
              updateProjectV2(input: $input) { projectV2 { id } }
            }""",
            {
                "input": {
                    "projectId": project["id"],
                    "readme": readme,
                    "shortDescription": schema["short_description"],
                }
            },
        )
    fields = {field["name"]: field for field in state["fields"]}
    for spec in schema["fields"]:
        existing = fields.get(spec["name"])
        if field_matches(spec, existing):
            continue
        value = field_input(spec, existing)
        if existing:
            value["fieldId"] = existing["id"]
            mutation, input_type = (
                "updateProjectV2Field",
                "UpdateProjectV2FieldInput",
            )
        else:
            value["projectId"] = project["id"]
            mutation, input_type = (
                "createProjectV2Field",
                "CreateProjectV2FieldInput",
            )
        response = github.graphql(
            "mutation($input: "
            + input_type
            + "!) { "
            + mutation
            + "(input: $input) { projectV2Field { "
            + FIELD_SELECTION
            + " } } }",
            {"input": value},
        )
        fields[spec["name"]] = response[mutation]["projectV2Field"]
    visible = [fields[name]["id"] for name in schema["visible_fields"]]
    for spec in schema["views"]:
        view = view_for(spec, state["views"])
        if view_matches(spec, view, visible):
            continue
        if not view:
            view = github.graphql(
                """mutation($input: CreateProjectV2ViewInput!) {
                  createProjectV2View(input: $input) {
                    projectV2View { id }
                  }
                }""",
                {
                    "input": {
                        "projectId": project["id"],
                        "name": spec["name"],
                        "layout": spec["layout"],
                    }
                },
            )["createProjectV2View"]["projectV2View"]
        github.graphql(
            """mutation($input: UpdateProjectV2ViewInput!) {
              updateProjectV2View(input: $input) { projectV2View { id } }
            }""",
            {
                "input": {
                    "viewId": view["id"],
                    "name": spec["name"],
                    "layout": spec["layout"],
                    "filter": spec["filter"],
                    "configuration": {"visibleFieldIds": visible},
                }
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--schema", type=Path, default=ROOT / "project_schema.json"
    )
    parser.add_argument("--readme", type=Path, default=ROOT / "PROJECT.md")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    schema = json.loads(args.schema.read_text())
    readme = args.readme.read_text()
    github = GitHub()
    state = read_project(github, config["project"])
    changes = plan_setup(state, schema, readme)
    if args.apply:
        apply_setup(github, config["project"], state, schema, readme)
        state = read_project(github, config["project"])
        remaining = plan_setup(state, schema, readme)
        if remaining:
            raise RuntimeError(f"Project verification failed: {remaining}")
    output = {"applied": args.apply, "changes": changes, "project": state}
    content = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
