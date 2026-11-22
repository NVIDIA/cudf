#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronize managed draft cards while preserving notes and delivery records."""

from __future__ import annotations

import base64
import json
import re
from copy import deepcopy

from project import connection
from report import parse_time

START = "<!-- cudf-work-tracking:start -->"
END = "<!-- cudf-work-tracking:end -->"
MARKER = re.compile(r"<!-- cudf-work-tracking:v1:([A-Za-z0-9+/=]+) -->")
ITEM_SELECTION = """
id fullDatabaseId isArchived
content { ... on DraftIssue {
  id title body assignees(first:100) {
    nodes { id login } pageInfo { hasNextPage }
  }
} }
fieldValues(first:100) {
  nodes {
    ... on ProjectV2ItemFieldTextValue {
      text field { ... on ProjectV2FieldCommon { name } }
    }
    ... on ProjectV2ItemFieldSingleSelectValue {
      name field { ... on ProjectV2FieldCommon { name } }
    }
    ... on ProjectV2ItemFieldDateValue {
      date field { ... on ProjectV2FieldCommon { name } }
    }
  }
  pageInfo { hasNextPage }
}
"""


def metadata(body):
    matches = MARKER.findall(body)
    if not matches:
        if START in body or END in body or "<!-- cudf-work-tracking:" in body:
            raise ValueError("Missing or damaged managed card metadata")
        return None
    if len(matches) != 1 or body.count(START) != 1 or body.count(END) != 1:
        raise ValueError("Ambiguous managed card metadata")
    marker = MARKER.search(body)
    if not body.index(START) < marker.start() < marker.end() < body.index(END):
        raise ValueError("Metadata marker must be inside the managed block")
    data = json.loads(base64.b64decode(matches[0], validate=True))
    validate_metadata(data)
    return data


def validate_metadata(data):
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Unsupported managed card metadata")
    action = data.get("action")
    required = (
        "key",
        "source",
        "title",
        "kind",
        "owner",
        "component",
        "description",
        "handoff_evidence",
    )
    if not isinstance(action, dict) or any(
        not isinstance(action.get(name), str) for name in required
    ):
        raise ValueError("Invalid managed action fields")
    if (
        not action["key"].startswith("cudf-action:v1:")
        or type(action.get("number")) is not int
    ):
        raise ValueError("Invalid managed action identity")
    if (
        "since" not in action
        or "source_sha" not in action
        or (
            action["source_sha"] is not None
            and not isinstance(action["source_sha"], str)
        )
    ):
        raise ValueError("Invalid managed source or waiting time")
    for name in ("since", "last_author_activity_at", "source_observed_at"):
        if action.get(name) is not None:
            parse_time(action[name])
    for name in ("observations", "uncertainty"):
        if not isinstance(action.get(name), list) or any(
            not isinstance(v, str) for v in action[name]
        ):
            raise ValueError("Invalid managed action notes")
    if "author_inactivity_days" in action and (
        type(action["author_inactivity_days"]) is not int
        or action["author_inactivity_days"] < 1
    ):
        raise ValueError("Invalid author inactivity threshold")
    for name in ("resolved_at", "closed_at"):
        if name in data:
            parse_time(data[name])
    for name in ("written_status", "manual_status", "resolution"):
        if name in data and not isinstance(data[name], str):
            raise ValueError("Invalid managed disposition")
    deliveries = data.get("deliveries")
    if not isinstance(deliveries, dict):
        raise ValueError("Invalid managed deliveries")
    for route, record in deliveries.items():
        if not isinstance(route, str) or not isinstance(record, dict):
            raise ValueError("Invalid managed delivery record")
        for name in ("first_notified_at", "last_notified_at", "escalated_at"):
            if name in record:
                parse_time(record[name])
        for name in ("thread_ts", "channel"):
            if name in record and not isinstance(record[name], str):
                raise ValueError("Invalid delivery thread")
        if "pending" in record:
            pending = record["pending"]
            if not isinstance(pending, dict) or any(
                not isinstance(pending.get(k), str)
                for k in ("id", "kind", "prepared_at")
            ):
                raise ValueError("Invalid pending delivery")
            parse_time(pending["prepared_at"])
    statuses = {"Awaiting action", "In progress", "Deferred", "Complete"}
    for name in ("written_status", "manual_status"):
        if name in data and data[name] not in statuses:
            raise ValueError("Invalid managed Status")
    transition = data.get("status_transition")
    if transition is not None and (
        not isinstance(transition, dict)
        or "from" not in transition
        or transition.get("to") not in statuses
        or (
            transition.get("from") is not None
            and transition["from"] not in statuses
        )
    ):
        raise ValueError("Invalid pending Status transition")


def managed_body(data, previous=""):
    action = data["action"]
    encoded = base64.b64encode(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    lines = [
        START,
        f"**Next owner:** @{action['owner']}",
        f"**Action:** {action['description']}.",
        f"**Source:** {action['source']}",
        f"**Handoff evidence:** {action['handoff_evidence']}",
        f"**Waiting since:** {action['since'] or 'Unknown'}",
        f"**Last author activity:** {action.get('last_author_activity_at') or 'Not applicable / unknown'}",
        "",
        "DM preview: notifications go only to the primary contact. "
        "DM delivery history does not start production escalation clocks.",
    ]
    lines.extend(f"- {note}" for note in action.get("observations", []))
    lines.extend(
        f"- Uncertainty: {note}" for note in action.get("uncertainty", [])
    )
    if data.get("resolution"):
        lines.extend(["", "**Disposition:** " + data["resolution"]])
    lines.extend(["", f"<!-- cudf-work-tracking:v1:{encoded} -->", END])
    block = "\n".join(lines)
    if START in previous and END in previous:
        prefix, rest = previous.split(START, 1)
        _, suffix = rest.split(END, 1)
        return prefix + block + suffix
    return block + "\n\n## Maintainer notes\n\n"


def action_record(action):
    # Live ages change every run; keep the durable identity and evidence instead.
    result = {
        key: deepcopy(action[key])
        for key in [
            "key",
            "source",
            "number",
            "title",
            "kind",
            "owner",
            "component",
            "since",
            "handoff_evidence",
            "description",
            "source_sha",
            "observations",
            "uncertainty",
        ]
    }
    if "source_observed_at" in action:
        result["source_observed_at"] = action["source_observed_at"]
    if "author_inactivity_days" in action:
        result["author_inactivity_days"] = action["author_inactivity_days"]
        result["last_author_activity_at"] = action.get(
            "last_author_activity_at"
        )
    return result


def disposition(old, item, current, omitted, config):
    issue = item["issue"]
    if issue["state"] == "closed":
        return "Complete", "Source closed or merged: " + issue["html_url"]
    labels = {label["name"] for label in issue["labels"]}
    if not labels.intersection(config["scope"]["labels"]):
        return None
    if omitted in {"Draft", "Explicitly deferred"}:
        return "Deferred", "Source is " + omitted.lower() + "."
    if old["kind"] == "Resolve conflicts":
        if item.get("pull_request", {}).get("mergeable") is True:
            return "Complete", (
                "GitHub reports the branch can be merged without conflicts; "
                "other merge requirements may remain. Evidence: "
                + issue["html_url"]
            )
        # A replacement reviewer action does not prove that an author fixed a conflict.
        return None
    if old["kind"] == "Review" and old["since"]:
        for review in reversed(item.get("reviews", [])):
            if (
                review["user"]["login"].casefold() == old["owner"].casefold()
                and review["state"] in {"APPROVED", "CHANGES_REQUESTED"}
                and parse_time(review["submitted_at"])
                > parse_time(old["since"])
            ):
                return "Complete", "Requested review submitted: " + review[
                    "html_url"
                ]
        for event in reversed(item["timeline"]):
            if (
                event.get("event") == "review_request_removed"
                and event.get("requested_reviewer", {})
                .get("login", "")
                .casefold()
                == old["owner"].casefold()
                and event.get("created_at")
                and parse_time(event["created_at"]) > parse_time(old["since"])
            ):
                return "Complete", "Review request withdrawn: " + event.get(
                    "html_url", issue["html_url"]
                )
    if current:
        return "Complete", (
            "Superseded by the current GitHub handoff; follow the replacement "
            "action card(s). Evidence: " + current[0]["handoff_evidence"]
        )
    return None


def resolution_time(data):
    # Legacy closure records did not distinguish closing a PR from fixing its conflicts.
    if (
        data["action"]["kind"] == "Resolve conflicts"
        and data.get("resolution", "").startswith("Source closed")
        and "closed_at" not in data
    ):
        return None
    return data.get("resolved_at")


def status_data(card=None):
    """Keep manual intent even while a source transition temporarily overrides it."""
    if card is None:
        return {"version": 1, "deliveries": {}}
    data = deepcopy(card["data"])
    if data.get("resolved_at") and resolution_time(data) is None:
        data.setdefault("written_status", "Complete")
        data["closed_at"] = data.pop("resolved_at")
    status = card["values"].get("Status")
    transition = data.get("status_transition")
    if transition and status in {transition["from"], transition["to"]}:
        return data
    data.pop("status_transition", None)
    written = data.get("written_status")
    # Earlier versions recorded automatic dispositions only as reasons.
    if written is None:
        if data.get("resolved_at"):
            written = "Complete"
        elif data.get("resolution") in {
            "Source is draft.",
            "Source is explicitly deferred.",
        }:
            written = "Deferred"
    if status != written:
        if status in {"In progress", "Deferred", "Complete"}:
            data["manual_status"] = status
        else:
            data.pop("manual_status", None)
    return data


class ProjectCards:
    def __init__(self, github, config, state, apply=False):
        self.github, self.config, self.state, self.apply = (
            github,
            config,
            state,
            apply,
        )
        self.fields = {field["name"]: field for field in state["fields"]}
        if apply and not state.get("viewerCanUpdate", True):
            raise ValueError(
                "The current GitHub account cannot update this Project"
            )
        self.cards, self.changes, self.users = {}, [], {}
        for item in connection(
            github, config["project"]["id"], "items", ITEM_SELECTION
        ):
            content = item.get("content") or {}
            values = {
                value["field"]["name"]: value.get(
                    "text", value.get("name", value.get("date"))
                )
                for value in item["fieldValues"]["nodes"]
                if value.get("field")
            }
            try:
                data = metadata(content.get("body", ""))
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                OverflowError,
            ) as error:
                raise ValueError(
                    f"Invalid metadata on Project item {item['id']}: {error}"
                ) from None
            if data is None:
                if str(values.get("Action key", "")).startswith(
                    "cudf-action:"
                ):
                    raise ValueError("Missing metadata for managed Action key")
                continue
            if (
                item["fieldValues"]["pageInfo"]["hasNextPage"]
                or content["assignees"]["pageInfo"]["hasNextPage"]
            ):
                raise ValueError("Incomplete managed card fields or assignees")
            key = data["action"]["key"]
            source = data["action"]["source"]
            if not source.startswith(
                "https://github.com/" + config["repository"] + "/"
            ):
                raise ValueError(
                    "Managed card repository does not match configuration"
                )
            if key in self.cards:
                raise ValueError("Duplicate managed action key: " + key)
            if values.get("Action key", key) != key:
                raise ValueError("Card body and Action key disagree")
            self.cards[key] = {"item": item, "data": data, "values": values}

    def refresh(self):
        self.cards = ProjectCards(
            self.github, self.config, self.state, self.apply
        ).cards

    def routing_history(self):
        result = []
        for card in self.cards.values():
            data = card["data"]
            result.append(
                {
                    **deepcopy(data["action"]),
                    "resolved_at": resolution_time(data),
                }
            )
        return result

    def mutate(self, name, input_type, value, selection):
        return self.github.graphql(
            f"mutation($input:{input_type}!) {{ {name}(input:$input) {{ {selection} }} }}",
            {"input": value},
        )[name]

    def user_id(self, login):
        if login not in self.users:
            user = self.github.graphql(
                "query($login:String!){user(login:$login){id}}",
                {"login": login},
            )["user"]
            if not user:
                raise ValueError("Unknown GitHub assignee: " + login)
            self.users[login] = user["id"]
        return self.users[login]

    def write_content(self, card, data, owner, title):
        assignee = self.user_id(owner)
        fresh = self.github.graphql(
            "query($id:ID!){node(id:$id){... on DraftIssue{id title body assignees(first:100){nodes{id login} pageInfo{hasNextPage}}}}}",
            {"id": card["item"]["content"]["id"]},
        )["node"]
        if (
            fresh is None
            or fresh["assignees"]["pageInfo"]["hasNextPage"]
            or metadata(fresh["body"]) != card["data"]
        ):
            raise RuntimeError(
                "Managed card changed during refresh; retry from a fresh read"
            )
        body = managed_body(data, fresh["body"])
        self.mutate(
            "updateProjectV2DraftIssue",
            "UpdateProjectV2DraftIssueInput",
            {
                "draftIssueId": fresh["id"],
                "title": title,
                "body": body,
                "assigneeIds": [assignee],
            },
            "draftIssue { id }",
        )
        card["item"]["content"] = {
            **fresh,
            "title": title,
            "body": body,
            "assignees": {
                "nodes": [{"login": owner}],
                "pageInfo": {"hasNextPage": False},
            },
        }
        card["data"] = deepcopy(data)

    def check_current_status(self, card):
        # GitHub offers no conditional field mutation, leaving a short read/write race.
        fresh = self.github.graphql(
            'query($id:ID!){node(id:$id){... on ProjectV2Item{fieldValueByName(name:"Status"){... on ProjectV2ItemFieldSingleSelectValue{name}}}}}',
            {"id": card["item"]["id"]},
        )["node"]
        if fresh is None:
            raise RuntimeError("Project item disappeared during refresh")
        status = (fresh["fieldValueByName"] or {}).get("name")
        if status != card["values"].get("Status"):
            raise RuntimeError(
                "Project Status changed during refresh; retry from a fresh read"
            )

    def save(self, key, data, values):
        card = self.cards.get(key)
        if card and card["item"]["isArchived"]:
            return
        previous_data = deepcopy(card["data"]) if card else None
        data = deepcopy(data)
        if self.apply and card and "Status" in values:
            self.check_current_status(card)
        if "Status" in values:
            current = (card or {}).get("values", {}).get("Status")
            if current != values["Status"]:
                data["status_transition"] = {
                    "from": current,
                    "to": values["Status"],
                }
            else:
                data.pop("status_transition", None)
        title = data["action"]["title"]
        old_content = card["item"]["content"] if card else {}
        body = managed_body(data, old_content.get("body", ""))
        owner = data["action"]["owner"]
        old_owners = {
            u["login"].casefold()
            for u in old_content.get("assignees", {}).get("nodes", [])
        }
        content_changed = (
            old_content.get("title") != title
            or old_content.get("body") != body
            or old_owners != {owner.casefold()}
        )
        changed = {
            name: value
            for name, value in values.items()
            if (card or {}).get("values", {}).get(name) != value
        }
        if not content_changed and not changed:
            return
        self.changes.append(
            {
                "key": key,
                "kind": "update" if card else "create",
                "title": title,
                "fields": list(changed),
            }
        )
        if not self.apply:
            return
        content = {
            "title": title,
            "body": body,
            "assigneeIds": [self.user_id(owner)],
        }
        if not card:
            result = self.mutate(
                "addProjectV2DraftIssue",
                "AddProjectV2DraftIssueInput",
                {"projectId": self.config["project"]["id"], **content},
                "projectItem { id fullDatabaseId content { ... on DraftIssue { id } } }",
            )["projectItem"]
            result.update(isArchived=False)
            card = {"item": result, "data": {}, "values": {}}
            self.cards[key] = card
        elif content_changed:
            self.write_content(card, data, owner, title)
        if not card.get("data"):
            card["item"]["content"].update(
                title=title,
                body=body,
                assignees={
                    "nodes": [{"login": owner}],
                    "pageInfo": {"hasNextPage": False},
                },
            )
            card["data"] = deepcopy(data)
        for name, value in changed.items():
            if name == "Status":
                try:
                    self.check_current_status(card)
                except RuntimeError:
                    # No Status mutation was attempted; avoid claiming a manual target as ours.
                    rollback = (
                        deepcopy(previous_data)
                        if previous_data is not None
                        else deepcopy(data)
                    )
                    if previous_data is None:
                        rollback.pop("status_transition", None)
                        rollback.pop("written_status", None)
                    self.write_content(
                        card,
                        rollback,
                        rollback["action"]["owner"],
                        rollback["action"]["title"],
                    )
                    raise
            field = self.fields[name]
            common = {
                "projectId": self.config["project"]["id"],
                "itemId": card["item"]["id"],
                "fieldId": field["id"],
            }
            if value is None:
                self.mutate(
                    "clearProjectV2ItemFieldValue",
                    "ClearProjectV2ItemFieldValueInput",
                    common,
                    "projectV2Item { id }",
                )
            else:
                if field["dataType"] == "SINGLE_SELECT":
                    encoded = {
                        "singleSelectOptionId": next(
                            option["id"]
                            for option in field["options"]
                            if option["name"] == value
                        )
                    }
                else:
                    encoded = {
                        "date"
                        if field["dataType"] == "DATE"
                        else "text": value
                    }
                self.mutate(
                    "updateProjectV2ItemFieldValue",
                    "UpdateProjectV2ItemFieldValueInput",
                    {**common, "value": encoded},
                    "projectV2Item { id }",
                )
            card["values"][name] = value

        if "Status" in values and card["data"].get("status_transition"):
            completed = deepcopy(card["data"])
            completed.pop("status_transition")
            self.write_content(card, completed, owner, title)

    def reconcile(self, plan, snapshot):
        desired = {action["key"]: action for action in plan["actions"]}
        for key, action in desired.items():
            previous = self.cards.get(key)
            data = status_data(previous)
            data["action"] = action_record(action)
            values = dict(action["project_fields"])
            values["Status"] = data.get("manual_status", values["Status"])
            data["written_status"] = values["Status"]
            data.pop("resolution", None)
            data.pop("resolved_at", None)
            data.pop("closed_at", None)
            self.save(key, data, values)
        sources = {
            item["issue"]["html_url"]: item for item in snapshot["items"]
        }
        omitted = {item["number"]: item["reason"] for item in plan["omitted"]}
        for key, card in list(self.cards.items()):
            old = card["data"]["action"]
            if key in desired or old["source"] not in sources:
                continue
            item = sources[old["source"]]
            conflict_cleared = (
                old["kind"] == "Resolve conflicts"
                and item["issue"]["state"] == "open"
                and item.get("pull_request", {}).get("mergeable") is True
                and not resolution_time(card["data"])
            )
            if (
                card["values"].get("Status") == "Complete"
                and not conflict_cleared
                and not card["data"].get("status_transition")
            ):
                continue
            current = [
                a for a in plan["actions"] if a["source"] == old["source"]
            ]
            result = disposition(
                old,
                sources[old["source"]],
                current,
                omitted.get(old["number"]),
                self.config,
            )
            if result:
                status, reason = result
                data = status_data(card)
                data["resolution"] = reason
                data["written_status"] = status
                if status == "Complete":
                    if conflict_cleared:
                        data["resolved_at"] = snapshot["collected_at"]
                    elif old["kind"] != "Resolve conflicts":
                        data.setdefault(
                            "resolved_at", snapshot["collected_at"]
                        )
                    if item["issue"]["state"] == "closed":
                        data.setdefault("closed_at", snapshot["collected_at"])
                self.save(
                    key,
                    data,
                    {"Status": status, "Follow-up": None, "Escalate on": None},
                )
