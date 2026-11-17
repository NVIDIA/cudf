#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preview independently owned Project actions and Slack notifications offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

from report import (
    classify,
    elapsed_days,
    escape,
    event_link,
    human_reviews,
    parse_time,
    ready_since,
    requested_key,
    review_requests,
)

ROOT = Path(__file__).parent
QUEUE_ACTIONS = {
    "Needs triage": (
        "Triage",
        "Decide component, disposition, and next owner",
    ),
    "Needs reviewer assignment": (
        "Assign reviewer",
        "Request a named reviewer",
    ),
    "Waiting for author": ("Respond", "Address the requested author action"),
    "Has approvals; inspect blockers": (
        "Inspect blockers",
        "Establish remaining requirements and the next handoff",
    ),
    "Unclear handoff": (
        "Clarify handoff",
        "Establish the next action and name its owner",
    ),
}


def add_business_days(start, days):
    current = parse_time(start) + timedelta(days=days)
    missing = days - elapsed_days(start, current.isoformat(), True)
    while missing > 1e-9:
        current += timedelta(days=missing)
        missing = days - elapsed_days(start, current.isoformat(), True)
    return current.isoformat()


def notification_state(action, config, history, now):
    """Derive a proposal without recording an attempted delivery as success."""
    if not action["overdue"]:
        return {"kind": "none", "follow_up": None, "escalate_on": None}
    record = history.get(action["key"], {})
    first = record.get("first_notified_at")
    if not first:
        return {"kind": "initial", "follow_up": None, "escalate_on": None}
    policy = config["follow_up"]
    escalation = parse_time(first) + timedelta(
        days=policy["escalate_after_calendar_days"]
    )
    last = record.get("last_notified_at", first)
    follow_up = add_business_days(last, policy["repeat_after_business_days"])
    kind = "none"
    if parse_time(now) >= escalation and not record.get("escalated_at"):
        kind = "escalation"
    elif parse_time(now) >= parse_time(follow_up):
        kind = "reminder"
    return {
        "kind": kind,
        "follow_up": follow_up,
        "escalate_on": escalation.isoformat(),
    }


def make_action(
    item,
    row,
    config,
    policy,
    now,
    kind,
    owner,
    since,
    evidence,
    description,
    queue=None,
    discriminator="",
):
    issue = item["issue"]
    identity = [
        issue["html_url"],
        kind,
        owner.casefold(),
        since,
        discriminator,
    ]
    key = (
        "cudf-action:v1:"
        + hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode()
        ).hexdigest()
    )
    threshold = policy["thresholds"].get(queue or row["queue"])
    if kind == "Clarify handoff":
        age, overdue = None, True
    else:
        age = (
            elapsed_days(since, now, threshold["business_days"])
            if since and threshold
            else None
        )
        overdue = age > threshold["days"] if age is not None else None
    attention = (
        "Needs routing"
        if kind == "Clarify handoff"
        else "Unknown age"
        if overdue is None
        else "Overdue"
        if overdue
        else "On track"
    )
    return {
        "key": key,
        "source": issue["html_url"],
        "number": issue["number"],
        "title": f"{kind} #{issue['number']}: {issue['title']}",
        "source_sha": (item.get("pull_request") or {})
        .get("head", {})
        .get("sha"),
        "kind": kind,
        "description": description,
        "owner": owner,
        "component": config["scope"]["component"],
        "status": "Awaiting action",
        "since": since,
        "age_days": age,
        "overdue": overdue,
        "attention": attention,
        "handoff_evidence": evidence,
        "observations": row["blockers"],
        "uncertainty": row["uncertainty"],
        "project_fields": {
            "Status": "Awaiting action",
            "Next action": kind,
            "Component": config["scope"]["component"],
            "Attention": attention,
            "Waiting since": parse_time(since).date().isoformat()
            if since
            else None,
            "Source": issue["html_url"],
            "Handoff evidence": evidence,
            "Action key": key,
        },
    }


def candidates(item, config, policy, now):
    issue = item["issue"]
    labels = {label["name"] for label in issue["labels"]}
    if not labels.intersection(config["scope"]["labels"]):
        return [], "No confirmed Python component label"
    row = classify(item, policy, now)
    if row["queue"] in {
        "Closed since selection",
        "Draft",
        "Explicitly deferred",
        "Outside intake queue",
    }:
        return [], row["queue"]
    primary = config["primary_contact"]
    result = []
    pr = item.get("pull_request")
    if pr:
        events = sorted(
            (event for event in item["timeline"] if event.get("created_at")),
            key=lambda event: parse_time(event["created_at"]),
        )
        requests = review_requests(pr, events, policy)
        individuals = [
            request for request in requests if request[0].startswith("user:")
        ]
        for identity, actor, since in requests:
            team = (
                identity.split(":", 1)[1]
                if identity.startswith("team:")
                else None
            )
            if team and (
                individuals or team not in config["scope"]["team_slugs"]
            ):
                continue
            ready = ready_since(issue, events)
            event_date = since
            since = (
                max(since, ready, key=parse_time) if since and ready else None
            )
            kind = "Assign reviewer" if team else "Review"
            queue = (
                "Needs reviewer assignment"
                if team
                else "Waiting for another review"
                if human_reviews(item, policy)
                else "Waiting for first review"
            )
            request_events = [
                event
                for event in events
                if event["event"] == "review_requested"
                and requested_key(event) == identity
                and event["created_at"] == event_date
            ]
            evidence = event_link(issue, request_events, event_date)
            result.append(
                make_action(
                    item,
                    row,
                    config,
                    policy,
                    now,
                    kind,
                    primary if team else actor.lstrip("@"),
                    since,
                    evidence,
                    f"Select a named reviewer for {actor}"
                    if team
                    else "Submit the requested review or explicitly hand it off",
                    queue=queue,
                    discriminator=identity if team else "",
                )
            )
    if row["queue"] in QUEUE_ACTIONS:
        kind, description = QUEUE_ACTIONS[row["queue"]]
        owner = issue["user"]["login"] if kind == "Respond" else primary
        evidence = (
            row["evidence"][0]["url"] if row["evidence"] else issue["html_url"]
        )
        result.append(
            make_action(
                item,
                row,
                config,
                policy,
                now,
                kind,
                owner,
                row["since"],
                evidence,
                description,
            )
        )
    return result, None


def slack_text(action, config, project_state):
    def contact(login):
        users = {
            name.casefold(): member
            for name, member in config["slack"]["users"].items()
        }
        member = users.get(login.casefold())
        return (
            f"<@{member}>"
            if member
            else f"[Slack mapping needed: GitHub {login}]"
        )

    mention = contact(action["owner"])
    if (
        action["notification"]["kind"] == "escalation"
        and action["owner"].casefold()
        != config["escalation_contact"].casefold()
    ):
        mention += " " + contact(config["escalation_contact"])
    base = config["project"]["url"]
    views = {
        view["name"]: f"{base}/views/{view['number']}"
        for view in (project_state or {}).get("views", [])
    }
    links = [
        f"<{action['source']}|Source issue/PR>",
        f"<{views.get('My actions', base)}|My outstanding actions>",
        f"<{views.get('Outstanding actions', base)}|Team queue>",
    ]
    kind = action["notification"]["kind"]
    lines = [
        f"{mention} — {action['kind']} needed for {config['repository']} #{action['number']}.",
        action["description"] + ".",
        f"Handoff evidence: <{action['handoff_evidence']}|GitHub>",
    ]
    if action["since"]:
        lines.append(f"Waiting since {action['since']}.")
    if kind == "escalation":
        lines.append(
            "One week has elapsed since the first notification without the required action."
        )
    lines.extend(
        [
            " · ".join(links),
            "Take the action or explicitly hand it off on GitHub; GitHub activity updates tracking.",
        ]
    )
    return "\n".join(lines)


def preview(snapshot, policy, config, project_state=None, history=None):
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema_version")
    if snapshot["repository"] != config["repository"]:
        raise ValueError("snapshot repository does not match pilot")
    policy = {**policy, "triage_owner": "@" + config["primary_contact"]}
    now = snapshot["collected_at"]
    actions, omitted = [], []
    for item in snapshot["items"]:
        selected, reason = candidates(item, config, policy, now)
        actions.extend(selected)
        if reason:
            omitted.append(
                {"number": item["issue"]["number"], "reason": reason}
            )
    missing = set()
    for action in actions:
        action["notification"] = notification_state(
            action, config, history or {}, now
        )
        for field, key in [
            ("Follow-up", "follow_up"),
            ("Escalate on", "escalate_on"),
        ]:
            value = action["notification"][key]
            action["project_fields"][field] = (
                parse_time(value).date().isoformat() if value else None
            )
        if action["owner"].casefold() not in {
            name.casefold() for name in config["slack"]["users"]
        }:
            missing.add(action["owner"])
        action["slack_preview"] = slack_text(action, config, project_state)
    return {
        "mode": "preview; no Project cards or Slack messages written",
        "as_of": now,
        "scope": config["scope"],
        "selection": snapshot["selection"],
        "actions": actions,
        "omitted": omitted,
        "missing_slack_mappings": sorted(missing),
    }


def render_markdown(plan):
    lines = [
        "# Python action preview",
        "",
        plan["mode"],
        "",
        f"As of {plan['as_of']}; {len(plan['actions'])} proposed action cards.",
        "",
        "Candidates come from the saved sample, not the complete Python backlog. "
        "Unlabeled issues require component classification before inclusion.",
        "",
        "The escalation clock has not started for unsent notifications. "
        "Slack mappings must be resolved before delivery.",
        "",
        "| Action | Owner | Waiting since | Attention | Notification proposal |",
        "| --- | --- | --- | --- | --- |",
    ]
    for action in plan["actions"]:
        lines.append(
            f"| [{escape(action['title'])}]({action['source']}) | "
            f"{escape(action['owner'])} | {action['since'] or 'Unknown'} | "
            f"{action['attention']} | {action['notification']['kind']} |"
        )
    lines.extend(["", "## Slack message previews", ""])
    for action in plan["actions"]:
        if action["notification"]["kind"] != "none":
            lines.extend(
                [
                    f"### {escape(action['title'])}",
                    "",
                    "```text",
                    action["slack_preview"],
                    "```",
                    "",
                ]
            )
    lines.extend(["## Omitted candidates", ""])
    lines.extend(
        f"- #{item['number']}: {item['reason']}" for item in plan["omitted"]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "pilot.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "policy.json")
    parser.add_argument("--project-state", type=Path)
    parser.add_argument(
        "--history",
        type=Path,
        help="Successful notification records keyed by action key",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--format", choices=["json", "markdown"], default="markdown"
    )
    args = parser.parse_args()
    state = (
        json.loads(args.project_state.read_text())
        if args.project_state
        else None
    )
    plan = preview(
        json.loads(args.snapshot.read_text()),
        json.loads(args.policy.read_text()),
        json.loads(args.config.read_text()),
        state.get("project", state) if state else None,
        json.loads(args.history.read_text()) if args.history else None,
    )
    content = (
        json.dumps(plan, indent=2) + "\n"
        if args.format == "json"
        else render_markdown(plan)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
