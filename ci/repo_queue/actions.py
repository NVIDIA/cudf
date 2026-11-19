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
    author_clock,
    classify,
    elapsed_days,
    escape,
    is_human,
    lifecycle,
    parse_time,
    queue_age,
    validate_history,
)
from storage import write_private

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
    if "author_inactivity_days" in action:
        since = action.get("last_author_activity_at")
        if not since:
            return {"kind": "none", "follow_up": None, "escalate_on": None}
        deadline = parse_time(since) + timedelta(
            days=action["author_inactivity_days"]
        )
        record = history.get(action["key"], {})
        escalated = record.get("escalated_at")
        last = record.get("last_notified_at")
        follow_up = (
            add_business_days(
                last, config["follow_up"]["repeat_after_business_days"]
            )
            if last and escalated and parse_time(escalated) > parse_time(since)
            else deadline.isoformat()
        )
        kind = "none"
        if parse_time(now) > deadline:
            if not escalated or parse_time(escalated) <= parse_time(since):
                kind = "escalation"
            elif follow_up and parse_time(now) >= parse_time(follow_up):
                kind = "reminder"
        return {
            "kind": kind,
            "follow_up": follow_up,
            "escalate_on": deadline.isoformat(),
        }
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
    elif kind in {"Respond", "Resolve conflicts"}:
        # Author aging uses confirmed activity retained across snapshots.
        age, overdue = None, None
    else:
        age, overdue = queue_age(since, now, threshold)
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


def candidates(item, config, policy, now, routing_history=None):
    issue = item["issue"]
    labels = {label["name"] for label in issue["labels"]}
    if not labels.intersection(config["scope"]["labels"]):
        return [], "No confirmed Python component label"
    evidence = lifecycle(item, policy)
    row = classify(item, policy, now, routing_history, evidence)
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
    previous_conflict = next(
        (
            action
            for action in routing_history or []
            if action["source"] == issue["html_url"]
            and action["kind"] == "Resolve conflicts"
            and not action.get("resolved_at")
            and action["owner"].casefold() == issue["user"]["login"].casefold()
        ),
        None,
    )
    author = (
        issue["user"]["login"] if is_human(issue["user"], policy) else None
    )
    author_conflict = (
        pr is not None
        and author is not None
        and policy.get("routing", {}).get("merge_conflicts") == "author"
        and (
            pr.get("mergeable") is False
            or (pr.get("mergeable") is None and previous_conflict is not None)
        )
    )
    if author_conflict:
        # Approval dates and commit dates cannot establish when a conflict began.
        since = previous_conflict["since"] if previous_conflict else now
        action = make_action(
            item,
            row,
            config,
            policy,
            now,
            "Resolve conflicts",
            issue["user"]["login"],
            since,
            issue["html_url"],
            "Update the PR branch and resolve its merge conflicts, then request re-review if needed",
            queue="Waiting for author",
        )
        action["uncertainty"] = [
            *action["uncertainty"],
            "Clock starts at the first confirmed conflict observation; the actual conflict start is unknown",
        ]
        result.append(action)
    if pr:
        requests = evidence["requests"]
        individuals = [
            r for r in requests if r["identity"].startswith("user:")
        ]
        for request in requests:
            identity, actor, since = (
                request["identity"],
                request["actor"],
                request["since"],
            )
            team = (
                identity.split(":", 1)[1]
                if identity.startswith("team:")
                else None
            )
            if team and (
                individuals or team not in config["scope"]["team_slugs"]
            ):
                continue
            kind = "Assign reviewer" if team else "Review"
            queue = (
                "Needs reviewer assignment"
                if team
                else evidence["review_queue"]
            )
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
                    request["evidence"],
                    f"Select a named reviewer for {actor}"
                    if team
                    else "Submit the requested review or explicitly hand it off",
                    queue=queue,
                    discriminator=identity if team else "",
                )
            )
    if row["queue"] in QUEUE_ACTIONS:
        kind, description = QUEUE_ACTIONS[row["queue"]]
        if author_conflict and kind in {
            "Inspect blockers",
            "Clarify handoff",
            "Assign reviewer",
        }:
            return result, None
        if kind == "Respond" and author is None:
            kind = "Clarify handoff"
            description = f"Identify a human to address changes requested from bot author @{issue['user']['login']}"
        owner = author if kind == "Respond" else primary
        since = row["since"]
        if kind == "Inspect blockers":
            resolved = [
                action.get("resolved_at") or now
                for action in routing_history or []
                if action["source"] == issue["html_url"]
                and action["kind"] == "Resolve conflicts"
                and (
                    action.get("resolved_at")
                    or (pr and pr.get("mergeable") is True)
                )
            ]
            if resolved:
                # Resuming inspection after author work is a new handoff episode.
                since = max(
                    [*resolved, *([since] if since else [])], key=parse_time
                )
        handoff_evidence = (
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
                since,
                handoff_evidence,
                description,
            )
        )
    return result, None


def author_inactivity(action, item, policy, now, routing_history):
    previous = next(
        (old for old in routing_history or [] if old["key"] == action["key"]),
        None,
    )
    since = author_clock(item, action["since"], previous)
    if previous and previous.get("source_sha") != action["source_sha"]:
        action["uncertainty"] = [
            *action["uncertainty"],
            "Head changed without confirmed push actor; commit authorship does not establish author activity",
        ]
    action["last_author_activity_at"] = since
    threshold = policy["thresholds"]["Waiting for author"]
    days = threshold["days"]
    action["author_inactivity_days"] = days
    age, overdue = queue_age(since, now, threshold)
    action.update(age_days=age, overdue=overdue)
    action["attention"] = (
        "Unknown age"
        if age is None
        else "Overdue"
        if age > days
        else "On track"
    )
    action["project_fields"]["Attention"] = action["attention"]
    action["project_fields"]["Last author activity"] = (
        parse_time(since).date().isoformat() if since else None
    )
    if action["kind"] == "Respond":
        action["description"] = (
            "Address the requested changes and explicitly request re-review when ready"
        )


def notification_recipients(action, config):
    kind = action["notification"]["kind"]
    if kind == "none":
        return []
    if "author_inactivity_days" in action:
        return [config["escalation_contact"]]
    return [
        action["owner"],
        *([config["escalation_contact"]] if kind == "escalation" else []),
    ]


def slack_text(action, config, project_state):
    def contact(login, required=True):
        users = {
            name.casefold(): member
            for name, member in config["slack"]["users"].items()
        }
        member = users.get(login.casefold())
        if member:
            return f"<@{member}>"
        if not required:
            return f"<https://github.com/{login}|@{login}>"
        return f"[Slack mapping needed: GitHub {login}]"

    mention = contact(
        action["owner"], required="author_inactivity_days" not in action
    )
    if (
        action["notification"]["kind"] == "escalation"
        or (
            "author_inactivity_days" in action
            and action["notification"]["kind"] == "reminder"
        )
    ) and action["owner"].casefold() != config[
        "escalation_contact"
    ].casefold():
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
    if "author_inactivity_days" in action:
        if action.get("last_author_activity_at"):
            lines.append(
                f"Last author activity: {action['last_author_activity_at']}."
            )
        if kind in {"escalation", "reminder"}:
            lines.append(
                f"More than {action['author_inactivity_days']} calendar days without author activity; component lead, please follow up. The author retains ownership."
            )
    elif kind == "escalation":
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


def preview(
    snapshot,
    policy,
    config,
    project_state=None,
    history=None,
    routing_history=None,
):
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema_version")
    if snapshot["repository"] != config["repository"]:
        raise ValueError("snapshot repository does not match configuration")
    policy = {**policy, "triage_owner": "@" + config["primary_contact"]}
    now = snapshot["collected_at"]
    validate_history(routing_history, now)
    actions, omitted = [], []
    for item in snapshot["items"]:
        selected, reason = candidates(
            item, config, policy, now, routing_history
        )
        actions.extend(selected)
        if reason:
            omitted.append(
                {"number": item["issue"]["number"], "reason": reason}
            )
    missing = set()
    for action in actions:
        previous = next(
            (
                old
                for old in routing_history or []
                if old["key"] == action["key"]
            ),
            {},
        )
        action["source_observed_at"] = (
            previous.get("source_observed_at", now)
            if previous.get("source_sha") == action["source_sha"]
            else now
        )
        if action["kind"] in {"Respond", "Resolve conflicts"}:
            author_inactivity(
                action,
                next(
                    item
                    for item in snapshot["items"]
                    if item["issue"]["html_url"] == action["source"]
                ),
                policy,
                now,
                routing_history,
            )
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
        for recipient in notification_recipients(action, config):
            if recipient.casefold() not in {
                name.casefold() for name in config["slack"]["users"]
            }:
                missing.add(recipient)
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
        "Reviewer/routing escalation clocks start at successful notification. "
        "Author follow-up deadlines use inactivity even before notification. "
        "Required Slack mappings must be resolved before delivery.",
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
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "policy.json")
    parser.add_argument("--project-state", type=Path)
    parser.add_argument(
        "--routing-history",
        type=Path,
        help="Saved routing history, preserving conflict and resumed-inspection clocks",
    )
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
        json.loads(args.routing_history.read_text())
        if args.routing_history
        else None,
    )
    content = (
        json.dumps(plan, indent=2) + "\n"
        if args.format == "json"
        else render_markdown(plan)
    )
    write_private(args.output, content)


if __name__ == "__main__":
    main()
