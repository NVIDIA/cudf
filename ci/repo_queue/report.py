#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify a saved repository snapshot and render a private queue report."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path


def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def elapsed_days(start, end, business_days=False):
    start, end = parse_time(start), parse_time(end)
    if end < start:
        raise ValueError("report time precedes a queue's waiting-since time")
    if not business_days:
        return (end - start).total_seconds() / 86400
    seconds = 0.0
    while start < end:
        midnight = datetime.combine(
            start.date() + timedelta(days=1), time(), timezone.utc
        )
        boundary = min(midnight, end)
        if start.weekday() < 5:
            seconds += (boundary - start).total_seconds()
        start = boundary
    return seconds / 86400


def is_human(user, policy):
    login = (user or {}).get("login", "")
    bots = {name.casefold() for name in policy["bot_logins"]}
    return bool(login) and (
        user.get("type") != "Bot"
        and not login.casefold().endswith("[bot]")
        and login.casefold() not in bots
    )


def label_since(events, labels):
    matches = [
        event["created_at"]
        for event in events
        if event["event"] == "labeled"
        and event.get("label", {}).get("name") in labels
    ]
    return max(matches, key=parse_time) if matches else None


def ready_since(issue, events):
    transitions = [
        event
        for event in events
        if event["event"] in {"ready_for_review", "convert_to_draft"}
    ]
    if transitions and transitions[-1]["event"] == "convert_to_draft":
        return None
    ready = (
        transitions[-1]["created_at"] if transitions else issue["created_at"]
    )
    reopened = [
        event["created_at"] for event in events if event["event"] == "reopened"
    ]
    return max([ready, *reopened], key=parse_time)


def event_link(issue, events, since):
    matches = [event for event in events if event.get("created_at") == since]
    if matches and matches[-1].get("id"):
        return f"{issue['html_url']}#event-{matches[-1]['id']}"
    return issue["html_url"]


def requested_key(event):
    reviewer = event.get("requested_reviewer") or {}
    team = event.get("requested_team") or {}
    if reviewer.get("login"):
        return "user:" + reviewer["login"].casefold()
    if team.get("slug"):
        return "team:" + team["slug"].casefold()
    return None


def review_requests(pr, events, policy):
    clocks = {}
    for event in events:
        key = requested_key(event)
        if event["event"] == "review_requested":
            clocks[key] = event["created_at"]
        elif event["event"] == "review_request_removed":
            clocks.pop(key, None)
    requests = []
    for user in pr.get("requested_reviewers", []):
        if is_human(user, policy):
            key = "user:" + user["login"].casefold()
            requests.append((key, "@" + user["login"], clocks.get(key)))
    owner = pr["base"]["repo"]["owner"]["login"]
    for team in pr.get("requested_teams", []):
        key = "team:" + team["slug"].casefold()
        requests.append((key, f"@{owner}/{team['slug']}", clocks.get(key)))
    return requests


def human_reviews(item, policy):
    author = item["issue"]["user"]["login"].casefold()
    return sorted(
        (
            review
            for review in item["reviews"]
            if review.get("submitted_at")
            and review["state"] != "PENDING"
            and is_human(review.get("user"), policy)
            and review["user"]["login"].casefold() != author
        ),
        key=lambda review: parse_time(review["submitted_at"]),
    )


def observed_blockers(item):
    pr = item.get("pull_request")
    if not pr:
        return []
    blockers = []
    if pr.get("mergeable") is False:
        blockers.append("Merge conflict reported by GitHub")
    elif pr.get("mergeable") is None:
        blockers.append("Mergeability unknown")
    for field, name_key, state_key, when in [
        ("check_runs", "name", "conclusion", "started_at"),
        ("statuses", "context", "state", "created_at"),
    ]:
        data = item.get(field)
        if data is None:
            blockers.append(f"{field} unavailable")
            continue
        latest = {}
        for entry in sorted(data, key=lambda row: row.get(when) or ""):
            key = (entry[name_key], (entry.get("app") or {}).get("id"))
            latest[key] = entry
        failed, pending = [], []
        for entry in latest.values():
            state = entry.get(state_key)
            if state in {
                "failure",
                "error",
                "cancelled",
                "timed_out",
                "action_required",
                "startup_failure",
                "stale",
            }:
                failed.append(entry[name_key])
            elif state == "pending" or (
                field == "check_runs" and entry.get("status") != "completed"
            ):
                pending.append(entry[name_key])
        if failed:
            blockers.append("Observed CI failure: " + ", ".join(failed))
        if pending:
            blockers.append("Observed CI pending: " + ", ".join(pending))
    return blockers


def classify(item, policy, now):
    issue = item["issue"]
    labels = {label["name"] for label in issue["labels"]}
    events = sorted(
        (event for event in item["timeline"] if event.get("created_at")),
        key=lambda event: parse_time(event["created_at"]),
    )
    components = sorted(labels & policy["component_owners"].keys())
    owner_teams = sorted(
        {"@" + policy["component_owners"][label] for label in components}
    )
    routing = ", ".join(owner_teams) or "Component owner unassigned"
    row = {
        "number": issue["number"],
        "url": issue["html_url"],
        "title": issue["title"],
        "components": components,
        "queue": "Unclear handoff",
        "next_actor": "Maintainer to establish next action",
        "since": None,
        "evidence": [],
        "blockers": observed_blockers(item),
        "uncertainty": list(item.get("warnings", [])),
    }

    def assign(queue, actor, since, reason, url=None):
        row.update(queue=queue, next_actor=actor, since=since)
        row["evidence"].append(
            {"reason": reason, "url": url or issue["html_url"]}
        )

    pr = item.get("pull_request")
    if issue["state"] != "open" or (pr and pr["state"] != "open"):
        assign("Closed since selection", "None", None, "Item is now closed")
    elif not pr:
        if policy["triage_label"] in labels:
            assign(
                "Needs triage",
                policy.get("triage_owner") or "Triage owner unassigned",
                label_since(events, {policy["triage_label"]}),
                f"Current label: {policy['triage_label']}",
                event_link(
                    issue,
                    events,
                    label_since(events, {policy["triage_label"]}),
                ),
            )
        else:
            assign(
                "Outside intake queue",
                "Not established",
                None,
                "Triage label absent; disposition has not been verified",
            )
    elif pr["draft"]:
        assign(
            "Draft",
            "@" + issue["user"]["login"],
            None,
            "Currently a draft; this does not imply abandonment",
        )
    elif labels & set(policy["parked_labels"]):
        matched = labels & set(policy["parked_labels"])
        assign(
            "Explicitly deferred",
            "@" + issue["user"]["login"],
            label_since(events, matched),
            "Explicit label: " + ", ".join(sorted(matched)),
        )
    elif labels & set(policy["waiting_for_author_labels"]):
        matched = labels & set(policy["waiting_for_author_labels"])
        assign(
            "Waiting for author",
            "@" + issue["user"]["login"],
            label_since(events, matched),
            "Explicit label: " + ", ".join(sorted(matched)),
        )
    else:
        reviews = human_reviews(item, policy)
        effective = {}
        for review in reviews:
            # A comment-only review does not revoke an earlier decision.
            if review["state"] != "COMMENTED":
                effective[review["user"]["login"].casefold()] = review
        changes = [
            review
            for review in effective.values()
            if review["state"] == "CHANGES_REQUESTED"
        ]
        approvals = [
            review
            for review in effective.values()
            if review["state"] == "APPROVED"
        ]
        requests = review_requests(pr, events, policy)
        clocks = {key: since for key, _, since in requests}
        unaddressed = [
            review
            for review in changes
            if not (
                clocks.get("user:" + review["user"]["login"].casefold())
                and parse_time(
                    clocks["user:" + review["user"]["login"].casefold()]
                )
                > parse_time(review["submitted_at"])
            )
        ]
        if unaddressed:
            since = min(
                (review["submitted_at"] for review in unaddressed),
                key=parse_time,
            )
            author = issue["user"]["login"].casefold()
            responses = [
                event
                for event in events
                if event["event"] == "commented"
                and (event.get("user") or {}).get("login", "").casefold()
                == author
                and parse_time(event["created_at"]) > parse_time(since)
            ]
            responses.extend(
                comment
                for comment in item.get("review_comments", [])
                if (comment.get("user") or {}).get("login", "").casefold()
                == author
                and parse_time(comment["created_at"]) > parse_time(since)
            )
            responses.extend(
                review
                for review in item["reviews"]
                if (review.get("user") or {}).get("login", "").casefold()
                == author
                and review.get("submitted_at")
                and parse_time(review["submitted_at"]) > parse_time(since)
            )
            # Commit dates cannot establish when a branch was pushed.
            commits = [
                event
                for event in item["timeline"]
                if event.get("event") == "committed"
                and event.get("committer", {}).get("date")
                and parse_time(event["committer"]["date"]) > parse_time(since)
            ]
            if "review_comments" not in item:
                row["uncertainty"].append(
                    "Inline author replies were not collected; "
                    "author handoff cannot be established"
                )
            if responses or commits or "review_comments" not in item:
                row["evidence"].append(
                    {
                        "reason": "Changes-requested review needs an "
                        "explicit handoff after author activity or "
                        "incomplete reply history",
                        "url": unaddressed[0]["html_url"],
                    }
                )
            else:
                assign(
                    "Waiting for author",
                    "@" + issue["user"]["login"],
                    since,
                    "Effective human changes-requested review",
                    unaddressed[0]["html_url"],
                )
        elif requests:
            dates = [since for _, _, since in requests]
            since = min(dates, key=parse_time) if all(dates) else None
            ready = ready_since(issue, events)
            if since and ready:
                since = max(since, ready, key=parse_time)
            else:
                since = None
            assign(
                "Waiting for another review"
                if reviews
                else "Waiting for first review",
                ", ".join(actor for _, actor, _ in requests),
                since,
                "Outstanding GitHub review requests; clock uses the "
                "oldest request, bounded by readiness/reopening",
                event_link(issue, events, since),
            )
            row["review_requests"] = [
                {"actor": actor, "since": date} for _, actor, date in requests
            ]
        elif approvals:
            assign(
                "Has approvals; inspect blockers",
                routing,
                min(
                    (review["submitted_at"] for review in approvals),
                    key=parse_time,
                ),
                f"{len(approvals)} effective human approval(s); "
                "required approval policy has not been evaluated",
                approvals[0]["html_url"],
            )
            if any(
                review.get("commit_id") != pr["head"]["sha"]
                for review in approvals
            ):
                row["uncertainty"].append(
                    "Some approvals reference a different head commit"
                )
        elif not reviews:
            assign(
                "Needs reviewer assignment",
                routing,
                ready_since(issue, events),
                "Ready PR with no human review or outstanding human/team "
                "review request",
            )
        else:
            row["evidence"].append(
                {
                    "reason": "Human reviews exist without an effective "
                    "decision or outstanding review request",
                    "url": reviews[-1]["html_url"],
                }
            )
    threshold = policy["thresholds"].get(row["queue"])
    row["age_days"] = None
    row["overdue"] = None
    row["clock"] = None
    if threshold:
        row["clock"] = (
            "business days" if threshold["business_days"] else "calendar days"
        )
        if row["since"]:
            row["age_days"] = elapsed_days(
                row["since"], now, threshold["business_days"]
            )
            row["overdue"] = row["age_days"] > threshold["days"]
        else:
            row["uncertainty"].append(
                "Waiting-since event unavailable; aging is unknown"
            )
    return row


def make_report(snapshot, policy, now=None):
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema_version")
    now = now or snapshot["collected_at"]
    if parse_time(now) < parse_time(snapshot["collected_at"]):
        raise ValueError("--now must not precede snapshot collection")
    return {
        "repository": snapshot["repository"],
        "collected_at": snapshot["collected_at"],
        "as_of": now,
        "selection": snapshot["selection"],
        "policy": policy,
        "items": [classify(item, policy, now) for item in snapshot["items"]],
    }


def escape(value):
    value = html.escape(str(value), quote=False).replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()|])", r"\\\1", value)


def render_markdown(report):
    rows = report["items"]
    overdue = sum(row["overdue"] is True for row in rows)
    lines = [
        f"# Repository queues: {report['repository']}",
        "",
        f"Snapshot: {report['collected_at']}; as of: {report['as_of']}.",
        "",
        f"Scope: {escape(report['selection']['mode'])}; {len(rows)} items; "
        f"{overdue} beyond configured thresholds.",
        "",
        "Business time counts Monday–Friday in UTC, excluding weekends "
        "but not holidays. Activity and bot updates do not reset handoffs.",
        "",
        "CI failures are observations, not proof of required merge blockers. "
        "Branch rules, unresolved threads, and ops-bot merge barriers are "
        "not evaluated. Collection is not an atomic snapshot.",
        "",
        "Component team routing is suggested from labels; it does not "
        "establish ownership of the next action.",
        "",
    ]
    if report["selection"]["mode"] == "sample":
        lines.extend(
            [
                "Sampling uses oldest updated items within search strata; "
                "counts do not estimate repository-wide prevalence.",
                "",
            ]
        )
    lines.extend(
        [
            "| Queue | Items | Overdue | Unknown age |",
            "| --- | --- | --- | --- |",
        ]
    )
    for queue in dict.fromkeys(row["queue"] for row in rows):
        group = [row for row in rows if row["queue"] == queue]
        late = sum(row["overdue"] is True for row in group)
        unknown = sum(
            row["clock"] is not None and row["age_days"] is None
            for row in group
        )
        lines.append(f"| {queue} | {len(group)} | {late} | {unknown} |")
    lines.append("")
    for queue in dict.fromkeys(row["queue"] for row in rows):
        group = [row for row in rows if row["queue"] == queue]
        group.sort(key=lambda row: -(row["age_days"] or 0))
        lines.extend(
            [
                f"## {queue} ({len(group)})",
                "",
                "| Item / component | Next actor | Waiting since / age | "
                "Evidence | Observations / uncertainty |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in group:
            label = f"#{row['number']} {escape(row['title'])}"
            item = f"[{label}]({row['url']})"
            item += " — " + escape(
                ", ".join(row["components"]) or "Component unknown"
            )
            age = "Not aged"
            if row["clock"]:
                age = (
                    "Unknown"
                    if row["age_days"] is None
                    else (
                        f"{row['since']} / {row['age_days']:.1f} {row['clock']}"
                        + ("; **overdue**" if row["overdue"] else "")
                    )
                )
            evidence = "; ".join(
                f"[{escape(event['reason'])}]({event['url']})"
                for event in row["evidence"]
            )
            notes = "; ".join(row["blockers"] + row["uncertainty"]) or "—"
            lines.append(
                f"| {item} | {escape(row['next_actor'])} | {age} | "
                f"{evidence} | {escape(notes)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--policy", type=Path, default=Path(__file__).with_name("policy.json")
    )
    parser.add_argument(
        "--now", help="UTC-offset timestamp; defaults to snapshot"
    )
    parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = make_report(
        json.loads(args.snapshot.read_text()),
        json.loads(args.policy.read_text()),
        args.now,
    )
    content = (
        json.dumps(report, indent=2) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
