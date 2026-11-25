#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Propose issue routing locally or send a private preview for a new issue."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from subprocess import TimeoutExpired

from collect import GitHub
from selection import validate_numbers
from slack_dm_test import payload, post, recipient, slack_token
from storage import save_json

ROOT = Path(__file__).parent
SIGNALS = {
    "cudf.pandas": r"\bcudf\.pandas\b",
    "cudf-polars": r"\bcudf[_-]polars\b",
    "pylibcudf": r"\bpylibcudf\b",
    "Java": r"\bai\.rapids\.cudf\b|\bjava/src/|\bcudf-java\b",
    "Python": r"\b(?:import|from)\s+cudf\b|\bpython/cudf/|\bcudf\.(?:Series|DataFrame|Index|read_\w+)\b",
    "libcudf": r"\bcudf::\w+|\bcpp/(?:src|include|tests)/",
    "CMake": r"CMake Error|CMakeLists\.txt|\bfind_package\s*\(",
}
PYTHON_SPECIALTIES = {"cudf.pandas", "cudf-polars", "pylibcudf"}


SECTION = re.compile(r"(?m)^(?:#{1,6}[ \t]+(.+)|\*\*(.+?)\*\*[ \t]*)$")
INTENT = re.compile(
    r"describe the bug|describe the problem|related to a problem|describe the solution|expected behavior",
    re.IGNORECASE,
)
DIAGNOSTICS = re.compile(r"^environment (?:overview|details)\b", re.IGNORECASE)
FENCED_BLOCK = re.compile(
    r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$", re.M | re.S
)
EXPLICIT_COMPONENTS = {
    "Java": r"\bJava\b",
    "libcudf": r"\blibcudf\b",
}


def match_components(text, component_owners, *, explicit=False):
    matches = {}
    for component, pattern in SIGNALS.items():
        if component not in component_owners:
            continue
        if explicit and component in EXPLICIT_COMPONENTS:
            pattern += "|" + EXPLICIT_COMPONENTS[component]
        if match := re.search(pattern, text):
            matches[component] = match.group()
    if matches.keys() & PYTHON_SPECIALTIES:
        matches.pop("Python", None)
    return matches


def classify(issue, component_owners):
    existing = sorted(
        {label["name"] for label in issue["labels"]} & component_owners.keys()
    )
    if existing:
        return existing, {"existing_labels": existing}

    # Split headings outside code fences, preserving blocks for fallback evidence.
    body = issue.get("body") or ""
    masked = FENCED_BLOCK.sub(
        lambda match: re.sub(r"[^\n]", " ", match[0]), body
    )
    headings = list(SECTION.finditer(masked))
    opening = body[: headings[0].start()] if headings else body
    intent = [opening]
    supporting = [opening]
    for i, heading in enumerate(headings):
        name = heading[1] or heading[2]
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section = body[heading.end() : end]
        if not DIAGNOSTICS.search(name):
            supporting.append(section)
        if INTENT.search(name):
            intent.append(section)

    # Higher tiers describe what the reporter wants fixed. Examples, logs, and
    # patches can still help when the report itself doesn't name a component.
    tiers = (
        ("title", issue["title"], True),
        (
            "problem or requested behavior",
            FENCED_BLOCK.sub("", "\n".join(intent)),
            True,
        ),
        ("supporting details", "\n".join(supporting), False),
    )
    for source, text, explicit in tiers:
        if matches := match_components(
            text, component_owners, explicit=explicit
        ):
            return sorted(matches), {
                "matched_signals": matches,
                "signal_source": source,
            }
    return [], {"matched_signals": {}}


def linked_pr_numbers(issue, repository):
    pattern = rf"https://github\.com/{re.escape(repository)}/pull/([1-9][0-9]*)(?![\w-])"
    text = issue["title"] + "\n" + (issue.get("body") or "")
    return list(
        dict.fromkeys(map(int, re.findall(pattern, text, re.IGNORECASE)))
    )[:3]


def propose(issue, component_owners, team_members, linked_activity=None):
    result = {
        "issue": issue["number"],
        "component": None,
        "suggested_label": None,
        "first_contact": None,
    }
    if issue.get("pull_request") is not None or issue["state"] != "open":
        return {**result, "reason": "Only open issues are eligible"}
    components, evidence = classify(issue, component_owners)
    result.update(evidence)
    if issue.get("assignees") and "existing_labels" in evidence:
        result["suppress_dm"] = True
        if len(components) == 1:
            result["component"] = components[0]
        return {
            **result,
            "reason": "Issue already has a component label and an assignee",
        }
    if len(components) != 1:
        reason = (
            "Conflicting component evidence"
            if components
            else "No strong component evidence"
        )
        return {**result, "reason": reason}

    component = components[0]
    team = component_owners[component]
    result.update(component=component, team=team)
    if "existing_labels" not in result:
        result["suggested_label"] = component
    if issue.get("assignees"):
        return {**result, "reason": "Issue already has an assignee"}

    author = issue["user"]["login"].casefold()
    try:
        members = team_members(team)
    except (RuntimeError, TimeoutExpired):
        return {**result, "reason": "Unable to read the codeowner team roster"}

    eligible = sorted(
        {
            member["login"].casefold()
            for member in members
            if member.get("type") == "User"
            and member["login"].casefold() != author
        }
    )
    if not eligible:
        return {**result, "reason": "No eligible team member"}
    activity = list(linked_activity(issue)) if linked_activity else []
    # Prefer people who implemented the linked work, then people who reviewed it.
    for role in ("author", "reviewer"):
        for number, participants in activity:
            if candidates := set(eligible).intersection(participants[role]):
                result["linked_pr"] = number
                result["linked_role"] = role
                eligible = sorted(candidates)
                break
        else:
            continue
        break
    # Spread intake across the team while keeping the same choice on a rerun.
    seed = issue["html_url"].encode()
    result["first_contact"] = max(
        eligible,
        key=lambda login: hashlib.sha256(
            seed + b":" + login.encode()
        ).digest(),
    )
    return {
        **result,
        "reason": (
            f"Eligible {result['linked_role']} on linked PR #{result['linked_pr']}"
            if "linked_pr" in result
            else "Stable selection from the component codeowners"
        ),
    }


def opened_issue_number(event, repository):
    if (
        event.get("action") != "opened"
        or event.get("repository", {}).get("full_name") != repository
        or "issue" not in event
    ):
        raise ValueError(
            "Expected an issue opened event for the configured repository"
        )
    number = event["issue"]["number"]
    validate_numbers([number])
    return number


def dm_payload(config, proposal):
    component = proposal["component"] or "Manual triage"
    label = proposal["suggested_label"] or "None"
    contact = proposal["first_contact"] or "None"
    url = (
        f"https://github.com/{config['repository']}/issues/{proposal['issue']}"
    )
    lines = [
        "Issue triage preview · no action requested",
        url,
        f"Component: {component}",
        f"Proposed label: {label}",
        f"Suggested first contact: {contact}",
        proposal["reason"],
    ]
    if labels := proposal.get("existing_labels"):
        lines.append("Existing component labels: " + ", ".join(labels))
    if source := proposal.get("signal_source"):
        lines.append(f"Evidence source: {source}")
    if number := proposal.get("linked_pr"):
        lines.append(
            f"https://github.com/{config['repository']}/pull/{number}"
        )
    for name, signal in proposal.get("matched_signals", {}).items():
        lines.append(f"Evidence for {name}: {' '.join(signal.split())}")
    # Issue text and contact names must never turn into Slack mentions.
    return {
        **payload(recipient(config), "\n".join(lines)),
        "mrkdwn": False,
        "link_names": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--numbers", nargs="+", type=int)
    mode.add_argument(
        "--event",
        type=Path,
        help="Send a DM preview for an issues.opened event",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    repository = config["repository"]
    if args.event:
        event = json.loads(args.event.read_text())
        try:
            numbers = [opened_issue_number(event, repository)]
        except ValueError as error:
            parser.error(str(error))
        if args.output:
            parser.error("--output is for local previews")
    else:
        if not args.output:
            parser.error("--numbers requires --output")
        numbers = args.numbers
        try:
            validate_numbers(numbers)
        except ValueError as error:
            parser.error(str(error))

    owners = json.loads((ROOT / "policy.json").read_text())["component_owners"]
    github = GitHub()
    rosters = {}

    def team_members(team):
        if team not in rosters:
            org, slug = team.split("/", 1)
            rosters[team] = github.get(
                f"orgs/{org}/teams/{slug}/members?per_page=100",
                paginate=True,
                timeout=30,
            )
        return rosters[team]

    def linked_activity(issue):
        for number in linked_pr_numbers(issue, repository):
            endpoint = f"repos/{repository}/pulls/{number}"
            participants = {"author": set(), "reviewer": set()}
            try:
                pr = github.get(endpoint, timeout=10)
                if user := pr.get("user"):
                    participants["author"].add(user["login"].casefold())
                reviews = github.get(
                    f"{endpoint}/reviews?per_page=100",
                    paginate=True,
                    timeout=10,
                )
            except (RuntimeError, TimeoutExpired):
                # Rotation remains available if GitHub cannot supply PR evidence.
                reviews = []
            for review in reviews:
                if review["state"] in {
                    "APPROVED",
                    "CHANGES_REQUESTED",
                    "COMMENTED",
                } and (user := review.get("user")):
                    participants["reviewer"].add(user["login"].casefold())
            yield number, participants

    proposals = []
    for number in numbers:
        issue = github.get(f"repos/{repository}/issues/{number}", timeout=30)
        if args.event and (
            issue["state"] != "open" or "pull_request" in issue
        ):
            return
        proposals.append(propose(issue, owners, team_members, linked_activity))
    if args.event:
        if not proposals[0].get("suppress_dm"):
            post(
                slack_token(allow_prompt=False),
                dm_payload(config, proposals[0]),
            )
    else:
        save_json(args.output, proposals)


if __name__ == "__main__":
    main()
