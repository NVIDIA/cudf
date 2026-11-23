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


def classify(issue, component_owners):
    existing = sorted(
        {label["name"] for label in issue["labels"]} & component_owners.keys()
    )
    if existing:
        return existing, {"existing_labels": existing}

    # Installation diagnostics often list packages unrelated to the reported issue.
    body = re.sub(
        r"(?ims)^[ \t]*(?:\*\*|#+[ \t]+)?Environment (?:overview|details)\b.*?"
        r"(?=^[ \t]*(?:\*\*|#+[ \t]+)[A-Z]|\Z)",
        "",
        issue.get("body") or "",
    )
    text = issue["title"] + "\n" + body
    matches = {}
    for component, pattern in SIGNALS.items():
        if component in component_owners and (
            match := re.search(pattern, text)
        ):
            matches[component] = match.group()
    candidates = set(matches)
    if candidates & PYTHON_SPECIALTIES:
        candidates.discard("Python")
    return sorted(candidates), {"matched_signals": matches}


def propose(issue, component_owners, team_members):
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
        "reason": "Stable selection from the component codeowners",
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
                f"orgs/{org}/teams/{slug}/members?per_page=100", paginate=True
            )
        return rosters[team]

    proposals = []
    for number in numbers:
        issue = github.get(f"repos/{repository}/issues/{number}")
        if args.event and (
            issue["state"] != "open" or "pull_request" in issue
        ):
            return
        proposals.append(propose(issue, owners, team_members))
    if args.event:
        post(slack_token(allow_prompt=False), dm_payload(config, proposals[0]))
    else:
        save_json(args.output, proposals)


if __name__ == "__main__":
    main()
