#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Propose a component and first responder without modifying GitHub or Slack."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from collect import GitHub
from selection import validate_numbers
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
    eligible = sorted(
        {
            member["login"].casefold()
            for member in team_members(team)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_numbers(args.numbers)
    except ValueError as error:
        parser.error(str(error))
    config = json.loads((ROOT / "config.json").read_text())
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

    proposals = [
        propose(
            github.get(f"repos/{config['repository']}/issues/{number}"),
            owners,
            team_members,
        )
        for number in args.numbers
    ]
    save_json(args.output, proposals)


if __name__ == "__main__":
    main()
