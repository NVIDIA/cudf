#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Save a read-only GitHub snapshot for offline queue classification."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class GitHub:
    """Use gh's existing authentication and pagination without a shell."""

    def get(self, endpoint, *, paginate=False, key=None):
        command = ["gh", "api", "--method", "GET", endpoint]
        if paginate:
            command.extend(["--paginate", "--slurp"])
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=180, check=False
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        data = json.loads(result.stdout)
        if not paginate:
            return data
        return [row for page in data for row in (page[key] if key else page)]

    def graphql(self, query, variables=None):
        result = subprocess.run(
            ["gh", "api", "graphql", "--input", "-"],
            input=json.dumps({"query": query, "variables": variables or {}}),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        data = json.loads(result.stdout)
        if data.get("errors"):
            raise RuntimeError(json.dumps(data["errors"]))
        return data["data"]

    def search(self, repository, query, count):
        params = urlencode(
            {
                "q": f"repo:{repository} is:open {query}",
                "sort": "updated",
                "order": "asc",
                "per_page": count,
            }
        )
        result = self.get(f"search/issues?{params}")
        if result.get("incomplete_results"):
            raise RuntimeError("GitHub returned incomplete search results")
        return result["items"]


def sample(github, repository, size, triage_label, label=None):
    """Sample several search strata; updated_at affects selection only."""
    queries = [
        f'is:issue label:"{triage_label}"',
        "is:pr draft:false review:approved",
        "is:pr draft:false review:changes_requested",
        "is:pr draft:false review:none",
        "is:pr draft:true",
    ]
    if label:
        queries = [f'{query} label:"{label}"' for query in queries]
    count = min(100, max(1, (size + len(queries) - 1) // len(queries)))
    buckets = [github.search(repository, query, count) for query in queries]
    selected = {}
    for index in range(count):
        for bucket in buckets:
            if index < len(bucket):
                item = bucket[index]
                selected.setdefault(item["number"], item)
            if len(selected) == size:
                return list(selected), queries
    if len(selected) < size:
        fallback = f'is:pr label:"{label}"' if label else "is:pr"
        for item in github.search(repository, fallback, size):
            selected.setdefault(item["number"], item)
            if len(selected) == size:
                break
    return list(selected), queries


def collect_item(github, repository, number):
    root = f"repos/{repository}"
    issue = github.get(f"{root}/issues/{number}")
    item = {
        "issue": issue,
        "timeline": github.get(
            f"{root}/issues/{number}/timeline?per_page=100", paginate=True
        ),
        "warnings": [],
    }
    if "pull_request" in issue:
        pr = github.get(f"{root}/pulls/{number}")
        item["pull_request"] = pr
        item["reviews"] = github.get(
            f"{root}/pulls/{number}/reviews?per_page=100", paginate=True
        )
        item["review_comments"] = github.get(
            f"{root}/pulls/{number}/comments?per_page=100", paginate=True
        )
        sha = pr["head"]["sha"]
        commit = github.get(f"{root}/commits/{sha}")
        item["head_commit_author"] = (commit.get("author") or {}).get("login")
        for field, endpoint, key in [
            (
                "check_runs",
                f"{root}/commits/{sha}/check-runs?per_page=100&filter=latest",
                "check_runs",
            ),
            ("statuses", f"{root}/commits/{sha}/statuses?per_page=100", None),
        ]:
            try:
                item[field] = github.get(endpoint, paginate=True, key=key)
            except RuntimeError as error:
                if "HTTP 403" not in str(error) and "HTTP 404" not in str(
                    error
                ):
                    raise
                item[field] = None
                item["warnings"].append(f"{field} unavailable: {error}")
    item["observed_at"] = timestamp()
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="NVIDIA/cudf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label", help="Restrict sample or full listing to a label"
    )
    parser.add_argument(
        "--policy", type=Path, default=Path(__file__).with_name("policy.json")
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--numbers", type=int, nargs="+")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--sample-size", type=int, default=25)
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo must be owner/name")
    if not 1 <= args.sample_size <= 100:
        parser.error("--sample-size must be between 1 and 100")
    if args.numbers and any(number < 1 for number in args.numbers):
        parser.error("--numbers must contain positive issue/PR numbers")
    policy = json.loads(args.policy.read_text())
    github = GitHub()
    started_at = timestamp()
    if args.numbers:
        numbers = list(dict.fromkeys(args.numbers))
        selection_info = {"mode": "explicit"}
    elif args.all:
        params = {"state": "open", "per_page": 100}
        if args.label:
            params["labels"] = args.label
        inventory = github.get(
            f"repos/{args.repo}/issues?{urlencode(params)}", paginate=True
        )
        numbers = [item["number"] for item in inventory]
        selection_info = {
            "mode": "all open issues and PRs",
            "label": args.label,
        }
    else:
        numbers, queries = sample(
            github,
            args.repo,
            args.sample_size,
            policy["triage_label"],
            args.label,
        )
        selection_info = {
            "mode": "sample",
            "requested_size": args.sample_size,
            "label": args.label,
            "queries": queries,
            "order": "oldest updated first within each stratum",
        }
    items = []
    for index, number in enumerate(numbers, start=1):
        sys.stderr.write(f"Collecting {index}/{len(numbers)}: #{number}\n")
        items.append(collect_item(github, args.repo, number))
    snapshot = {
        "schema_version": 1,
        "repository": args.repo,
        "started_at": started_at,
        "collected_at": timestamp(),
        "selection": selection_info,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        sys.exit(f"Collection failed: {error}")
