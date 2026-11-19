#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refresh selected PRs, synchronize Project cards, and optionally deliver to a test DM."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import sys
import uuid
from copy import deepcopy
from pathlib import Path

from actions import notification_recipients, preview
from collect import GitHub, collect_item, timestamp
from project import read_project
from slack_dm_test import payload, post, recipient
from sync import ProjectCards

ROOT = Path(__file__).parent


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def delivery_history(cards, route):
    return {
        key: card["data"]["deliveries"].get(route, {})
        for key, card in cards.cards.items()
    }


def acknowledge(cards, key, route, pending, receipt):
    card = cards.cards[key]
    data = deepcopy(card["data"])
    record = data["deliveries"][route]
    record.pop("pending")
    record.setdefault("first_notified_at", receipt["sent_at"])
    record["last_notified_at"] = receipt["sent_at"]
    record.setdefault("thread_ts", receipt["ts"])
    record["channel"] = receipt["channel"]
    if pending["kind"] == "escalation":
        record["escalated_at"] = receipt["sent_at"]
    thread = (
        cards.config["slack"]["workspace_url"]
        + "/archives/"
        + record["channel"]
        + "/p"
        + record["thread_ts"].replace(".", "")
    )
    cards.save(key, data, {"Slack thread": thread})


def recover_receipts(cards, route, journal):
    unresolved = []
    for key, card in list(cards.cards.items()):
        pending = card["data"]["deliveries"].get(route, {}).get("pending")
        if not pending:
            continue
        receipt = journal.get(pending["id"])
        if receipt:
            acknowledge(cards, key, route, pending, receipt)
        else:
            unresolved.append(key)
    return unresolved


def send_notifications(
    cards,
    plan,
    route,
    member,
    token,
    journal,
    journal_path,
    *,
    clock=None,
    post_message=None,
    heading="DM PILOT — preview for the primary contact; no action requested",
):
    clock = clock or timestamp
    post_message = post_message or post
    sent = []
    for action in plan["actions"]:
        key = action["key"]
        card = cards.cards[key]
        if card["item"]["isArchived"] or card["values"].get("Status") in {
            "Complete",
            "Deferred",
        }:
            continue
        kind = action["notification"]["kind"]
        if kind == "none":
            continue
        data = deepcopy(card["data"])
        record = data["deliveries"].setdefault(route, {})
        if record.get("pending"):
            raise RuntimeError(
                "Unresolved delivery; inspect the DM before retrying: " + key
            )
        pending = {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "prepared_at": clock(),
        }
        record["pending"] = pending
        # Persist intent before sending; an uncertain attempt must never be retried blindly.
        cards.save(key, data, {})
        text = f"*{heading}*\n"
        text += action["slack_preview"]
        database_id = card["item"].get("fullDatabaseId")
        if database_id:
            text += f"\n<{cards.config['project']['url']}?pane=issue&itemId={database_id}|Project action card>"
        text += "\nTest delivery history is separate from production reminders and escalations."
        # Always resolve the configured contact's DM, even if stored history is damaged.
        response = post_message(
            token, payload(member, text, record.get("thread_ts"))
        )
        receipt = {**response, "sent_at": clock()}
        journal[pending["id"]] = receipt
        # A local receipt repairs a successful send whose subsequent Project write failed.
        save_json(journal_path, journal)
        print(json.dumps({"sent": key, **receipt}), flush=True)
        acknowledge(cards, key, route, pending, receipt)
        sent.append(key)
    return sent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", type=int, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "pilot.json")
    parser.add_argument("--policy", type=Path, default=ROOT / "policy.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--apply", action="store_true", help="Synchronize Project draft cards"
    )
    parser.add_argument(
        "--send-dm",
        action="store_true",
        help="Deliver only to the primary contact's bot DM",
    )
    args = parser.parse_args()
    if args.send_dm and not args.apply:
        parser.error("--send-dm requires --apply")
    if (
        any(number < 1 for number in args.numbers)
        or len(set(args.numbers)) > 10
    ):
        parser.error("select 1–10 positive issue/PR numbers for the pilot")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    policy = json.loads(args.policy.read_text())
    member = recipient(config)
    route = "dm-test:" + member
    with (args.output_dir / "pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        github = GitHub()
        state = read_project(github, config["project"])
        cards = ProjectCards(github, config, state, args.apply)
        journal_path = args.output_dir / "dm-receipts.json"
        journal = (
            json.loads(journal_path.read_text())
            if journal_path.exists()
            else {}
        )
        if args.apply:
            unresolved = recover_receipts(cards, route, journal)
            if args.send_dm and unresolved:
                raise RuntimeError(
                    "Uncertain sends without local receipts. Inspect the DM and resolve these records before retrying: "
                    + ", ".join(unresolved)
                )
        token = None
        if args.send_dm:
            token = os.environ.get("SLACK_BOT_TOKEN")
            if not token:
                if not sys.stdin.isatty():
                    parser.error(
                        "set SLACK_BOT_TOKEN or use an interactive terminal"
                    )
                token = getpass.getpass("Slack bot token (hidden): ").strip()
            if not token:
                parser.error("missing Slack bot token")
        snapshot = {
            "schema_version": 1,
            "repository": config["repository"],
            "started_at": timestamp(),
            "selection": {"mode": "explicit"},
            "items": [
                collect_item(github, config["repository"], number)
                for number in dict.fromkeys(args.numbers)
            ],
            "collected_at": timestamp(),
        }
        save_json(args.output_dir / "snapshot.json", snapshot)
        plan = preview(
            snapshot,
            policy,
            config,
            state,
            routing_history=cards.routing_history(),
        )
        cards.reconcile(plan, snapshot)
        dm_plan = preview(
            snapshot,
            policy,
            config,
            state,
            delivery_history(cards, route),
            cards.routing_history(),
        )
        save_json(args.output_dir / "dm-plan.json", dm_plan)
        save_json(
            args.output_dir / "routing-history.json", cards.routing_history()
        )
        summary = {
            "project": config["project"]["url"],
            "dm_recipient": member,
            "applied": args.apply,
            "changes": cards.changes,
            "sent": [],
        }
        if args.send_dm:
            mapped = {login.casefold() for login in config["slack"]["users"]}
            missing = {
                login
                for action in dm_plan["actions"]
                if not cards.cards[action["key"]]["item"]["isArchived"]
                and cards.cards[action["key"]]["values"].get("Status")
                not in {"Complete", "Deferred"}
                for login in notification_recipients(action, config)
                if login.casefold() not in mapped
            }
            if missing:
                raise ValueError(
                    "Resolve Slack mappings before sending: "
                    + ", ".join(sorted(missing))
                )
            summary["sent"] = send_notifications(
                cards, dm_plan, route, member, token, journal, journal_path
            )
        if args.apply:
            verified = ProjectCards(github, config, state)
            verified.reconcile(plan, snapshot)
            if verified.changes:
                raise RuntimeError(
                    "Project read-back differs from the plan: "
                    + json.dumps(verified.changes)
                )
            summary["verified"] = True
        save_json(args.output_dir / "result.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        sys.exit(str(error))
