#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rehearse initial, reminder, and escalation DMs using an isolated local clock."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

from actions import add_business_days, notification_state, slack_text
from collect import timestamp
from pilot import (
    delivery_history,
    recover_receipts,
    save_json,
    send_notifications,
)
from report import parse_time
from slack_dm_test import post, recipient, slack_token

ROOT = Path(__file__).parent
KEY = "cudf-action:v1:dm-reminder-rehearsal"


class LocalCards:
    """Persist delivery intent and receipts without a GitHub API client."""

    def __init__(self, config, path):
        self.config = config
        self.path = path
        self.cards = (
            json.loads(path.read_text())
            if path.exists()
            else {
                KEY: {
                    "item": {"isArchived": False},
                    "values": {"Status": "Awaiting action"},
                    "data": {"deliveries": {}},
                }
            }
        )

    def save(self, key, data, fields):
        updated = deepcopy(self.cards)
        updated[key]["data"] = data
        updated[key]["values"].update(fields)
        save_json(self.path, updated)
        self.cards = updated


def rehearse(config, directory, sender, start):
    directory.mkdir(parents=True, exist_ok=True)
    clock_path = directory / "clock.json"
    if not clock_path.exists():
        save_json(clock_path, {"start": start})
    start = json.loads(clock_path.read_text())["start"]
    cards = LocalCards(config, directory / "cards.json")
    journal_path = directory / "dm-receipts.json"
    journal = (
        json.loads(journal_path.read_text()) if journal_path.exists() else {}
    )
    member = recipient(config)
    route = "dm-rehearsal:" + member
    if recover_receipts(cards, route, journal):
        raise RuntimeError(
            "Uncertain rehearsal send; inspect the DM before retrying"
        )
    policy = config["follow_up"]
    reminder = add_business_days(start, policy["repeat_after_business_days"])
    escalation = (
        parse_time(start)
        + timedelta(days=policy["escalate_after_calendar_days"])
    ).isoformat()
    if parse_time(reminder) >= parse_time(escalation):
        raise ValueError(
            "rehearsal requires a reminder before the escalation deadline"
        )
    owner = next(
        (
            login
            for login in config["slack"]["users"]
            if login != config["primary_contact"]
        ),
        config["primary_contact"],
    )
    action = {
        "key": KEY,
        "number": 25414,
        "source": "https://github.com/NVIDIA/cudf/pull/25414",
        "handoff_evidence": "https://github.com/NVIDIA/cudf/pull/25414",
        "kind": "Review",
        "owner": owner,
        "description": "Synthetic review request for this delivery rehearsal; no action is required",
        "since": (parse_time(start) - timedelta(days=14)).isoformat(),
        "overdue": True,
    }
    sent = []
    for at in (start, reminder, escalation):
        action["notification"] = notification_state(
            action, config, delivery_history(cards, route), at
        )
        kind = action["notification"]["kind"]
        action["slack_preview"] = (
            f"Simulated time: {at}. This is an accelerated test.\n"
            + slack_text(action, config, None)
        )
        delivered = send_notifications(
            cards,
            {"actions": [action]},
            route,
            member,
            "",
            journal,
            journal_path,
            clock=lambda: at,
            post_message=sender,
            heading=f"DM REHEARSAL — {kind}; no action required",
        )
        if delivered:
            sent.append(kind)
    result = {"dm_recipient": member, "sent": sent, "project_writes": 0}
    save_json(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--send-dm", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "pilot.json").read_text())
    token = None
    if args.send_dm:
        token = slack_token()
    # Preview receipts must never suppress a later real DM rehearsal.
    directory = args.output_dir / (
        "rehearsal-dm" if args.send_dm else "rehearsal-preview"
    )
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "rehearsal.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        messages = []

        def sender(unused_token, message):
            if args.send_dm:
                return post(token, message)
            messages.append(message)
            print(json.dumps({"preview": message}, indent=2))
            return {"channel": "DREHEARSAL", "ts": f"{len(messages)}.0"}

        print(
            json.dumps(
                rehearse(config, directory, sender, timestamp()), indent=2
            )
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(str(error))
