#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile one manually verified DM outcome without sending a Slack message."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from collect import GitHub
from project import read_project
from slack_dm_test import recipient
from storage import save_json
from sync import ProjectCards
from track import (
    acknowledge_with_retry,
    clear_rejected_delivery,
    delivery_card,
    retry_project_write,
)

ROOT = Path(__file__).parent


def message_receipt(url, workspace):
    parsed = urlparse(url)
    match = re.fullmatch(r"/archives/(D[A-Z0-9]+)/p(\d+)(\d{6})", parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.netloc != urlparse(workspace).netloc
        or not match
    ):
        raise ValueError(
            "Use the exact DM message permalink from the configured Slack workspace"
        )
    channel, seconds, micros = match.groups()
    sent_at = datetime.fromtimestamp(int(seconds), timezone.utc).replace(
        microsecond=int(micros)
    )
    return {
        "channel": channel,
        "ts": seconds + "." + micros,
        "sent_at": sent_at.isoformat(),
    }


def inspect_pending(cards, key):
    card = cards.cards[key]
    route = "dm-test:" + recipient(cards.config)
    pending = card["data"]["deliveries"].get(route, {}).get("pending")
    if not pending:
        raise ValueError("This card has no pending DM delivery")
    return {
        "key": key,
        "source": card["data"]["action"]["source"],
        "description": card["data"]["action"]["description"],
        "archived": card["item"]["isArchived"],
        "pending": pending,
    }


def reconcile_pending(
    cards, key, pending_id, receipt, journal_path, apply=False
):
    route = "dm-test:" + recipient(cards.config)
    card = delivery_card(cards, key)
    record = card["data"]["deliveries"].get(route, {})
    pending = record.get("pending")
    if not pending or pending["id"] != pending_id:
        raise ValueError(
            "Pending delivery does not match; inspect fresh Project state before proceeding"
        )
    if (
        receipt
        and record.get("channel", receipt["channel"]) != receipt["channel"]
    ):
        raise ValueError(
            "Message permalink points to a different DM than the recorded delivery history"
        )
    if (
        receipt
        and pending["kind"] != "initial"
        and record.get("thread_ts") == receipt["ts"]
    ):
        raise ValueError(
            "Use the delivered reminder or escalation link, not its parent thread"
        )
    result = {
        "key": key,
        "pending_id": pending_id,
        "outcome": "sent" if receipt else "not-sent",
        "applied": apply,
    }
    journal = (
        json.loads(journal_path.read_text()) if journal_path.exists() else {}
    )
    if not isinstance(journal, dict):
        raise ValueError("Receipt journal must be an object")
    saved = journal.get(pending_id)
    if saved is not None and (
        not receipt
        or not isinstance(saved, dict)
        or any(saved.get(name) != receipt[name] for name in ("channel", "ts"))
    ):
        raise ValueError(
            "Existing receipt contradicts this outcome; recover using the recorded send"
        )
    if not apply:
        return result
    if not cards.apply:
        raise ValueError("Recovery requires Project write access")
    if receipt:
        # Keep a durable receipt before any acknowledgement can fail partway through.
        journal[pending_id] = receipt
        save_json(journal_path, journal)
        acknowledge_with_retry(cards, key, route, pending, receipt)
    else:
        retry_project_write(
            cards, lambda: clear_rejected_delivery(cards, key, route, pending)
        )
    cards.refresh()
    verified = delivery_card(cards, key)
    if verified["data"]["deliveries"].get(route, {}).get("pending"):
        raise RuntimeError(
            "Recovery read-back still has pending intent; keep delivery disabled"
        )
    if receipt:
        record = verified["data"]["deliveries"][route]
        expected_thread = (
            cards.config["slack"]["workspace_url"]
            + "/archives/"
            + record["channel"]
            + "/p"
            + record["thread_ts"].replace(".", "")
        )
        if (
            verified["values"].get("Slack thread") != expected_thread
            or record.get("last_notified_at") != receipt["sent_at"]
        ):
            raise RuntimeError(
                "Recovery read-back differs from the confirmed send; keep delivery disabled"
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--key", required=True)
    parser.add_argument("--pending-id")
    outcome = parser.add_mutually_exclusive_group(required=True)
    outcome.add_argument(
        "--inspect",
        action="store_true",
        help="Read the pending identity and handoff evidence without changing the Project",
    )
    outcome.add_argument(
        "--sent-message-url",
        help="Exact delivered message permalink, not just its parent thread",
    )
    outcome.add_argument(
        "--not-sent",
        action="store_true",
        help="Operator has confirmed that no message was delivered",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.inspect and (args.apply or args.pending_id):
        parser.error(
            "--inspect cannot be combined with --apply or --pending-id"
        )
    if not args.inspect and not args.pending_id:
        parser.error(
            "reconciliation requires --pending-id from a fresh --inspect"
        )
    config = json.loads(args.config.read_text())
    receipt = (
        message_receipt(
            args.sent_message_url, config["slack"]["workspace_url"]
        )
        if args.sent_message_url
        else None
    )
    github = GitHub()
    cards = ProjectCards(
        github, config, read_project(github, config["project"]), args.apply
    )
    result = (
        inspect_pending(cards, args.key)
        if args.inspect
        else reconcile_pending(
            cards,
            args.key,
            args.pending_id,
            receipt,
            args.output_dir / "dm-receipts.json",
            args.apply,
        )
    )
    save_json(args.output_dir / "recovery.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        OverflowError,
    ) as error:
        sys.exit(str(error))
