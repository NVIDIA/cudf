#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send a smoke test to the primary contact's bot DM, independently of delivery history."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent


def slack_token(*, allow_prompt=True):
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token and allow_prompt and sys.stdin.isatty():
        token = getpass.getpass("Slack bot token (hidden): ").strip()
    if not token:
        raise RuntimeError(
            "SLACK_BOT_TOKEN is required; interactive commands can prompt in a terminal"
        )
    return token


def recipient(config):
    contact = config["primary_contact"].casefold()
    users = {
        login.casefold(): member
        for login, member in config["slack"]["users"].items()
    }
    member = users[contact]
    if not re.fullmatch(r"U[A-Z0-9]+", member):
        raise ValueError("DM recipient must be a Slack user ID")
    return member


def payload(channel, text, thread_ts=None):
    result = {
        "channel": channel,
        "text": text,
        "unfurl_links": False,
        "unfurl_media": False,
    }
    if thread_ts:
        result["thread_ts"] = thread_ts
    return result


def post(token, message):
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(message).encode(),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Slack HTTP {error.code}") from None
    except urllib.error.URLError:
        # An uncertain send must be inspected in Slack before retrying.
        raise RuntimeError(
            "Slack connection failed; check your bot DM before retrying"
        ) from None
    if not result.get("ok"):
        code = result.get("error", "unknown_error")
        if not isinstance(code, str) or not re.fullmatch(r"[a-z_]+", code):
            code = "unknown_error"
        raise RuntimeError("Slack API: " + code)
    if not re.fullmatch(r"D[A-Z0-9]+", result.get("channel", "")):
        raise RuntimeError("Slack did not return a DM; stopping the test")
    if (
        message["channel"].startswith("D")
        and result["channel"] != message["channel"]
    ):
        raise RuntimeError("Slack returned a different DM; stopping the test")
    if not re.fullmatch(r"\d+\.\d+", result.get("ts", "")):
        raise RuntimeError("Slack did not return a valid message timestamp")
    return {"channel": result["channel"], "ts": result["ts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Optional message body; preserves mentions",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send a test parent and threaded reply",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    member = recipient(config)
    text = (
        args.text_file.read_text()
        if args.text_file
        else "cuDF work tracking connectivity test. No action is required."
    )
    if not text.strip() or len(text) > 35000:
        parser.error("test message must contain 1–35000 characters")
    message = payload(member, "*DM TEST — no action required*\n" + text)
    if not args.send:
        print(json.dumps(message, indent=2))
        return
    token = slack_token()
    parent = post(token, message)
    print(json.dumps({"test_parent": parent}), flush=True)
    reply = post(
        token,
        payload(
            parent["channel"],
            "Threaded reminder test. No reminder or escalation clock was started.",
            parent["ts"],
        ),
    )
    print(json.dumps({"test_reply": reply}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as error:
        sys.exit(str(error))
