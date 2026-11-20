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
# Other API errors may represent partial success, including internal_error/fatal_error.
DEFINITE_REJECTIONS = {
    "access_denied",
    "accesslimited",
    "agent_prompt_display_not_allowed",
    "app_access_restricted",
    "as_user_not_supported",
    "attachment_payload_limit_exceeded",
    "cannot_reply_to_message",
    "deprecated_endpoint",
    "draft_already_deleted",
    "draft_has_conflict",
    "draft_not_found",
    "duplicate_channel_not_found",
    "duplicate_message_not_found",
    "enterprise_is_restricted",
    "invalid_arg_name",
    "invalid_array_arg",
    "invalid_blocks",
    "invalid_blocks_format",
    "invalid_charset",
    "invalid_form_data",
    "invalid_metadata_format",
    "invalid_metadata_schema",
    "invalid_post_type",
    "markdown_text_conflict",
    "message_limit_exceeded",
    "messages_tab_disabled",
    "metadata_must_be_sent_from_app",
    "metadata_too_large",
    "method_deprecated",
    "missing_argument",
    "missing_file_data",
    "missing_post_type",
    "msg_blocks_too_long",
    "not_allowed_token_type",
    "org_login_required",
    "posting_to_channel_denied",
    "ratelimited",
    "request_timeout",
    "restricted_action",
    "restricted_action_non_threadable_channel",
    "restricted_action_read_only_channel",
    "restricted_action_thread_locked",
    "restricted_action_thread_only_channel",
    "send_on_behalf_not_allowed",
    "slack_connect_canvas_sharing_blocked",
    "slack_connect_file_link_sharing_blocked",
    "slack_connect_lists_sharing_blocked",
    "team_access_not_granted",
    "team_not_found",
    "token_expired",
    "token_revoked",
    "too_many_attachments",
    "too_many_contact_cards",
    "two_factor_setup_required",
    "invalid_auth",
    "not_authed",
    "account_inactive",
    "missing_scope",
    "no_permission",
    "channel_not_found",
    "not_in_channel",
    "is_archived",
    "rate_limited",
    "invalid_arguments",
    "no_text",
    "ekm_access_denied",
}


class SlackRejected(RuntimeError):
    """Slack confirmed that a message was not accepted for delivery."""


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
    member = users.get(contact)
    if not member:
        raise ValueError("Primary contact requires a Slack user mapping")
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
        if error.code == 429:
            raise SlackRejected("Slack HTTP 429") from None
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
        error_type = (
            SlackRejected if code in DEFINITE_REJECTIONS else RuntimeError
        )
        raise error_type("Slack API: " + code)
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
