# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check DM routing and error handling without Slack credentials."""

import json
import os
import unittest
from contextlib import ExitStack
from io import BytesIO
from unittest.mock import patch

from slack_dm_test import SlackRejected, payload, post, recipient, slack_token


class SlackDMTests(unittest.TestCase):
    def test_token_policy_for_environment_prompt_and_unattended_use(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ, {"SLACK_BOT_TOKEN": " env-token "}, clear=True
                )
            )
            prompt = stack.enter_context(
                patch(
                    "slack_dm_test.getpass.getpass", return_value=" prompted "
                )
            )
            stack.enter_context(
                patch("slack_dm_test.sys.stdin.isatty", return_value=True)
            )
            self.assertEqual(slack_token(allow_prompt=False), "env-token")
            prompt.assert_not_called()
            os.environ["SLACK_BOT_TOKEN"] = "  "
            with self.assertRaises(RuntimeError):
                slack_token(allow_prompt=False)
            prompt.assert_not_called()
            self.assertEqual(slack_token(), "prompted")
            prompt.return_value = "  "
            with self.assertRaises(RuntimeError):
                slack_token()
            with patch("slack_dm_test.sys.stdin.isatty", return_value=False):
                prompt.reset_mock()
                with self.assertRaises(RuntimeError):
                    slack_token()
                prompt.assert_not_called()

    def test_recipient_ignores_production_channel(self):
        config = {
            "primary_contact": "vyasr",
            "slack": {
                "channel_id": "CPRODUCTION",
                "users": {"VyasR": "U123"},
            },
        }
        self.assertEqual(recipient(config), "U123")
        config["slack"]["users"]["VyasR"] = "CPRODUCTION"
        with self.assertRaises(ValueError):
            recipient(config)

    def test_mentions_and_thread_parent_are_preserved(self):
        message = payload("D123", "Owner <@U456>", "123.456")
        self.assertEqual(message["text"], "Owner <@U456>")
        self.assertEqual(message["thread_ts"], "123.456")
        self.assertFalse(message["unfurl_links"])

    def test_post_returns_only_delivery_identifiers(self):
        response = {"ok": True, "channel": "D123", "ts": "123.456"}
        with patch("slack_dm_test.urllib.request.urlopen") as urlopen:
            urlopen.return_value = BytesIO(json.dumps(response).encode())
            self.assertEqual(
                post("secret", payload("U123", "test")),
                {"channel": "D123", "ts": "123.456"},
            )
            request = urlopen.call_args[0][0]
            self.assertEqual(
                request.get_header("Authorization"), "Bearer secret"
            )
            self.assertEqual(json.loads(request.data)["channel"], "U123")

    def test_partial_success_and_unknown_slack_errors_remain_ambiguous(self):
        for code in ("internal_error", "fatal_error", "unknown_server_error"):
            with ExitStack() as stack:
                stack.enter_context(self.subTest(code=code))
                urlopen = stack.enter_context(
                    patch("slack_dm_test.urllib.request.urlopen")
                )
                urlopen.return_value = BytesIO(
                    json.dumps({"ok": False, "error": code}).encode()
                )
                with self.assertRaises(RuntimeError) as error:
                    post("secret", payload("U123", "test"))
                self.assertNotIsInstance(error.exception, SlackRejected)
                self.assertEqual(urlopen.call_count, 1)

    def test_failed_post_stops_without_retries_or_raw_error_echo(self):
        response = {"ok": False, "error": "invalid_auth"}
        with patch("slack_dm_test.urllib.request.urlopen") as urlopen:
            urlopen.return_value = BytesIO(json.dumps(response).encode())
            with self.assertRaisesRegex(RuntimeError, "invalid_auth"):
                post("secret", payload("U123", "test"))
            self.assertEqual(urlopen.call_count, 1)
        response["error"] = "unexpected secret-containing error"
        with patch("slack_dm_test.urllib.request.urlopen") as urlopen:
            urlopen.return_value = BytesIO(json.dumps(response).encode())
            with self.assertRaisesRegex(RuntimeError, "unknown_error"):
                post("secret", payload("U123", "test"))


if __name__ == "__main__":
    unittest.main()
