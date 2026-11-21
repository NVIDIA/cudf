# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check private output replacement and diagnostic redaction boundaries."""

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from progress import read_stage, record_stage
from slack_dm_test import SlackRejected, post
from storage import write_private


class StorageTests(unittest.TestCase):
    def test_replacing_permissive_output_is_atomic_and_owner_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text("old")
            path.chmod(0o644)
            write_private(path, "private")
            self.assertEqual(path.read_text(), "private")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with patch(
                "storage.os.replace", side_effect=OSError("disk failure")
            ):
                with self.assertRaises(OSError):
                    write_private(path, "incomplete")
            self.assertEqual(path.read_text(), "private")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_operational_stage_rejects_arbitrary_or_malformed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            record_stage(path, "project_sync")
            self.assertEqual(read_stage(path), "project_sync")
            for data in (
                {"stage": "PRIVATE INFO"},
                {"stage": []},
                [],
                None,
                "secret",
            ):
                (path / "progress.json").write_text(json.dumps(data))
                self.assertEqual(read_stage(path), "startup")

    def test_slack_api_rejections_and_rate_limits_are_definite(self):
        response = BytesIO(b'{"ok":false,"error":"invalid_auth"}')
        with patch(
            "slack_dm_test.urllib.request.urlopen", return_value=response
        ):
            with self.assertRaises(SlackRejected):
                post("secret", {"channel": "U123"})
        error = HTTPError("https://slack.com", 429, "rate limited", {}, None)
        with patch("slack_dm_test.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(SlackRejected):
                post("secret", {"channel": "U123"})
