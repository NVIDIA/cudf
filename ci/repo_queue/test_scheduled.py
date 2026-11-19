# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise scheduler boundaries and the isolated threaded reminder rehearsal."""

import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from progress import read_stage, record_stage
from rehearse import KEY, rehearse
from rehearse import main as rehearsal_main
from scheduled import command, main
from test_actions import CONFIG


class ScheduledTests(unittest.TestCase):
    def test_modes_keep_the_same_selection_and_explicit_write_flags(self):
        selection = {"numbers": [24176, 23772]}
        preview = command(selection, "preview", Path("/tmp/queue"))
        self.assertIn("24176", preview)
        self.assertIn("23772", preview)
        self.assertNotIn("--apply", preview)
        self.assertEqual(
            command(selection, "refresh", Path("/tmp/queue")),
            [*preview, "--apply"],
        )
        self.assertEqual(
            command(selection, "dm", Path("/tmp/queue")),
            [*preview, "--apply", "--send-dm"],
        )
        rehearsal = command(selection, "rehearsal", Path("/tmp/queue"))
        self.assertTrue(rehearsal[1].endswith("rehearse.py"))
        self.assertNotIn("--apply", rehearsal)

    def test_invalid_selection_is_rejected_before_a_child_can_run(self):
        for numbers in (
            None,
            [],
            [True],
            ["23772"],
            [0],
            [-1],
            [1, 1],
            list(range(1, 12)),
        ):
            with self.subTest(numbers=numbers), self.assertRaises(ValueError):
                command({"numbers": numbers}, "dm", Path("/tmp/queue"))

    def test_non_object_selection_uses_fixed_cli_error_without_running_child(
        self,
    ):
        for value in ([], None, "private value", 42):
            with self.subTest(value=value), ExitStack() as stack:
                directory = Path(
                    stack.enter_context(tempfile.TemporaryDirectory())
                )
                selection = directory / "selection.json"
                selection.write_text(json.dumps(value))
                stack.enter_context(
                    patch(
                        "sys.argv",
                        [
                            "scheduled.py",
                            "--selection",
                            str(selection),
                            "--output-dir",
                            str(directory),
                        ],
                    )
                )
                stderr = stack.enter_context(patch("sys.stderr", StringIO()))
                run = stack.enter_context(patch("scheduled.subprocess.run"))
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(
                    "Invalid tracking selection configuration",
                    stderr.getvalue(),
                )
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertNotIn("private value", stderr.getvalue())
                run.assert_not_called()

    def test_missing_slack_token_fails_before_any_project_mutation(self):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(
                patch(
                    "sys.argv",
                    [
                        "scheduled.py",
                        "--mode",
                        "dm",
                        "--output-dir",
                        "/tmp/queue",
                    ],
                )
            )
            run = stack.enter_context(patch("scheduled.subprocess.run"))
            stack.enter_context(patch("sys.stderr", StringIO()))
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        run.assert_not_called()

    def test_child_output_is_private_on_success_and_failure(self):
        for status in (0, 7):
            with ExitStack() as stack:
                directory = stack.enter_context(tempfile.TemporaryDirectory())
                public = stack.enter_context(redirect_stdout(StringIO()))
                stack.enter_context(patch.dict(os.environ, {}, clear=True))
                stack.enter_context(
                    patch(
                        "sys.argv", ["scheduled.py", "--output-dir", directory]
                    )
                )

                def child(command, **kwargs):
                    record_stage(Path(directory), "project_sync")
                    kwargs["stdout"].write("PRIVATE PROJECT STATE\n")
                    kwargs["stderr"].write("PRIVATE DM RECEIPT\n")
                    return CompletedProcess(command, status)

                stack.enter_context(
                    patch("scheduled.subprocess.run", side_effect=child)
                )
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, status)
                self.assertNotIn("PRIVATE", public.getvalue())
                if status:
                    self.assertIn("project_sync (exit 7)", public.getvalue())
                private = (Path(directory) / "diagnostics.log").read_text()
                self.assertIn("PRIVATE PROJECT STATE", private)
                self.assertIn("PRIVATE DM RECEIPT", private)
                self.assertEqual(
                    (Path(directory) / "diagnostics.log").stat().st_mode
                    & 0o077,
                    0,
                )


class RehearsalTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(CONFIG)
        self.config["slack"]["users"] = {
            "reviewer": "UREVIEWER",
            "vyasr": "U01PXPJ5741",
        }
        self.messages = []

    def sender(self, token, message):
        self.messages.append(message)
        return {"channel": "D123", "ts": f"{len(self.messages)}.0"}

    def test_threaded_sequence_and_rerun_use_durable_isolated_history(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            directory = Path(tmp)
            result = rehearse(
                self.config, directory, self.sender, "2026-11-16T12:00:00Z"
            )
            again = rehearse(
                self.config, directory, self.sender, "2026-12-01T12:00:00Z"
            )
            state = json.loads((directory / "cards.json").read_text())
            receipts = json.loads((directory / "dm-receipts.json").read_text())
        self.assertEqual(result["sent"], ["initial", "reminder", "escalation"])
        self.assertEqual(again["sent"], [])
        self.assertEqual(result["project_writes"], 0)
        self.assertEqual(len(self.messages), 3)
        self.assertNotIn("thread_ts", self.messages[0])
        for message in self.messages[1:]:
            self.assertEqual(message["thread_ts"], "1.0")
        for message in self.messages:
            self.assertEqual(message["channel"], "U01PXPJ5741")
            self.assertIn("DM REHEARSAL", message["text"])
        self.assertIn("<@UREVIEWER> <@U01PXPJ5741>", self.messages[2]["text"])
        self.assertEqual(len(receipts), 3)
        self.assertEqual(
            list(state[KEY]["data"]["deliveries"]),
            ["dm-rehearsal:U01PXPJ5741"],
        )

    def test_rehearsal_cli_reports_delivery_in_scheduler_output_directory(
        self,
    ):
        with ExitStack() as stack:
            directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory())
            )
            stack.enter_context(
                patch(
                    "sys.argv",
                    [
                        "rehearse.py",
                        "--send-dm",
                        "--output-dir",
                        str(directory),
                    ],
                )
            )
            stack.enter_context(
                patch("rehearse.slack_token", return_value="secret")
            )
            stack.enter_context(
                patch(
                    "rehearse.post",
                    side_effect=RuntimeError("Slack unavailable"),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "Slack unavailable"):
                rehearsal_main()
            self.assertEqual(read_stage(directory), "slack_delivery")

    def test_missing_rehearsal_recipient_reports_configuration(self):
        config = deepcopy(self.config)
        config["slack"]["users"].pop(config["primary_contact"])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Slack user mapping"):
                rehearse(
                    config, directory, self.sender, "2026-11-16T12:00:00Z"
                )
            self.assertEqual(read_stage(directory), "configuration")
            self.assertFalse((directory / "cards.json").exists())
        self.assertEqual(self.messages, [])

    def test_rehearsal_reports_configuration_and_recovery_failures(self):
        for phase in ("configuration", "receipt_recovery"):
            with ExitStack() as stack:
                stack.enter_context(self.subTest(phase=phase))
                directory = Path(
                    stack.enter_context(tempfile.TemporaryDirectory())
                )
                if phase == "receipt_recovery":
                    (directory / "cards.json").write_text("invalid JSON")
                start = (
                    "invalid time"
                    if phase == "configuration"
                    else "2026-11-16T12:00:00Z"
                )
                with self.assertRaises(ValueError):
                    rehearse(self.config, directory, self.sender, start)
                self.assertEqual(read_stage(directory), phase)
        self.assertEqual(self.messages, [])

    def test_ambiguous_send_blocks_a_restart_instead_of_duplicating(self):
        def uncertain(token, message):
            raise RuntimeError("connection lost")

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            directory = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                rehearse(
                    self.config, directory, uncertain, "2026-11-16T12:00:00Z"
                )
            with self.assertRaisesRegex(
                RuntimeError, "Uncertain rehearsal send"
            ):
                rehearse(
                    self.config, directory, self.sender, "2026-11-16T12:00:00Z"
                )
        self.assertEqual(self.messages, [])


if __name__ == "__main__":
    unittest.main()
