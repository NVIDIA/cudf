# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose fixed operational stages without publishing tracking or delivery data."""

import json

from storage import save_json

STAGES = {
    "startup",
    "configuration",
    "project_read",
    "receipt_recovery",
    "source_collection",
    "project_sync",
    "slack_delivery",
    "verification",
    "complete",
}


def record_stage(directory, name):
    if name not in STAGES:
        raise ValueError("Unknown operational stage")
    save_json(directory / "progress.json", {"stage": name})


def read_stage(directory):
    try:
        data = json.loads((directory / "progress.json").read_text())
        stage = data.get("stage") if isinstance(data, dict) else None
        return stage if stage in STAGES else "startup"
    except (OSError, ValueError, TypeError):
        return "startup"
