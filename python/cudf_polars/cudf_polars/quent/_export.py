# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export Quent events to an archive."""

from __future__ import annotations

import json
import zipfile
from typing import TYPE_CHECKING, Any

from cudf_polars.quent._types import EventName, new_quent_id

if TYPE_CHECKING:
    import uuid
    from pathlib import Path


SIDECAR_FILE_NAME = "model.qmi"
EXTENSION = "ndjson"

MODEL_QMI: dict[str, Any] = {
    "quent": {
        "version": "0.1.0",
        "commit": "153d422ae3392c24dfaf6ac5743a8682f783f864",
        "remote": "https://github.com/rapidsai/quent",
    },
    "model": {
        "name": "Simulator",
        "package": "quent-simulator-instrumentation",
        "type_path": "quent_simulator_instrumentation::SimulatorEvent",
        "source": {
            "version": "0.1.0",
            "commit": "153d422ae3392c24dfaf6ac5743a8682f783f864",
            "remote": "https://github.com/rapidsai/quent",
        },
        "analyzer_package": "quent-simulator-analyzer",
    },
}

ENTITY_DIRECTORIES: dict[str, str] = {
    EventName.ENGINE.value: "engine",
    EventName.WORKER.value: "worker",
    EventName.QUERY_GROUP.value: "query_group",
    EventName.QUERY.value: "query",
    EventName.PLAN.value: "plan",
    EventName.OPERATOR.value: "operator",
    EventName.PORT.value: "port",
    EventName.TASK.value: "task",
}


def unwrap_event_data(data: dict[str, Any]) -> tuple[str, Any]:
    """
    Extract the entity name and unwrapped payload from a buffered event.

    Buffered events wrap payloads as ``{"Engine": {...}}``; archive export
    stores the payload directly because the entity type is implied by the
    subdirectory name.
    """
    if len(data) != 1:
        msg = (
            "Expected event data with exactly one entity wrapper, "
            f"got {len(data)} keys: {sorted(data)}"
        )
        raise ValueError(msg)
    entity_name, payload = next(iter(data.items()))
    if entity_name not in ENTITY_DIRECTORIES:
        msg = f"Unknown Quent entity type: {entity_name!r}"
        raise ValueError(msg)
    return entity_name, payload


def to_export_line(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert a buffered event envelope to archive export line format."""
    entity_name, payload = unwrap_event_data(event["data"])
    directory = ENTITY_DIRECTORIES[entity_name]
    export_line = {
        "id": event["id"],
        "timestamp": event["timestamp"],
        "data": payload,
    }
    return directory, export_line


def write_quent_export(
    events: list[dict[str, Any]],
    export_root: Path,
    context_id: uuid.UUID,
    quent_archive: Path,
    *,
    sidecar: dict[str, Any] | None = None,
) -> Path:
    """
    Write Quent events to a ZIP archive.

    Parameters
    ----------
    events
        Buffered Quent event envelopes from ``engine._quent_events``.
    export_root
        Directory for exported archives (e.g. ``logs``).
    context_id
        Context UUID, typically the engine/run id.
    quent_archive
        Quent archive path. The archive will be written to this path.
    sidecar
        Optional provenance payload for ``model.qmi``. Defaults to
        :data:`MODEL_QMI`.

    Returns
    -------
    Path
        The archive path ``export_root/<context_id>.zip``. The archive contains
        the Quent export layout under a top-level ``<context_id>/`` directory.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        directory, export_line = to_export_line(event)
        grouped.setdefault(directory, []).append(export_line)

    export_root.mkdir(parents=True, exist_ok=True)
    tmp_path = export_root / f".{context_id}.zip.tmp"
    context_dir = str(context_id)

    with zipfile.ZipFile(
        tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            f"{context_dir}/{SIDECAR_FILE_NAME}",
            json.dumps(MODEL_QMI if sidecar is None else sidecar, indent=2) + "\n",
        )
        for directory, lines in grouped.items():
            stream_path = f"{context_dir}/{directory}/{new_quent_id()}.{EXTENSION}"
            contents = [json.dumps(line, separators=(",", ":")) for line in lines]
            archive.writestr(stream_path, "\n".join(contents) + "\n")

    tmp_path.replace(quent_archive)
    return quent_archive
