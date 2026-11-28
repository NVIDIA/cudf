# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record critical GPU events during a diagnostic test process."""

from datetime import UTC, datetime

import pynvml


def watch_gpu_events(uuid, path, stopped):
    with path.open("w", buffering=1) as log:
        try:
            pynvml.nvmlInit()
            try:
                device = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                events = pynvml.nvmlEventSetCreate()
                try:
                    pynvml.nvmlDeviceRegisterEvents(
                        device, pynvml.nvmlEventTypeXidCriticalError, events
                    )
                    print(f"Watching critical GPU events for {uuid}", file=log)
                    while not stopped.is_set():
                        try:
                            event = pynvml.nvmlEventSetWait(events, 1000)
                        except pynvml.NVMLError_Timeout:
                            continue
                        print(
                            f"{datetime.now(UTC).isoformat()} Xid {event.eventData}",
                            file=log,
                        )
                    print("Stopped watching critical GPU events", file=log)
                finally:
                    pynvml.nvmlEventSetFree(events)
            finally:
                pynvml.nvmlShutdown()
        except pynvml.NVMLError as error:
            print(f"NVML event subscription unavailable: {error}", file=log)
