# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correlate a faulting stream's copies with replay-only aliased buffer lifetimes."""

import argparse
import json
import re
from bisect import bisect_right
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
from typing import NamedTuple

from read_stream_trace import rows


class Record(NamedTuple):
    seq: int
    ns: int
    stream: int
    address: int
    aux: int
    context: int
    tid: int
    correlation: int
    phase: int
    result: int
    api: str


def _key(record):
    return record.tid, record.correlation, record.api


def _clock(record):
    return record.ns, record.seq


def _successful_calls(records):
    entries = {}
    for row in records:
        if row.phase == 0:
            entries[_key(row)] = row
        else:
            start = entries.pop(_key(row), None)
            if start is not None and row.result == 0:
                yield start, row


def _copy_ranges(start, end):
    if start.api.startswith(("cuMemcpyAsync", "cuMemcpyDtoDAsync")):
        # Entry holds the endpoints. A successful exit holds the byte count.
        return [(start.address, end.address), (start.aux, end.address)]
    if start.api.startswith(("cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync")):
        return [(start.address, start.aux)]
    return []


def _allocation_range(start, end):
    if start.api.startswith("cuMemAllocFromPoolAsync"):
        return end.address, start.address
    if start.api.startswith("cuMemAlloc"):
        return end.address, end.aux
    raise ValueError(start.api)


def _intersects(a, b):
    a_start, a_size = a
    b_start, b_size = b
    return (
        a_size > 0
        and b_size > 0
        and a_start < b_start + b_size
        and b_start < a_start + a_size
    )


def _stream_dependencies(calls):
    """Track proven dependencies, omitting implicit and potentially concurrent edges."""
    timeline = sorted(
        ((row, end) for start, end in calls for row in (start, end)),
        key=lambda item: _clock(item[0]),
    )
    last = {}
    synchronized = {}
    events = {}
    recording = defaultdict(list)
    for start, end in calls:
        if start.api.startswith(("cuEventRecord", "cuEventDestroy")):
            recording[start.context, start.address].append((start.ns, end.ns))

    event_windows = {}
    for event, intervals in recording.items():
        ordered = sorted(intervals)
        event_windows[event] = (
            tuple(start for start, _ in ordered),
            tuple(accumulate((end for _, end in ordered), max)),
        )

    dependencies = {}
    for row, end in timeline:
        stream = row.context, row.stream
        thread = row.context, row.tid
        event = row.context, row.address
        name = row.api
        is_stream_operation = row.stream > 2
        if row.phase == 0:
            prior = [
                last.get(stream) if is_stream_operation else None,
                synchronized.get(thread),
            ]
            if name.startswith("cuStreamWaitEvent"):
                # A concurrent re-record or destroy makes the target event generation ambiguous.
                starts, ends = event_windows.get(event, ((), ()))
                index = bisect_right(starts, end.ns) - 1
                if index < 0 or ends[index] < row.ns:
                    prior.append(events.get(event))
            dependencies[end.seq] = tuple(t for t in prior if t is not None)
        elif name.startswith("cuStreamCreate"):
            last.pop(stream, None)
        elif name.startswith("cuStreamDestroy"):
            last.pop(stream, None)
        elif name.startswith(("cuEventCreate", "cuEventDestroy")):
            events.pop(event, None)
        else:
            if is_stream_operation:
                last[stream] = row.seq
            if name.startswith("cuEventRecord"):
                if is_stream_operation:
                    events[event] = row.seq
                else:
                    events.pop(event, None)
            if name.startswith("cuStreamSynchronize") and is_stream_operation:
                synchronized[thread] = row.seq
    return dependencies


def _is_after(calls, before, after):
    frontier = [after]
    seen = set()
    while frontier:
        step = frontier.pop()
        if step == before:
            return True
        if step not in seen:
            seen.add(step)
            frontier.extend(calls.get(step, ()))
    return False


def analyze(trace, fault, journal):
    fields = {
        k: int(v, 16)
        for k, v in re.findall(
            r"^(\w+)=0x([0-9a-f]+)$", fault.read_text(), re.M
        )
    }
    records = sorted(
        (
            Record(*row)
            for row in rows(trace)
            if row[0] <= fields["trace_sequence"]
            and row[1] <= fields["trace_monotonic_ns"]
        ),
        key=_clock,
    )
    calls = sorted(
        _successful_calls(records), key=lambda pair: _clock(pair[1])
    )
    finished = {_key(r) for r in records if r.phase == 1}
    sync = next(
        (
            r
            for r in reversed(records)
            if r.tid == fields["tid"]
            and r.phase == 0
            and r.api.startswith("cuStreamSynchronize")
            and _key(r) not in finished
        ),
        None,
    )
    if sync is None:
        print("No unmatched stream synchronization on the fault thread.")
        return
    print(
        f"Faulting sync {sync.seq}, stream {sync.stream:#x}, context {sync.context:#x}."
    )
    if not journal.exists():
        print("This process has no aliased handoff journal.")
        return
    gaps = (
        max(r.seq for r in records)
        - min(r.seq for r in records)
        + 1
        - len(records)
    )
    dropped = fields["trace_write_failures"]
    print(
        f"Retained {len(records)} callbacks, {gaps} sequence gaps, {dropped} write failures."
    )
    generations = [
        (start, end) for start, end in calls if end.context == sync.context
    ]
    created = max(
        (
            end.ns
            for _, end in generations
            if end.stream == sync.stream
            and end.api.startswith("cuStreamCreate")
        ),
        default=records[0].ns,
    )
    last_sync = max(
        (
            end.ns
            for _, end in generations
            if end.stream == sync.stream
            and created <= end.ns < sync.ns
            and end.api.startswith("cuStreamSynchronize")
        ),
        default=created,
    )
    aliases = []
    on_stream = unreadable = 0
    for line in journal.read_text().splitlines():
        try:
            ns, _tid, producer, consumer, test, ranges = json.loads(line)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        if records[0].ns <= ns <= sync.ns:
            for source, output in ranges:
                begin = max(source[2], output[2])
                end = min(source[2] + source[3], output[2] + output[3])
                aliases.append((ns, (begin, end - begin), test))
                on_stream += created <= ns and sync.stream in (
                    producer,
                    consumer,
                )
    print(
        f"{len(aliases)} aliased ranges recorded in the window, "
        f"{on_stream} tagged with this stream generation, {unreadable} unreadable journal records."
    )
    if not aliases:
        return

    allocations = [
        (end, _allocation_range(start, end))
        for start, end in generations
        if end.api.startswith("cuMemAlloc")
    ]
    frees = defaultdict(list)
    for start, end in generations:
        if end.api.startswith("cuMemFree"):
            frees[start.address].append((start, end))
    edges = _stream_dependencies(generations)
    hits = pending_hits = proven = pending_proven = ambiguous = 0
    launches = sum(
        row.phase == 0
        and row.stream == sync.stream
        and row.context == sync.context
        and last_sync <= row.ns <= sync.ns
        and row.api.startswith("cuLaunchKernel")
        for row in records
    )
    for start, end in generations:
        if (
            end.stream != sync.stream
            or start.ns < created
            or end.ns >= sync.ns
        ):
            continue
        for target in _copy_ranges(start, end):
            for ns, shared, test in aliases:
                if ns > start.ns or not _intersects(target, shared):
                    continue
                hits += 1
                is_pending = end.ns > last_sync
                pending_hits += is_pending
                allocation = next(
                    (
                        (op, region)
                        for op, region in reversed(allocations)
                        if op.ns <= ns
                        and region[0] <= shared[0]
                        and shared[0] + shared[1] <= region[0] + region[1]
                    ),
                    None,
                )
                if allocation is None:
                    ambiguous += 1
                    continue
                alloc, region = allocation
                if any(
                    alloc.ns < op.ns <= end.ns and _intersects(shared, newer)
                    for op, newer in allocations
                ):
                    ambiguous += 1
                    continue
                for release, released in frees[region[0]]:
                    if not alloc.ns < release.ns < end.ns:
                        continue
                    ordered = (
                        released.ns <= start.ns
                        if released.api == "cuMemFree_v2"
                        else _is_after(edges, released.seq, end.seq)
                    )
                    print(
                        f"Copy {start.seq}, aliased range {shared[0]:#x}+{shared[1]}, "
                        f"allocation {alloc.seq}, free {release.seq}, "
                        f"free ordered before copy={ordered}, {test}"
                    )
                    if ordered and not (gaps or dropped):
                        proven += 1
                        pending_proven += is_pending
    # Each counted hit is a range intersection, not a distinct API call.
    print(
        f"{hits} copy/range intersections ({pending_hits} after the last successful sync)."
    )
    print(f"{ambiguous} intersections with an ambiguous allocation lifetime.")
    print(
        f"{proven} intersections with an ordered free ({pending_proven} after the last successful sync)."
    )
    print(
        f"{launches} kernel launches on this stream after the last successful sync. "
        "Kernel load/store addresses are outside this trace."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fault", type=Path)
    args = parser.parse_args()
    suffix = ".fault.log"
    if not args.fault.name.endswith(suffix):
        parser.error("expected a .fault.log file")
    prefix = args.fault.name[: -len(suffix)]
    analyze(
        args.fault.with_name(prefix + ".stream.bin"),
        args.fault,
        args.fault.with_name(prefix + ".lifetime.jsonl"),
    )


if __name__ == "__main__":
    main()
