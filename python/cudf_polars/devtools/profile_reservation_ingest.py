#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profile reservation-event ingestion and write a flame graph.

Records real reservation events, then runs ``ReservationEventIngestor.ingest``
under a deterministic call-stack profiler (``sys.setprofile``). Python frames
and calls into C/Rust extension methods are both recorded, so time spent in the
generated Quent bindings shows up as leaf frames such as
``MemoryReservationRequestDynamicFsmHandle.granted_at``. Time *inside* the Rust
code is not broken down further.

The profiler adds overhead to every call, so absolute times are inflated; use
the output for relative costs. Writes ``<out>.svg`` (flame graph, open in a
browser) and ``<out>.collapsed`` (folded stacks, usable with flamegraph.pl or
speedscope).

Example::

    python devtools/profile_reservation_ingest.py --lifecycles 10000 --out /tmp/ingest
"""

from __future__ import annotations

import argparse
import collections
import html
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_reservation_events import (
    ATTRIBUTES,
    _new_buffer_resource,
    _produce,
)

from cudf_polars.quent import QuentConfig
from cudf_polars.quent._runtime import (
    QuentControllerRuntime,
    QuentWorkerRuntime,
)

if TYPE_CHECKING:
    from types import FrameType


class StackProfiler:
    """Accumulate self time per call stack, including C/Rust method calls."""

    def __init__(self) -> None:
        self.self_ns: collections.Counter[tuple[str, ...]] = collections.Counter()
        self._stack: list[str] = []
        self._start: list[int] = []
        self._child: list[int] = []

    @staticmethod
    def _frame_name(frame: FrameType) -> str:
        code = frame.f_code
        filename = code.co_filename.rsplit("/", 1)[-1]
        return f"{code.co_name} ({filename})"

    @staticmethod
    def _c_name(func: object) -> str:
        owner = getattr(func, "__self__", None)
        owner_name = type(owner).__name__ if owner is not None else None
        name = getattr(func, "__name__", repr(func))
        return f"{owner_name}.{name}" if owner_name else name

    def _push(self, name: str) -> None:
        self._stack.append(name)
        self._child.append(0)
        self._start.append(time.perf_counter_ns())

    def _pop(self) -> None:
        elapsed = time.perf_counter_ns() - self._start.pop()
        child = self._child.pop()
        self.self_ns[tuple(self._stack)] += max(elapsed - child, 0)
        self._stack.pop()
        if self._child:
            self._child[-1] += elapsed

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        """Record one Python profiler callback."""
        if event == "call":
            self._push(self._frame_name(frame))
        elif event == "c_call":
            self._push(self._c_name(arg))
        elif event in {"return", "c_return", "c_exception"} and self._stack:
            self._pop()

    def run(self, func, *args):
        """Profile ``func(*args)`` and return its result."""
        base = len(self._stack)
        self._push("<profiled>")
        sys.setprofile(self)
        try:
            return func(*args)
        finally:
            sys.setprofile(None)
            while len(self._stack) > base:
                self._pop()


def write_collapsed(profiler: StackProfiler, path: str) -> None:
    """Write folded stacks (microseconds of self time)."""
    with Path(path).open("w") as f:
        for stack, ns in sorted(profiler.self_ns.items()):
            if ns >= 1_000:
                f.write(";".join(stack) + f" {ns // 1_000}\n")


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.total = 0
        self.children: dict[str, _Node] = {}


def _build_tree(profiler: StackProfiler) -> _Node:
    root = _Node("all")
    for stack, ns in profiler.self_ns.items():
        node = root
        node.total += ns
        for name in stack:
            node = node.children.setdefault(name, _Node(name))
            node.total += ns
    return root


def _color(name: str) -> str:
    if "Handle." in name or "Observer." in name or "Context." in name:
        return "#e8743b"  # generated Quent bindings
    if name.startswith(("uuid5", "UUID", "_hashlib", "openssl")):
        return "#8e6bbf"
    if "_reservation_events.py" in name:
        return "#4c9be8"  # ingestor Python code
    return "#9aa0a6"


def write_svg(profiler: StackProfiler, path: str, title: str) -> None:
    """Render a flame graph (root at bottom, width proportional to time)."""
    root = _build_tree(profiler)
    width, row, margin = 1600, 18, 10
    scale = (width - 2 * margin) / max(root.total, 1)

    def depth(node: _Node) -> int:
        return 1 + max((depth(c) for c in node.children.values()), default=0)

    height = depth(root) * row + 60
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'font-family="monospace" font-size="11">',
        f'<text x="{margin}" y="18" font-size="14">{html.escape(title)}</text>',
        f'<text x="{margin}" y="34" fill="#666">orange = generated Quent binding call, '
        "blue = ingestor Python, purple = UUID derivation, grey = other</text>",
    ]

    def emit(node: _Node, x: float, level: int) -> None:
        w = node.total * scale
        if w < 0.5:
            return
        y = height - (level + 1) * row - margin
        pct = node.total / root.total * 100
        label = html.escape(node.name)
        tip = f"{label}: {node.total / 1e6:.1f} ms ({pct:.1f}%)"
        out.append(
            f'<g><title>{tip}</title><rect x="{x:.2f}" y="{y}" width="{w:.2f}" '
            f'height="{row - 1}" fill="{_color(node.name)}" stroke="#fff" '
            'stroke-width="0.5"/>'
        )
        chars = int(w / 6.6)
        if chars >= 4:
            text = (
                node.name if len(node.name) <= chars else node.name[: chars - 1] + "…"
            )
            out.append(
                f'<text x="{x + 2:.2f}" y="{y + row - 6}">{html.escape(text)}</text>'
            )
        out.append("</g>")
        child_x = x
        for child in sorted(node.children.values(), key=lambda c: -c.total):
            emit(child, child_x, level + 1)
            child_x += child.total * scale

    emit(root, margin, 0)
    out.append("</svg>")
    with Path(path).open("w") as f:
        f.write("\n".join(out))


def print_summary(profiler: StackProfiler, events: int, top: int) -> None:
    """Print inclusive and self time per function, as percentages."""
    total = sum(profiler.self_ns.values())
    self_by_leaf: collections.Counter[str] = collections.Counter()
    incl: collections.Counter[str] = collections.Counter()
    for stack, ns in profiler.self_ns.items():
        self_by_leaf[stack[-1]] += ns
        for name in set(stack):
            incl[name] += ns
    print(
        f"profiled time {total / 1e6:.0f} ms for {events} events "
        f"({total / events / 1e3:.1f} us/event, inflated by profiler overhead)"
    )
    print("\ninclusive time:")
    for name, ns in incl.most_common(top):
        print(f"  {ns / total * 100:5.1f}%  {name}")
    print("\nself time:")
    for name, ns in self_by_leaf.most_common(top):
        print(f"  {ns / total * 100:5.1f}%  {name}")


def main() -> None:
    """Record events, profile ingestion, write the flame graph."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lifecycles", type=int, default=10_000)
    parser.add_argument("--attributes", choices=sorted(ATTRIBUTES), default="three")
    parser.add_argument("--out", default="/tmp/reservation_ingest_profile")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    attributes = ATTRIBUTES[args.attributes]
    br = _new_buffer_resource()
    with tempfile.TemporaryDirectory(prefix="reservation-profile-") as root:
        config = QuentConfig(output_root=root)
        controller = QuentControllerRuntime.create(config, backend="profile")
        worker = QuentWorkerRuntime.create(
            config,
            controller.collector_address,
            worker_id=uuid.uuid4(),
            rank=0,
            nranks=1,
            instance_name="profile",
            buffer_resource=br,
        )
        try:
            _produce(br, args.lifecycles, attributes)
            events = br.drain_reservation_events()
            n_events = len(events)
            ingestor = worker._reservation_ingestor
            assert ingestor is not None
            profiler = StackProfiler()
            start = time.perf_counter()
            profiler.run(ingestor.ingest, events)
            wall = time.perf_counter() - start
            print(f"ingest wall time under profiler: {wall:.2f}s", flush=True)
        finally:
            # The ingestor holds live Quent handles; drop our reference so the
            # session can close and the Collector can shut down.
            ingestor = None
            worker.close()
            controller.close()

    write_collapsed(profiler, f"{args.out}.collapsed")
    write_svg(
        profiler,
        f"{args.out}.svg",
        f"ReservationEventIngestor.ingest: {n_events} events, "
        f"attributes={args.attributes}",
    )
    print_summary(profiler, n_events, args.top)
    print(f"\nwrote {args.out}.svg and {args.out}.collapsed")


if __name__ == "__main__":
    main()
