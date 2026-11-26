# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode a CUPTI trace, optionally locating an unmatched API entry at a crash."""

import argparse
import csv
import re
import struct
import sys
from pathlib import Path

RECORD = struct.Struct("<QQQQQQIIii40s")


def rows(path):
    result = []
    for i, entry in enumerate(RECORD.iter_unpack(path.read_bytes())):
        *fields, api = entry
        if fields[0] and (fields[0] - 1) % (1 << 18) == i:
            result.append((*fields, api.rstrip(b"\0").decode("ascii")))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--fault", type=Path)
    args = parser.parse_args()
    entries = rows(args.trace)
    if args.fault:
        fault = dict(
            (key, int(value, 16))
            for key, value in re.findall(
                r"^(\w+)=0x([0-9a-f]+)$", args.fault.read_text(), re.M
            )
        )
        entries = [
            row
            for row in entries
            if row[0] <= fault["trace_sequence"]
            and row[1] <= fault["trace_monotonic_ns"]
        ]
        inflight = {}
        for row in entries:
            key = (row[6], row[7], row[-1])
            if row[8] == 0:
                inflight[key] = row
            else:
                inflight.pop(key, None)
        for row in inflight.values():
            if row[6] == fault["tid"]:
                print(
                    f"Open API on fault thread: seq={row[0]} {row[-1]} "
                    f"stream={row[2]:#x} context={row[5]:#x}",
                    file=sys.stderr,
                )
        if entries:
            gaps = entries[-1][0] - entries[0][0] + 1 - len(entries)
            print(
                f"Retained sequence {entries[0][0]}..{entries[-1][0]}, "
                f"internal gaps={gaps}, write failures={fault['trace_write_failures']}",
                file=sys.stderr,
            )
    writer = csv.writer(sys.stdout, delimiter="\t")
    writer.writerow(
        "sequence ns stream object aux context tid correlation phase result api".split()
    )
    for (
        seq,
        ns,
        stream,
        obj,
        aux,
        context,
        tid,
        corr,
        phase,
        status,
        api,
    ) in entries:
        writer.writerow(
            (
                seq,
                ns,
                hex(stream),
                hex(obj),
                hex(aux),
                hex(context),
                tid,
                corr,
                "enter" if phase == 0 else "exit",
                status,
                api,
            )
        )


if __name__ == "__main__":
    main()
