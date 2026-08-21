# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Demonstrate regex searching of a text file with pylibcudf.

This example intentionally focuses on the GPU data path rather than emulating
the full grep command-line interface. It reports both end-to-end throughput
(file read plus regex scan) and scan-only throughput (text already resident on
the GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class BenchmarkResult:
    """Measurements from one GDS configuration."""

    gds: bool
    bytes: int
    lines: int
    matches: int
    end_to_end_seconds: float
    scan_seconds: float
    pylibcudf_version: str = "unknown"

    @property
    def end_to_end_gbps(self) -> float:
        """Return end-to-end throughput in decimal GB/s."""
        return self.bytes / self.end_to_end_seconds / 1e9

    @property
    def scan_gbps(self) -> float:
        """Return resident-data regex throughput in decimal GB/s."""
        return self.bytes / self.scan_seconds / 1e9


def _write_line(text: str, *, error: bool = False) -> None:
    """Write one newline-terminated message to stdout or stderr."""
    stream = sys.stderr if error else sys.stdout
    stream.write(text + "\n")


def configure_gds(enabled: bool) -> None:
    """Select the KvikIO I/O path before importing pylibcudf.

    KvikIO compatibility mode uses POSIX I/O. Disabling compatibility mode
    selects the cuFile path used for GPUDirect Storage.
    """
    os.environ["KVIKIO_COMPAT_MODE"] = "OFF" if enabled else "ON"


def _require_native_gds() -> None:
    """Raise when cuFile would use its internal compatibility fallback."""
    try:
        from kvikio import cufile_driver
    except ImportError as exc:
        raise RuntimeError(
            "native GDS validation requires the kvikio Python package"
        ) from exc
    if not cufile_driver.get("is_gds_available"):
        raise RuntimeError(
            "native GDS is unavailable; cuFile would use its internal "
            "compatibility fallback"
        )
    if cufile_driver.get("allow_compat_mode"):
        raise RuntimeError(
            "native GDS benchmarking requires cuFile allow_compat_mode=false"
        )


def _import_gpu_modules() -> tuple[Any, Any]:
    """Import pylibcudf and the default stream after GDS configuration."""
    try:
        from rmm.pylibrmm.stream import DEFAULT_STREAM

        import pylibcudf as plc
    except ImportError as exc:  # pragma: no cover - depends on GPU environment
        raise RuntimeError(
            "pylibcudf and RMM are required; install a RAPIDS build that "
            "matches the system CUDA version"
        ) from exc
    return plc, DEFAULT_STREAM


def _package_version(import_name: str) -> str:
    """Return the installed distribution version for an import package."""
    distributions = metadata.packages_distributions().get(import_name, [])
    for distribution in distributions:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def read_lines(path: str, plc: Any, stream: Any) -> Any:
    """Read ``path`` into a pylibcudf strings column, one row per line."""
    source = plc.io.text.make_source_from_file(path)
    options = plc.io.text.ParseOptions(strip_delimiters=True)
    return plc.io.text.multibyte_split(
        source, "\n", options=options, stream=stream
    )


def compile_regex(pattern: str, ignore_case: bool, plc: Any) -> Any:
    """Compile a libcudf regex program."""
    regex_flags = plc.strings.regex_flags.RegexFlags
    if ignore_case:
        try:
            flags = regex_flags.IGNORECASE
        except AttributeError as exc:
            version = _package_version("pylibcudf")
            raise RuntimeError(
                "--ignore-case requires a pylibcudf build that exposes "
                "RegexFlags.IGNORECASE; the installed pylibcudf version "
                f"is {version}. Install a current RAPIDS nightly build or "
                "omit --ignore-case"
            ) from exc
    else:
        flags = regex_flags.DEFAULT
    return plc.strings.regex_program.RegexProgram.create(pattern, flags)


def scan_lines(lines: Any, program: Any, plc: Any, stream: Any) -> Any:
    """Return a Boolean column indicating which lines match ``program``."""
    return plc.strings.contains.contains_re(lines, program, stream=stream)


def count_matches(lines: Any, mask: Any, plc: Any, stream: Any) -> int:
    """Count selected rows without converting the strings column to pandas."""
    selected = plc.stream_compaction.apply_boolean_mask(
        plc.Table([lines]), mask, stream=stream
    )
    stream.synchronize()
    return selected.num_rows()


def _best_time(operation: Any, stream: Any, repeats: int) -> float:
    """Return the best synchronized runtime from ``repeats`` executions."""
    best = float("inf")
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = operation()
        stream.synchronize()
        best = min(best, time.perf_counter() - start)
        del result
    return best


def benchmark_file(
    path: str,
    pattern: str,
    *,
    gds: bool,
    ignore_case: bool = False,
    repeats: int = 3,
    plc: Any | None = None,
    stream: Any | None = None,
) -> BenchmarkResult:
    """Benchmark text loading and regex scanning for one GDS configuration."""
    if plc is None or stream is None:
        configure_gds(gds)
        if gds:
            _require_native_gds()
        plc, stream = _import_gpu_modules()

    program = compile_regex(pattern, ignore_case, plc)
    resident_lines = read_lines(path, plc, stream)
    resident_mask = scan_lines(resident_lines, program, plc, stream)
    stream.synchronize()

    matches = count_matches(
        resident_lines, resident_mask, plc=plc, stream=stream
    )

    scan_seconds = _best_time(
        lambda: scan_lines(resident_lines, program, plc, stream),
        stream,
        repeats,
    )

    def load_and_scan() -> Any:
        lines = read_lines(path, plc, stream)
        mask = scan_lines(lines, program, plc, stream)
        # Retain the input until the caller synchronizes the CUDA stream.
        return lines, mask

    end_to_end_seconds = _best_time(load_and_scan, stream, repeats)
    return BenchmarkResult(
        gds=gds,
        bytes=os.path.getsize(path),
        lines=resident_lines.size(),
        matches=matches,
        end_to_end_seconds=end_to_end_seconds,
        scan_seconds=scan_seconds,
        pylibcudf_version=_package_version("pylibcudf"),
    )


def _print_result(result: BenchmarkResult, as_json: bool) -> None:
    """Print one benchmark result."""
    if as_json:
        payload = asdict(result)
        payload["end_to_end_gbps"] = result.end_to_end_gbps
        payload["scan_gbps"] = result.scan_gbps
        _write_line(json.dumps(payload, sort_keys=True))
        return

    mode = "on" if result.gds else "off"
    _write_line(
        f"GDS={mode}  size={result.bytes / 1e9:.2f} GB  "
        f"lines={result.lines}  matches={result.matches}  "
        f"pylibcudf={result.pylibcudf_version}"
    )
    _write_line(
        f"  end-to-end: {result.end_to_end_seconds:.6f} s  "
        f"{result.end_to_end_gbps:.2f} GB/s"
    )
    _write_line(
        f"  scan-only:  {result.scan_seconds:.6f} s  "
        f"{result.scan_gbps:.2f} GB/s"
    )


def _result_from_json(output: str) -> BenchmarkResult:
    """Parse the last JSON line emitted by a benchmark child process."""
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("benchmark child produced no output")
    payload = json.loads(lines[-1])
    fields = BenchmarkResult.__dataclass_fields__
    return BenchmarkResult(**{name: payload[name] for name in fields})


def _results_agree(results: Sequence[BenchmarkResult]) -> bool:
    """Return whether all modes produced identical input and match data."""
    signatures = {(r.bytes, r.lines, r.matches) for r in results}
    return len(signatures) == 1


def _run_both_modes(args: argparse.Namespace) -> int:
    """Run GDS off/on in fresh processes so KvikIO is reinitialized."""
    results = []
    script = str(Path(__file__).resolve())
    for mode in ("off", "on"):
        command = [
            sys.executable,
            script,
            args.path,
            args.pattern,
            "--gds-mode",
            mode,
            "--repeats",
            str(args.repeats),
            "--json",
        ]
        if args.ignore_case:
            command.append("--ignore-case")
        if args.expected_matches is not None:
            command.extend(["--expected-matches", str(args.expected_matches)])
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        if completed.returncode != 0:
            if completed.stdout:
                sys.stdout.write(completed.stdout)
            if completed.stderr:
                sys.stderr.write(completed.stderr)
            return completed.returncode
        try:
            results.append(_result_from_json(completed.stdout))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            _write_line(
                f"regex_search: invalid {mode} child result: {exc}", error=True
            )
            return 2

    if not _results_agree(results):
        _write_line(
            "regex_search: GDS modes produced different file, line, or match counts",
            error=True,
        )
        return 2
    for result in results:
        _print_result(result, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the example's intentionally small argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark pylibcudf text regex performance."
    )
    parser.add_argument("path", help="plain-text input file")
    parser.add_argument("pattern", help="libcudf regular expression")
    parser.add_argument(
        "--gds-mode",
        choices=("off", "on", "both"),
        default="off",
        help="I/O mode to benchmark (default: off)",
    )
    parser.add_argument("--ignore-case", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--expected-matches",
        type=int,
        help="fail if the GPU result does not match this known count",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pylibcudf regex performance example."""
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        _write_line("--repeats must be at least 1", error=True)
        return 2
    if args.expected_matches is not None and args.expected_matches < 0:
        _write_line("--expected-matches cannot be negative", error=True)
        return 2
    if args.gds_mode == "both":
        return _run_both_modes(args)

    gds = args.gds_mode == "on"
    configure_gds(gds)
    try:
        result = benchmark_file(
            args.path,
            args.pattern,
            gds=gds,
            ignore_case=args.ignore_case,
            repeats=args.repeats,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _write_line(f"regex_search: {exc}", error=True)
        return 2
    if (
        args.expected_matches is not None
        and result.matches != args.expected_matches
    ):
        _write_line(
            f"regex_search: expected {args.expected_matches} matches, "
            f"got {result.matches}",
            error=True,
        )
        return 2
    _print_result(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
