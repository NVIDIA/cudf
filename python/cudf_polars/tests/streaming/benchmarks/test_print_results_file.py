# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the benchmark results-file printer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from cudf_polars.streaming.benchmarks.print_results_file import (
    _io_summaries,
    iter_records,
    load_runs,
    main,
    parse_args,
)
from cudf_polars.streaming.benchmarks.utils import SuccessRecord

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _summary(**overrides: Any) -> dict[str, Any]:
    """One rank's I/O summary."""
    return {
        "num_ops": 10,
        "bytes_read": 1024,
        "busy_ns": 1_000_000,
        "busy_fraction": 0.5,
        "busy_bytes_per_sec": 1e9,
        "by_backend": {"POSIX": {"num_ops": 10}, "GDS": {"num_ops": 0}},
        **overrides,
    }


def _record(**overrides: Any) -> dict[str, Any]:
    """One successful iteration."""
    return {
        "query": 1,
        "iteration": 0,
        "duration": 0.25,
        "status": "success",
        **overrides,
    }


def _failed(**overrides: Any) -> dict[str, Any]:
    """One failed iteration."""
    return {
        "query": 1,
        "iteration": 0,
        "status": "error",
        "traceback": "boom",
        **overrides,
    }


def _run(*records: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """A run holding ``records``, grouped by query as the runners write them."""
    by_query: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_query.setdefault(str(record["query"]), []).append(record)
    return {
        "run_id": "abc",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "engine_name": "cudf-polars",
        "frontend": "ray",
        "dataset_path": "/data",
        "scale_factor": 10,
        "n_workers": 2,
        "iterations": 1,
        "records": by_query,
        **overrides,
    }


@pytest.fixture
def report(tmp_path: Path, capsys: pytest.CaptureFixture) -> Callable[..., str]:
    """Write runs as the NDJSON the runners append, print them, return the output."""

    def run(*runs: dict[str, Any], args: tuple[str, ...] = ()) -> str:
        path = tmp_path / "out.json"
        path.write_text("".join(f"{json.dumps(r)}\n" for r in runs))
        main(parse_args([str(path), *args]))
        return capsys.readouterr().out

    return run


def test_load_runs_reads_every_line(tmp_path: Path) -> None:
    """Each line is a run, and blank lines are skipped."""
    path = tmp_path / "out.json"
    path.write_text(
        f"{json.dumps(_run(run_id='first'))}\n\n{json.dumps(_run(run_id='second'))}\n"
    )
    assert [r["run_id"] for r in load_runs(path)] == ["first", "second"]


def test_load_runs_rejects_an_empty_file(tmp_path: Path) -> None:
    """An empty file is an error rather than an empty report."""
    path = tmp_path / "empty.json"
    path.write_text("")
    with pytest.raises(ValueError, match="no runs"):
        load_runs(path)


def test_iter_records_orders_by_query() -> None:
    """Records come out grouped by query, whatever order the keys are in."""
    run = _run(_record(query=3), _record(query=1), _record(query=1, iteration=1))
    assert [(r.query, r.iteration) for r in iter_records(run)] == [
        (1, 0),
        (1, 1),
        (3, 0),
    ]


@pytest.mark.parametrize(
    "raw, expected",
    [
        # JSON object keys are strings, so they are cast back to ranks.
        ({"0": {}, "1": {}}, [0, 1]),
        # A run made without statistics has nothing to report.
        (None, []),
    ],
)
def test_io_summaries_keys(raw: Any, expected: list[int]) -> None:
    """Rank keys are integers, and a run without statistics yields nothing."""
    record = SuccessRecord(query=1, iteration=0, duration=0.5, io_summaries=raw)
    assert sorted(_io_summaries(record)) == expected


def test_an_unreadable_older_run_does_not_stop_the_report(
    report: Callable[..., str],
) -> None:
    """The default path shows the last run, so an older bad one is not touched."""
    out = report(_run(_record(io_summaries=[{}]), run_id="old"), _run(run_id="new"))
    assert "new" in out


def test_prints_timings_and_io(report: Callable[..., str]) -> None:
    """The last run is printed, with a row per rank per iteration."""
    out = report(_run(_record(io_summaries={"0": _summary(), "1": _summary()})))
    assert "Timings" in out
    assert "0.2500s" in out
    # One row per rank, naming the backend that carried the work.
    assert out.count("POSIX") == 2


def test_reports_when_io_was_not_collected(report: Callable[..., str]) -> None:
    """A run made without --rapidsmpf-statistics says so rather than printing nothing."""
    out = report(_run(_record()))
    assert "not collected" in out
    assert "I/O per rank" not in out


def test_failed_iterations_are_left_out_of_the_timings(
    report: Callable[..., str],
) -> None:
    """A query whose every iteration failed has no min, max or mean to report."""
    assert "no successful iterations" in report(_run(_failed()))


def test_a_rank_that_read_nothing_is_called_out(report: Callable[..., str]) -> None:
    """Zero bytes on one rank has no finite ratio, so it gets its own line."""
    out = report(
        _run(
            _record(
                io_summaries={
                    "0": _summary(bytes_read=4096),
                    "1": _summary(
                        num_ops=0,
                        bytes_read=0,
                        busy_ns=0,
                        busy_fraction=0.0,
                        busy_bytes_per_sec=0.0,
                        by_backend={"POSIX": {"num_ops": 0}},
                    ),
                }
            )
        )
    )
    assert "ranks that read nothing while peers did: q1 iter 0" in out


def test_read_skew_is_reported_as_a_ratio(report: Callable[..., str]) -> None:
    """With every rank reading, the spread is the max over the min."""
    out = report(
        _run(
            _record(
                io_summaries={
                    "0": _summary(bytes_read=4000),
                    "1": _summary(bytes_read=1000),
                }
            )
        )
    )
    assert "widest read skew: 4.00x" in out


def test_all_prints_every_run(report: Callable[..., str]) -> None:
    """Without --all only the last run is printed, since --output appends."""
    runs = (_run(run_id="older"), _run(run_id="newer"))

    out = report(*runs)
    assert "newer" in out
    assert "older" not in out

    out = report(*runs, args=("--all",))
    assert "older" in out
    assert "newer" in out


def test_no_io_skips_the_io_section(report: Callable[..., str]) -> None:
    """--no-io leaves the timings alone."""
    out = report(_run(_record(io_summaries={"0": _summary()})), args=("--no-io",))
    assert "Timings" in out
    assert "I/O per rank" not in out
