# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import threading
from typing import TYPE_CHECKING, Any

import pytest

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.streaming.actor_graph.collectives import closing
from cudf_polars.streaming.benchmarks.pdsds import PDSDSPolarsQueries
from cudf_polars.streaming.benchmarks.pdsh import PDSHQueries
from cudf_polars.streaming.benchmarks.utils import (
    POLARS_VALIDATION_OPTIONS,
    RunOptions,
    ValidationMethod,
    _add_dataset_args,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Generator, Iterator


class FakeCollective:
    """
    Fake rapidsmpf collective whose peers never finish until released.

    The calls that wait for peers, ``extract_all()`` and ``insert_finished()``
    (unless ``finish_waits`` is false), set ``waiting`` and never return.
    ``aclose()`` sets ``closing`` and returns only after ``release()``. It
    works from any event loop.
    """

    def __init__(self) -> None:
        self.finish_waits = True
        self.waiting = asyncio.Event()
        self.closing = asyncio.Event()
        self.close_calls = 0
        self._peers_finished = threading.Event()

    def release(self) -> None:
        self._peers_finished.set()

    def __enter__(self) -> FakeCollective:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    async def __aenter__(self) -> FakeCollective:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    def insert(self, *args: object) -> None:
        pass

    def extract(self, *args: object) -> list[Any]:
        return []

    async def extract_all(self, *, ordered: bool) -> list[Any]:
        return await self._wait_for_peers()

    async def insert_finished(self) -> None:
        if self.finish_waits:
            await self._wait_for_peers()

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closing.set()
        while not self._peers_finished.is_set():
            await asyncio.sleep(0.001)

    async def _wait_for_peers(self) -> list[Any]:
        self.waiting.set()
        await asyncio.Event().wait()
        return []


@pytest.fixture
def fake_collective() -> Iterator[FakeCollective]:
    collective = FakeCollective()
    yield collective
    collective.release()


@pytest.fixture
def cancel_when(
    timeout_seconds: int,
) -> Callable[[Callable[[], Coroutine[Any, Any, Any]], asyncio.Event], None]:
    """Run a coroutine, cancel it once an event is set, and check it exits."""

    def run(
        make_coroutine: Callable[[], Coroutine[Any, Any, Any]], event: asyncio.Event
    ) -> None:
        async def main() -> None:
            task = asyncio.create_task(make_coroutine())
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=timeout_seconds)

        asyncio.run(main())

    return run


@pytest.fixture
def leave_pending_close(
    pending_closes: dict[Any, Any],
    fake_collective: FakeCollective,
    cancel_when: Callable[
        [Callable[[], Coroutine[Any, Any, Any]], asyncio.Event], None
    ],
) -> Callable[[Any, int], None]:
    """Leave a cancelled close of ``fake_collective`` pending on a communicator."""

    def leave(comm: Any, op_id: int) -> None:
        async def body() -> None:
            async with closing.aclosing_collective(fake_collective, comm, op_id):
                await fake_collective.extract_all(ordered=True)

        cancel_when(body, fake_collective.waiting)

    return leave


@pytest.fixture
def assert_pending_then_released(
    pending_closes: dict[Any, Any], timeout_seconds: int
) -> Callable[[Any, FakeCollective, int], None]:
    """Check a close is pending on a communicator until its collective is released."""

    def check(comm: Any, collective: FakeCollective, op_id: int) -> None:
        assert [close.op_id for close in pending_closes[comm]] == [op_id]
        collective.release()
        closing.wait_for_pending_closes(comm, timeout=timeout_seconds)
        assert pending_closes == {}

    return check


class PytestGroupAdapter:
    """Adapts a pytest option group to the argparse group interface."""

    def __init__(self, group: Any) -> None:
        self._group = group

    def add_argument(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("action") is argparse.BooleanOptionalAction:
            dest = kwargs.get("dest") or args[0].lstrip("-").replace("-", "_")
            self._group.addoption(
                *args,
                dest=dest,
                action="store_true",
                default=None,
                help=kwargs.get("help", ""),
            )
            for opt in args:
                self._group.addoption(
                    f"--no-{opt.lstrip('-')}",
                    dest=dest,
                    action="store_false",
                )
        else:
            kwargs.pop("metavar", None)
            self._group.addoption(*args, **kwargs)

    def add_argument_group(self, *args: Any, **kwargs: Any) -> PytestGroupAdapter:
        return self


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("TPC benchmark options")
    adapter = PytestGroupAdapter(group)
    StreamingOptions._add_cli_args(adapter)  # type: ignore[arg-type]
    _add_dataset_args(adapter)  # type: ignore[arg-type]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "q_id" not in metafunc.fixturenames:
        return
    name = metafunc.function.__name__
    if "tpch" in name:
        num_queries = PDSHQueries.num_queries
    elif "tpcds" in name:
        num_queries = PDSDSPolarsQueries.num_queries
    else:
        return
    metafunc.parametrize(
        "q_id",
        range(1, num_queries + 1),
        ids=[f"q{i:02d}" for i in range(1, num_queries + 1)],
    )


@pytest.fixture
def pending_closes(monkeypatch: pytest.MonkeyPatch) -> dict[Any, Any]:
    """Isolate the registry of cancelled collective closes."""
    pending: dict[Any, Any] = {}
    monkeypatch.setattr(closing, "_pending", pending)
    monkeypatch.setattr(closing, "_unusable", {})
    monkeypatch.setattr(closing, "_gate_locks", {})
    return pending


@pytest.fixture(scope="session")
def tpc_streaming_options(request: pytest.FixtureRequest) -> StreamingOptions:
    ref = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(ref)
    _add_dataset_args(ref)
    ns: dict[str, Any] = {}
    for action in ref._actions:
        if not action.option_strings or action.dest == "help":
            continue
        ns[action.dest] = request.config.getoption(action.dest)
    # Raise on fallback unless --no-raise-on-fail was given.
    if ns["raise_on_fail"] is None:
        ns["raise_on_fail"] = True
    return StreamingOptions._from_argparse(argparse.Namespace(**ns))


@pytest.fixture(scope="session")
def tpc_run_options(request: pytest.FixtureRequest) -> RunOptions:
    return RunOptions(
        validation_abs_tol=request.config.getoption("validation_abs_tol"),
    )


@pytest.fixture(scope="session")
def tpc_validation_method(tpc_run_options: RunOptions) -> ValidationMethod:
    return ValidationMethod(
        expected_source="duckdb",
        comparison_method="polars",
        comparison_options={
            **POLARS_VALIDATION_OPTIONS,
            "abs_tol": tpc_run_options.validation_abs_tol,
        },
        expected_location=None,
    )


@pytest.fixture(scope="session")
def tpc_spmd_engine(
    tpc_streaming_options: StreamingOptions,
) -> Generator[SPMDEngine, None, None]:
    with SPMDEngine(
        rapidsmpf_options=tpc_streaming_options.to_rapidsmpf_options(),
        executor_options=tpc_streaming_options.to_executor_options(),
        engine_options=tpc_streaming_options.to_engine_options(),
    ) as engine:
        yield engine
