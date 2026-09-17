# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from rapidsmpf.bootstrap import get_nranks, is_running_with_rrun

from cudf_polars.testing._engine_pool import EnginePool
from cudf_polars.testing.engine_utils import (
    ALL_ENGINE_FIXTURE_PARAMS,
    STREAMING_ENGINE_FIXTURE_PARAMS,
    EngineFixtureParam,
    configure_streaming_engine,
    create_streaming_options,
    merge_streaming_options,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_140, POLARS_VERSION_LT_141

_TEST_CALL_FAILED = pytest.StashKey[bool]()
_ENGINE_POOL_TIMINGS = pytest.StashKey[list[str]]()
_ENGINE_POOL_WORKER_TIMINGS = pytest.StashKey[dict[str, list[str]]]()
_PHASE_TIMINGS = pytest.StashKey[dict[str, dict[str, float | int]]]()
_WORKER_PHASE_TIMINGS = pytest.StashKey[dict[str, dict[str, dict[str, float | int]]]]()


@pytest.fixture
def xfail_decimal_sum_precision_polars_140(request: pytest.FixtureRequest) -> None:
    """xfail decimal ``sum`` tests on polars 1.40."""
    request.applymarker(
        pytest.mark.xfail(
            condition=(not POLARS_VERSION_LT_140) and POLARS_VERSION_LT_141,
            reason="polars 1.40 reports narrow precision for decimal sum "
            "(Decimal(9,2) vs (38,2)), fixed in 1.41. "
            "See https://github.com/pola-rs/polars/issues/27269",
        )
    )


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from cudf_polars.engine.core import StreamingEngine
    from cudf_polars.engine.dask import DaskEngine
    from cudf_polars.engine.options import StreamingOptions
    from cudf_polars.engine.ray import RayEngine
    from cudf_polars.engine.spmd import SPMDEngine


@pytest.fixture(scope="session")
def ray_num_ranks() -> int:
    """
    Number of ranks for multi-rank streaming engines that share one GPU
    (currently ``RayEngine``). Single-GPU dev hosts and CI runners require
    ``allow_gpu_sharing=True`` to oversubscribe one device across actors.
    """
    return 2


@pytest.fixture(scope="session")
def ray_init_options(ray_num_ranks: int) -> dict[str, Any]:
    """
    Keyword arguments forwarded to ``ray.init`` for the test Ray cluster.

    When using this fixture, a ``RayEngine`` must be constructed with:
    - ``num_ranks`` set
    - ``engine_options={"allow_gpu_sharing": True}``

    This is required because the cluster is configured with ``num_gpus=0``,
    so Ray does not autodetect or track GPU resources.
    """
    return {
        "num_cpus": ray_num_ranks,
        "num_gpus": 0,
        "include_dashboard": False,
        "object_store_memory": 256 * 1024 * 1024,  # 256 MB
    }


@pytest.fixture(params=[False, True], ids=["no_nulls", "nulls"], scope="session")
def with_nulls(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(autouse=True)
def _skip_unless_spmd(request: pytest.FixtureRequest) -> None:
    """Skip tests in SPMD multi-rank mode unless marked with ``pytest.mark.spmd``."""
    if (
        is_running_with_rrun()
        and get_nranks() > 1
        and not request.node.get_closest_marker("spmd")
    ):
        pytest.skip("skip: SPMD nranks > 1 (mark with pytest.mark.spmd to run)")


@pytest.fixture(scope="session")
def _engine_param(request: pytest.FixtureRequest) -> EngineFixtureParam:
    """Decoded engine variant selected by pytest parametrization.

    :func:`pytest_generate_tests` inspects all tests and then filters those
    with ``_engine_param`` in their fixture list according to the public
    engine fixture being used. For example, if a given test requests the
    :func:`spmd_engine` fixture then its underlying ``_engine_param`` is
    rebound to only loop over spmd engines for that test.
    """
    return EngineFixtureParam(full_name=request.param)


@pytest.fixture(scope="session")
def _engine_pool(
    request: pytest.FixtureRequest,
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
) -> Generator[EnginePool, None, None]:
    """Own the standard distributed engines for this pytest worker."""
    pool = EnginePool(
        ray_num_ranks=ray_num_ranks,
        ray_init_options=ray_init_options,
    )
    try:
        yield pool
    finally:
        pool.close()
        if request.config.getoption("--engine-pool-timings"):
            request.config.stash[_ENGINE_POOL_TIMINGS] = pool.timing_lines()


@pytest.fixture
def _unconfigured_engine(
    request: pytest.FixtureRequest,
    _engine_param: EngineFixtureParam,
    _engine_pool: EnginePool,
) -> Generator[tuple[pl.GPUEngine, StreamingOptions | None], None, None]:
    """
    Fixture generating an engine resource and options to apply before use.

    Parameters
    ----------
    _engine_param
        The parameterisation of the engine

    Returns
    -------
    tuple
        Of an engine and options to apply to the engine to configure it (or
        None if no configuration is needed).

    Notes
    -----
    Ray and Dask engines are leased from a pool local to this pytest worker.
    A successful test resets and health-checks its engine before returning it
    to the pool; any failed or unhealthy engine is shut down instead.
    """
    if _engine_param.engine_name == "in-memory":
        yield pl.GPUEngine(executor="in-memory", raise_on_fail=True), None
    else:
        engine: StreamingEngine
        if _engine_param.engine_name in ("dask", "ray") and is_running_with_rrun():
            pytest.skip(
                f"{_engine_param.engine_name} engine cannot be constructed "
                "inside an rrun cluster"
            )
        options = create_streaming_options(_engine_param.blocksize_mode)
        if _engine_param.engine_name in ("dask", "ray"):
            engine = _engine_pool.acquire(_engine_param.engine_name)
            try:
                yield engine, options
            finally:
                _engine_pool.release(
                    _engine_param.engine_name,
                    engine,
                    options,
                    test_failed=request.node.stash.get(_TEST_CALL_FAILED, False),
                    nodeid=request.node.nodeid,
                )
            return
        match _engine_param.engine_name:
            case "spmd":
                from cudf_polars.engine.spmd import SPMDEngine

                engine = SPMDEngine()
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown streaming engine: {_engine_param.engine_name!r}"
                )
        with engine:
            yield engine, options


@pytest.fixture
def spmd_engine(
    _unconfigured_engine: tuple[SPMDEngine, StreamingOptions],
) -> SPMDEngine:
    """Return the shared configured :class:`SPMDEngine`."""
    engine, options = _unconfigured_engine
    return configure_streaming_engine(engine, options)


@pytest.fixture
def spmd_engine_factory(
    _unconfigured_engine: tuple[SPMDEngine, StreamingOptions],
) -> Callable[[StreamingOptions], SPMDEngine]:
    """
    Return a function that, when called, produces a :class:`SPMDEngine`.

    Parameters
    ----------
    _unconfigured_engine
        Session-scoped engine selected by pytest parametrization.

    Returns
    -------
    Factory function that returns the shared :class:`SPMDEngine` when
    provided with :class:`StreamingOptions`.

    Notes
    -----
    Use this in place of :func:`streaming_engine_factory` for tests that
    must run on SPMD only.
    """
    engine, base = _unconfigured_engine

    def factory(options: StreamingOptions) -> SPMDEngine:
        return configure_streaming_engine(
            engine,
            merge_streaming_options(base, options),
        )

    return factory


@pytest.fixture
def ray_engine(
    _unconfigured_engine: tuple[RayEngine, StreamingOptions],
) -> RayEngine:
    """Return the shared configured :class:`RayEngine`."""
    engine, options = _unconfigured_engine
    return configure_streaming_engine(engine, options)


@pytest.fixture
def dask_engine(
    _unconfigured_engine: tuple[DaskEngine, StreamingOptions],
) -> DaskEngine:
    """Return the shared configured :class:`DaskEngine`."""
    engine, options = _unconfigured_engine
    return configure_streaming_engine(engine, options)


@pytest.fixture
def streaming_engine_factory(
    _unconfigured_engine: tuple[StreamingEngine, StreamingOptions],
) -> Callable[[StreamingOptions], StreamingEngine]:
    """
    Return a function that, when called, produces a :class:`StreamingEngine`.

    Parameters
    ----------
    _unconfigured_engine
        Session-scoped engine selected by pytest parametrization.

    Returns
    -------
    Factory function that returns the shared :class:`StreamingEngine` when
    provided with :class:`StreamingOptions`.
    """
    engine, base = _unconfigured_engine

    def factory(options: StreamingOptions) -> StreamingEngine:
        return configure_streaming_engine(
            engine,
            merge_streaming_options(base, options),
        )

    return factory


@pytest.fixture
def streaming_engine(
    _unconfigured_engine: tuple[StreamingEngine, StreamingOptions],
) -> StreamingEngine:
    """
    Return the shared configured :class:`StreamingEngine`.

    Inherits the parametrization of ``_unconfigured_engine``, so
    tests using this fixture run once per ``(backend, blocksize_mode)``
    combination.

    Parameters
    ----------
    _unconfigured_engine
        Session-scoped engine selected by pytest parametrization.

    Returns
    -------
    A streaming engine configured with the parametrized baseline.
    """
    engine, options = _unconfigured_engine
    return configure_streaming_engine(engine, options)


@pytest.fixture
def engine(
    _unconfigured_engine: tuple[pl.GPUEngine, StreamingOptions | None],
) -> pl.GPUEngine:
    """
    Return a :class:`polars.GPUEngine` for each engine variant under test.

    Every variant is configured with ``raise_on_fail=True``, so an unsupported
    GPU path raises instead of silently falling back to the CPU engine.

    Parameters
    ----------
    _unconfigured_engine
        Session-scoped engine selected by pytest parametrization.

    Returns
    -------
    Engine instance matching the parametrized variant.

    Notes
    -----
    For tests that require a :class:`StreamingEngine` only, use the
    :func:`streaming_engine` fixture instead.
    """
    engine, options = _unconfigured_engine
    if options is None:
        return engine
    from cudf_polars.engine.core import StreamingEngine

    assert isinstance(engine, StreamingEngine)
    return configure_streaming_engine(engine, options)


@pytest.fixture
def in_memory_engine() -> pl.GPUEngine:
    """
    Return an in-memory :class:`polars.GPUEngine` with ``raise_on_fail=True``.

    Use this for tests that intentionally exercise in-memory-only behavior
    instead of the parametrized engine matrix, but generally prefer the ``engine``
    fixture instead.
    """
    return pl.GPUEngine(executor="in-memory", raise_on_fail=True)


@pytest.fixture
def timeout_seconds() -> int:
    """
    Conservative timeout for APIs that accept a timeout parameter.
    """
    return 30


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--engine-pool-timings",
        action="store_true",
        help="report cudf-polars Ray/Dask test-engine pool lifecycle timings",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    """Make real setup/call failures available to fixture teardown.

    A fixture can skip while another fixture that leased an engine is being
    set up. That is not an engine failure and the lease is still safe to
    reuse; only an actual failed setup or test call must poison it.
    """
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("--engine-pool-timings"):
        _record_phase_timing(item, report)
    if report.when in ("setup", "call") and report.failed:
        item.stash[_TEST_CALL_FAILED] = True


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Report worker-local distributed-engine lifecycle timing on request."""
    timings = config.stash.get(_ENGINE_POOL_TIMINGS, [])
    if timings:
        terminalreporter.write_line("cudf-polars engine pool:")
        for line in timings:
            terminalreporter.write_line(line)
    for worker, worker_timings in config.stash.get(
        _ENGINE_POOL_WORKER_TIMINGS, {}
    ).items():
        terminalreporter.write_line(f"cudf-polars engine pool ({worker}):")
        for line in worker_timings:
            terminalreporter.write_line(line)
    for worker, phase_timings in config.stash.get(_WORKER_PHASE_TIMINGS, {}).items():
        terminalreporter.write_line(f"cudf-polars pytest phase timings ({worker}):")
        for engine_name, phases in phase_timings.items():
            terminalreporter.write_line(
                f"  {engine_name}: "
                f"setup={phases['setup']:.2f}s ({phases['setup_count']}), "
                f"call={phases['call']:.2f}s ({phases['call_count']}), "
                f"teardown={phases['teardown']:.2f}s ({phases['teardown_count']})"
            )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Transfer opt-in timings from each xdist worker to its controller."""
    worker_output = getattr(session.config, "workeroutput", None)
    timings = session.config.stash.get(_ENGINE_POOL_TIMINGS, [])
    if worker_output is not None and timings:
        worker_output["cudf_polars_engine_pool_timings"] = timings
    phase_timings = session.config.stash.get(_PHASE_TIMINGS, {})
    if worker_output is not None and phase_timings:
        worker_output["cudf_polars_phase_timings"] = phase_timings


def pytest_testnodedown(node: Any, error: Exception | None) -> None:
    """Collect an xdist worker's engine-pool timings for terminal reporting."""
    # xdist invokes this hook for a worker that exits before it can emit its
    # final ``workerfinished`` event as well; that controller has no
    # ``workeroutput`` attribute to collect.
    worker_output = getattr(node, "workeroutput", {})
    timings = worker_output.get("cudf_polars_engine_pool_timings")
    if timings:
        all_timings = node.config.stash.get(_ENGINE_POOL_WORKER_TIMINGS, {})
        all_timings[node.gateway.id] = timings
        node.config.stash[_ENGINE_POOL_WORKER_TIMINGS] = all_timings
    phase_timings = worker_output.get("cudf_polars_phase_timings")
    if phase_timings:
        all_phase_timings = node.config.stash.get(_WORKER_PHASE_TIMINGS, {})
        all_phase_timings[node.gateway.id] = phase_timings
        node.config.stash[_WORKER_PHASE_TIMINGS] = all_phase_timings


def _record_phase_timing(item: pytest.Item, report: pytest.TestReport) -> None:
    """Accumulate xdist-worker test-phase time by parametrized engine."""
    callspec = getattr(item, "callspec", None)
    engine_param = (
        callspec.params.get("_engine_param", "other")
        if callspec is not None
        else "other"
    )
    engine_name = getattr(engine_param, "engine_name", str(engine_param))
    timings = item.config.stash.get(_PHASE_TIMINGS, {})
    for name in ("all", engine_name):
        phases = timings.setdefault(
            name,
            {
                "setup": 0.0,
                "setup_count": 0,
                "call": 0.0,
                "call_count": 0,
                "teardown": 0.0,
                "teardown_count": 0,
            },
        )
        phases[report.when] += report.duration
        phases[f"{report.when}_count"] += 1
    item.config.stash[_PHASE_TIMINGS] = timings


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "skip_on_streaming_engine(reason, *, engine=None): skip the test for "
        'streaming ``engine`` variants (e.g. ``"spmd"``, ``"spmd-small"``, '
        '``"dask"``, ``"ray"``) while still allowing the in-memory variant to run.',
    )
    config.addinivalue_line(
        "markers",
        "engine_params(params): run an engine-fixture test only for the "
        "listed engine parameter ids.",
    )

    # Ray's internal subprocess management leaks `/dev/null` file handles, and
    # distributed's shutdown leaves unclosed sockets. Under Python 3.14 +
    # pytest 9, these surface as unraisable `ResourceWarning`s and, combined
    # with `filterwarnings = ["error", ...]` in pyproject.toml, fail
    # otherwise-unrelated tests when the GC finalizer happens to fire during
    # them. With `pytest-xdist --dist=worksteal`, the leak can land in any
    # test that shares a worker with a ray/dask test, so the suppression must
    # apply globally rather than per-module.
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")
    # https://github.com/open-telemetry/opentelemetry-python/issues/5231 (used by Ray)
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")


def pytest_generate_tests(metafunc: pytest.Metafunc):
    """Parametrize the shared engine fixture without cartesian products."""
    fixtures = set(metafunc.fixturenames)
    if "_engine_param" not in fixtures:
        return

    if "spmd_engine" in fixtures or "spmd_engine_factory" in fixtures:
        engines = ["spmd"]
    elif "ray_engine" in fixtures:
        engines = ["ray"]
    elif "dask_engine" in fixtures:
        engines = ["dask"]
    elif "streaming_engine" in fixtures or "streaming_engine_factory" in fixtures:
        engines = STREAMING_ENGINE_FIXTURE_PARAMS
    elif "engine" in fixtures:
        engines = ALL_ENGINE_FIXTURE_PARAMS
    else:
        raise AssertionError("Unknown engine fixture")

    marker = metafunc.definition.get_closest_marker("engine_params")
    if marker is not None:
        engines = list(marker.args[0])

    metafunc.parametrize(
        "_engine_param",
        engines,
        indirect=True,
        ids=engines,
        scope="session",
    )


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
):
    """Apply ``skip_on_streaming_engine`` markers to streaming engine items."""
    for item in items:
        marker = item.get_closest_marker("skip_on_streaming_engine")
        if marker is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        engine_param = callspec.params.get("_engine_param")
        if engine_param is None or engine_param == "in-memory":
            continue
        engine_filter = marker.kwargs.get("engine")
        if engine_filter is not None:
            if isinstance(engine_filter, str):
                engine_filter = (engine_filter,)
            # Strip the ``-small`` suffix so ``"spmd-small"`` matches
            # ``engine=("spmd",)``.
            engine_name = engine_param.removesuffix("-small")
            if engine_name not in engine_filter:
                continue
        reason = (
            marker.args[0]
            if marker.args
            else marker.kwargs.get("reason", "unsupported on streaming engine")
        )
        item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="module")
def parquet_stats_executor() -> concurrent.futures.ThreadPoolExecutor:  # type: ignore[misc]
    """A thread pool to use for cudf-polars status collection."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        yield executor
