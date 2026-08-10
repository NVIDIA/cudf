# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from rapidsmpf.bootstrap import get_nranks, is_running_with_rrun

from cudf_polars.testing.engine_utils import (
    ALL_ENGINE_FIXTURE_PARAMS,
    STREAMING_ENGINE_FIXTURE_PARAMS,
    EngineFixtureParam,
    configure_streaming_engine,
    create_streaming_options,
    merge_streaming_options,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_140, POLARS_VERSION_LT_141


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
def _unconfigured_engine(
    _engine_param: EngineFixtureParam,
    ray_num_ranks: int,
    ray_init_options: dict[str, Any],
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
    This session-scoped fixture keeps the heavy state of an engine alive
    for the lifetime of its use, shutting it down once the particular
    engine is not required any more. Tests should not use this fixture
    directly, but rather one of the parameterised "public" engine fixtures.
    Those take care of applying the configuration to the base engine each
    time it is used in a test.
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
        match _engine_param.engine_name:
            case "spmd":
                from cudf_polars.engine.spmd import SPMDEngine

                engine = SPMDEngine()
            case "dask":  # pragma: no cover
                from cudf_polars.engine.dask import DaskEngine

                engine = DaskEngine(engine_options={"allow_gpu_sharing": True})
            case "ray":  # pragma: no cover
                from cudf_polars.engine.ray import RayEngine

                # Always specify num_ranks so the engine has a fixed size
                # regardless of how many GPUs the host happens to have;
                # otherwise ``RayEngine`` defaults to
                # ``get_num_gpus_in_ray_cluster()``
                engine = RayEngine(
                    num_ranks=ray_num_ranks,
                    engine_options={"allow_gpu_sharing": True},
                    ray_init_options=ray_init_options,
                )
            case _:  # pragma: no cover
                raise ValueError(
                    f"Unknown streaming engine: {_engine_param.engine_name!r}"
                )
        with engine:
            yield engine, create_streaming_options(_engine_param.blocksize_mode)


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
def engine_raise_on_fail() -> pl.GPUEngine:
    """
    Return a default :class:`polars.GPUEngine` with ``raise_on_fail=True``.

    Returns
    -------
    In-memory engine configured to raise exceptions on failure.

    Notes
    -----
    Intended for error-path tests that assert specific exceptions propagate
    from ``.collect()``. Uses the in-memory executor so errors are not wrapped
    by a streaming task group.
    """
    # TODO: We should be testing with all supported engine variants
    return pl.GPUEngine(executor="in-memory", raise_on_fail=True)


@pytest.fixture
def timeout_seconds() -> int:
    """
    Conservative timeout for APIs that accept a timeout parameter.
    """
    return 30


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "skip_on_streaming_engine(reason, *, engine=None): skip the test for "
        'streaming ``engine`` variants (e.g. ``"spmd"``, ``"spmd-small"``, '
        '``"dask"``, ``"ray"``) while still allowing the in-memory variant to run.',
    )

    # Ray's internal subprocess management leaks `/dev/null` file handles, and
    # distributed's shutdown leaves unclosed sockets. Under Python 3.14 +
    # pytest 9, these surface as unraisable `ResourceWarning`s and, combined
    # with `filterwarnings = ["error", ...]` in pyproject.toml, fail
    # otherwise-unrelated tests when the GC finalizer happens to fire during
    # them. With pytest-xdist, the leak can land in any test that shares a
    # worker with a ray/dask test, so the suppression must apply globally
    # rather than per-module.
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

    # Group by full engine param id so each variant (including ``spmd`` vs
    # ``spmd-small``) stays on one xdist worker when using ``--dist=loadgroup``.
    params = [
        pytest.param(
            eng,
            marks=pytest.mark.xdist_group(eng),
        )
        for eng in engines
    ]
    metafunc.parametrize(
        "_engine_param",
        params,
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
