# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Ray and Dask engines create the cache in their worker processes."""

import pytest
from cudf_polars.utils import cucascade

CACHE_BYTES = 64 << 20


@pytest.fixture
def aws_env(monkeypatch):
    # The cache is created with these but makes no request until a read.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def _ray_engine(nranks, executor_options):
    pytest.importorskip("ray")
    from cudf_polars.engine.ray import RayEngine

    return RayEngine(
        num_ranks=nranks,
        executor_options=executor_options,
        engine_options={"allow_gpu_sharing": True},
        ray_init_options={"include_dashboard": False},
    )


def _dask_engine(nranks, executor_options):
    distributed = pytest.importorskip("distributed")
    from cudf_polars.engine.dask import DaskEngine

    cluster = distributed.LocalCluster(
        n_workers=nranks, threads_per_worker=1, dashboard_address=None
    )
    client = distributed.Client(cluster)
    try:
        return DaskEngine(
            dask_client=client,
            executor_options=executor_options,
            engine_options={"allow_gpu_sharing": True},
        ), (client, cluster)
    except BaseException:
        client.close()
        cluster.close()
        raise


@pytest.mark.usefixtures("aws_env")
def test_ray_single_rank_creates_the_cache_on_its_rank():
    engine = _ray_engine(1, {"cucascade_cache_bytes": CACHE_BYTES})
    try:
        assert engine._run(cucascade.enabled) == [True]
        # The client process reads nothing, so it has no cache of its own.
        assert not cucascade.enabled()
    finally:
        engine.shutdown()


@pytest.mark.usefixtures("aws_env")
def test_dask_single_rank_creates_the_cache_on_its_rank():
    engine, (client, cluster) = _dask_engine(
        1, {"cucascade_cache_bytes": CACHE_BYTES}
    )
    try:
        assert engine._run(cucascade.enabled) == [True]
        # The client process reads nothing, so it has no cache of its own.
        assert not cucascade.enabled()
    finally:
        engine.shutdown()
        client.close()
        cluster.close()


def test_ray_refuses_the_cache_with_more_than_one_rank():
    with pytest.raises(ValueError, match="single rank"):
        _ray_engine(2, {"cucascade_cache_bytes": CACHE_BYTES})


def test_dask_refuses_the_cache_with_more_than_one_rank():
    distributed = pytest.importorskip("distributed")
    from cudf_polars.engine.dask import DaskEngine

    with (
        distributed.LocalCluster(
            n_workers=2, threads_per_worker=1, dashboard_address=None
        ) as cluster,
        distributed.Client(cluster) as client,
        pytest.raises(ValueError, match="single rank"),
    ):
        DaskEngine(
            dask_client=client,
            executor_options={"cucascade_cache_bytes": CACHE_BYTES},
            engine_options={"allow_gpu_sharing": True},
        )
