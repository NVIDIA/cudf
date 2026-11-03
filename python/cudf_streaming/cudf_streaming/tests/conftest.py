# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import rmm.mr
from rapidsmpf.communicator import COMMUNICATORS
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.streaming.core.context import Context
from rmm.pylibrmm.stream import DEFAULT_STREAM

if TYPE_CHECKING:
    from collections.abc import Generator

    from rapidsmpf.communicator.communicator import Communicator
    from rmm.pylibrmm.stream import Stream


@pytest.fixture(scope="session")
def _mpi_comm() -> Communicator:
    """Session-wide MPI communicator."""
    if "mpi" not in COMMUNICATORS:
        pytest.skip("RapidsMPF not built with MPI support")

    from mpi4py import MPI

    from rapidsmpf.communicator.mpi import new_communicator

    return new_communicator(
        MPI.COMM_WORLD, Options(get_environment_variables()), ProgressThread()
    )


@pytest.fixture(scope="session")
def _ucxx_comm() -> Communicator:
    """Session-wide UCXX communicator."""
    if "ucxx" not in COMMUNICATORS:
        pytest.skip("RapidsMPF not built with UCXX support")
    if "mpi" not in COMMUNICATORS:
        pytest.skip("MPI required to bootstrap UCXX test communicator")

    from rapidsmpf.communicator.testing import ucxx_mpi_setup

    return ucxx_mpi_setup(
        None, Options(get_environment_variables()), ProgressThread()
    )


@pytest.fixture(params=["mpi", "ucxx"])
def comm(
    request: pytest.FixtureRequest,
) -> Generator[Communicator, None, None]:
    """Communicator fixture parametrized over MPI and UCXX transports."""
    comm_name = request.param

    if "mpi" not in COMMUNICATORS:
        pytest.skip("RapidsMPF not built with MPI support")
    if "ucxx" not in COMMUNICATORS:
        pytest.skip("RapidsMPF not built with UCXX support")

    from mpi4py import MPI

    MPI.COMM_WORLD.barrier()
    yield request.getfixturevalue(f"_{comm_name}_comm")
    MPI.COMM_WORLD.barrier()


@pytest.fixture
def device_mr() -> Generator[rmm.mr.CudaMemoryResource, None, None]:
    """
    Fixture for creating a new cuda memory resource and making it the
    current rmm resource temporarily.
    """
    prior_mr = rmm.mr.get_current_device_resource()
    try:
        mr = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(mr)
        yield mr
    finally:
        rmm.mr.set_current_device_resource(prior_mr)


@pytest.fixture
def stream() -> Stream:
    """CUDA stream for test operations."""
    return DEFAULT_STREAM


@pytest.fixture
def context(comm: Communicator) -> Generator[Context, None, None]:
    """Streaming context backed by a fresh memory resource."""
    options = Options(get_environment_variables())
    br = BufferResource(rmm.mr.CudaMemoryResource())

    with Context(comm.logger, br, options) as ctx:
        yield ctx
