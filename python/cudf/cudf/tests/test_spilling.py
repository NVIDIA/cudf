# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import importlib
import warnings
import weakref

import cupy
import numpy as np
import pandas
import pandas.testing
import pytest

import rmm

import cudf
import cudf.core.buffer.spill_manager
import cudf.options
from cudf.core.abc import Serializable
from cudf.core.buffer import Buffer, as_buffer
from cudf.core.buffer.spill_manager import (
    SpillManager,
    get_global_manager,
    get_rmm_memory_resource_stack,
    set_global_manager,
    spill_on_demand_globally,
)
from cudf.core.buffer.spillable_buffer import (
    SpillableBuffer,
    SpillableBufferOwner,
)
from cudf.testing import assert_eq

if get_global_manager() is not None:
    pytest.skip(
        "cannot test spilling when enabled globally, set `CUDF_SPILL=off`",
        allow_module_level=True,
    )


@contextlib.contextmanager
def set_rmm_memory_pool(nbytes: int):
    mr = rmm.mr.get_current_device_resource()
    rmm.mr.set_current_device_resource(
        rmm.mr.PoolMemoryResource(
            mr,
            initial_pool_size=nbytes,
            maximum_pool_size=nbytes,
        )
    )
    try:
        yield
    finally:
        rmm.mr.set_current_device_resource(mr)


def single_column_df(target="gpu") -> cudf.DataFrame:
    """Create a standard single column dataframe used for testing

    Use `single_column_df_data`, `single_column_df_base_data`,
    `gen_df_data_nbytes` for easy access to the buffer of the column.

    Notice, this is just for convenience, there is nothing special
    about this dataframe.

    Parameters
    ----------
    target : str, optional
        Set the spill state of the dataframe

    Return
    ------
    DataFrame
        A standard dataframe with a single column
    """
    ret = cudf.DataFrame({"a": [1, 2, 3]})
    if target != "gpu":
        single_column_df_data(ret).owner.spill(target=target)
    return ret


def single_column_df_data(df: cudf.DataFrame) -> SpillableBuffer:
    """Access `.data` of the column of a standard dataframe"""
    ret = df._data._data["a"].data
    assert isinstance(ret, SpillableBuffer)
    return ret


# Get number of bytes of the column of a standard dataframe
gen_df_data_nbytes = single_column_df()._data._data["a"].data.size


def spilled_and_unspilled(manager: SpillManager) -> tuple[int, int]:
    """Get bytes spilled and unspilled known by the manager"""
    spilled = sum(buf.size for buf in manager.buffers() if buf.is_spilled)
    unspilled = sum(
        buf.size for buf in manager.buffers() if not buf.is_spilled
    )
    return spilled, unspilled


@pytest.fixture
def manager(request):
    """Fixture to enable and make a spilling manager availabe"""
    kwargs = dict(getattr(request, "param", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        set_global_manager(manager=SpillManager(**kwargs))
        yield get_global_manager()
        # Retrieving the test result using the `pytest_runtest_makereport`
        # hook from conftest.py
        if request.node.report["call"].failed:
            # Ignore `overwriting non-empty manager` errors when
            # test is failing.
            warnings.simplefilter("ignore")
        set_global_manager(manager=None)


def test_spillable_buffer(manager: SpillManager):
    buf = as_buffer(data=rmm.DeviceBuffer(size=10))
    assert isinstance(buf, SpillableBuffer)
    assert buf.owner.spillable
    buf.owner.mark_exposed()
    assert buf.owner.exposed
    assert not buf.owner.spillable
    buf = as_buffer(data=rmm.DeviceBuffer(size=10))
    # Notice, accessing `__cuda_array_interface__` itself doesn't
    # expose the pointer, only accessing the "data" field exposes
    # the pointer.
    iface = buf.__cuda_array_interface__
    assert not buf.owner.exposed
    assert buf.owner.spillable
    iface["data"][0]  # Expose pointer
    assert buf.owner.exposed
    assert not buf.owner.spillable


@pytest.mark.parametrize(
    "attribute",
    [
        "ptr",
        "memoryview",
        "memory_info",
    ],
)
def test_spillable_buffer_view_attributes(manager: SpillManager, attribute):
    base = as_buffer(data=rmm.DeviceBuffer(size=10))
    view = base[:]
    attr_base = getattr(base, attribute)
    attr_view = getattr(view, attribute)
    if callable(attr_view):
        pass
    else:
        assert attr_base == attr_view


@pytest.mark.parametrize("target", ["gpu", "cpu"])
def test_memory_info(manager: SpillManager, target):
    if target == "gpu":
        mem = rmm.DeviceBuffer(size=10)
        ptr = mem.ptr
        size = mem.size
    elif target == "cpu":
        data = np.empty(10, dtype="u1")
        mem = memoryview(data)
        ptr = data.__array_interface__["data"][0]
        size = data.size
    b = as_buffer(data=mem)
    assert b.memory_info() == (ptr, size, target)
    assert b[:].memory_info() == (ptr, size, target)
    assert b[:-1].memory_info() == (ptr, size - 1, target)
    assert b[1:].memory_info() == (ptr + 1, size - 1, target)
    assert b[2:4].memory_info() == (ptr + 2, 2, target)


def test_from_pandas(manager: SpillManager):
    pdf1 = pandas.DataFrame({"a": [1, 2, 3]})
    df = cudf.from_pandas(pdf1)
    assert single_column_df_data(df).owner.spillable
    pdf2 = df.to_pandas()
    pandas.testing.assert_frame_equal(pdf1, pdf2)


def test_creations(manager: SpillManager):
    df = single_column_df()
    assert single_column_df_data(df).owner.spillable

    df = cudf.datasets.timeseries(dtypes={"a": float})
    assert single_column_df_data(df).owner.spillable

    df = cudf.datasets.randomdata(dtypes={"a": float})
    assert single_column_df_data(df).owner.spillable


def test_spillable_df_groupby(manager: SpillManager):
    """Test that GroupBy context manager enters/exits access contexts properly."""
    df = cudf.DataFrame({"a": [1, 1, 1]})
    gb = df.groupby("a")

    # Before using context manager, no spill locks
    assert len(single_column_df_data(df).owner._spill_locks) == 0

    with gb._groupby:
        assert len(single_column_df_data(df).owner._spill_locks) == 1
        assert not single_column_df_data(df).owner.spillable

    assert len(single_column_df_data(df).owner._spill_locks) == 0
    assert single_column_df_data(df).owner.spillable

    # Operations should work correctly
    result = gb.sum()  # noqa: F841

    # After operation completes, no persistent locks
    assert len(single_column_df_data(df).owner._spill_locks) == 0


def test_spilling_buffer(manager: SpillManager):
    buf = as_buffer(rmm.DeviceBuffer(size=10))
    buf.owner.spill(target="cpu")
    assert buf.owner.is_spilled
    buf.owner.mark_exposed()  # Expose pointer and trigger unspill
    assert not buf.owner.is_spilled
    with pytest.raises(ValueError, match="unspillable buffer"):
        buf.owner.spill(target="cpu")


def _reload_options():
    # In order to enabling monkey patching of the environment variables
    # mark the global manager as uninitialized.
    set_global_manager(None)
    cudf.core.buffer.spill_manager._global_manager_uninitialized = True
    importlib.reload(cudf.options)


@contextlib.contextmanager
def _get_manager_in_env(monkeypatch, var_vals):
    with monkeypatch.context() as m:
        for var, val in var_vals:
            m.setenv(var, val)
        _reload_options()
        yield get_global_manager()
    _reload_options()


def test_environment_variables_spill_off(monkeypatch):
    with _get_manager_in_env(
        monkeypatch,
        [("CUDF_SPILL", "off")],
    ) as manager:
        assert manager is None


def test_environment_variables_spill_on(monkeypatch):
    with _get_manager_in_env(
        monkeypatch,
        [("CUDF_SPILL", "on"), ("CUDF_SPILL_ON_DEMAND", "off")],
    ) as manager:
        assert isinstance(manager, SpillManager)
        assert manager._device_memory_limit is None
        assert manager.statistics.level == 0


def test_environment_variables_device_limit(monkeypatch):
    with _get_manager_in_env(
        monkeypatch,
        [
            ("CUDF_SPILL", "on"),
            ("CUDF_SPILL_ON_DEMAND", "off"),
            ("CUDF_SPILL_DEVICE_LIMIT", "1000"),
        ],
    ) as manager:
        assert isinstance(manager, SpillManager)
        assert manager._device_memory_limit == 1000
        assert manager.statistics.level == 0


@pytest.mark.parametrize("level", (1, 2))
def test_environment_variables_spill_stats(monkeypatch, level):
    with _get_manager_in_env(
        monkeypatch,
        [
            ("CUDF_SPILL", "on"),
            ("CUDF_SPILL_ON_DEMAND", "off"),
            ("CUDF_SPILL_DEVICE_LIMIT", "1000"),
            ("CUDF_SPILL_STATS", f"{level}"),
        ],
    ) as manager:
        assert isinstance(manager, SpillManager)
        assert manager._device_memory_limit == 1000
        assert manager.statistics.level == level


def test_spill_device_memory(manager: SpillManager):
    df = single_column_df()
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes)
    manager.spill_device_memory(nbytes=1)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes, 0)
    del df
    assert spilled_and_unspilled(manager) == (0, 0)
    df1 = single_column_df()
    df2 = single_column_df()
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).owner.is_spilled
    assert not single_column_df_data(df2).owner.is_spilled
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).owner.is_spilled
    assert single_column_df_data(df2).owner.is_spilled
    df3 = df1 + df2
    assert not single_column_df_data(df1).owner.is_spilled
    assert not single_column_df_data(df2).owner.is_spilled
    assert not single_column_df_data(df3).owner.is_spilled
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).owner.is_spilled
    assert not single_column_df_data(df2).owner.is_spilled
    assert not single_column_df_data(df3).owner.is_spilled
    df2.abs()  # Should change the access time
    manager.spill_device_memory(nbytes=1)
    assert single_column_df_data(df1).owner.is_spilled
    assert not single_column_df_data(df2).owner.is_spilled
    assert single_column_df_data(df3).owner.is_spilled


def test_spill_to_device_limit(manager: SpillManager):
    df1 = single_column_df()
    df2 = single_column_df()
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes * 2)
    manager.spill_to_device_limit(device_limit=0)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)
    df3 = df1 + df2
    manager.spill_to_device_limit(device_limit=0)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 3, 0)
    assert single_column_df_data(df1).owner.is_spilled
    assert single_column_df_data(df2).owner.is_spilled
    assert single_column_df_data(df3).owner.is_spilled


@pytest.mark.parametrize(
    "manager", [{"device_memory_limit": 0}], indirect=True
)
def test_zero_device_limit(manager: SpillManager):
    assert manager._device_memory_limit == 0
    df1 = single_column_df()
    df2 = single_column_df()
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)
    df1 + df2
    # Notice, while performing the addintion both df1 and df2 are unspillable
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes * 2)
    manager.spill_to_device_limit()
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)


def test_spill_df_index(manager: SpillManager):
    df = single_column_df()
    df.index = [1, 3, 2]  # use a materialized index
    assert spilled_and_unspilled(manager) == (0, gen_df_data_nbytes * 2)

    manager.spill_to_device_limit(gen_df_data_nbytes)
    assert spilled_and_unspilled(manager) == (
        gen_df_data_nbytes,
        gen_df_data_nbytes,
    )

    manager.spill_to_device_limit(0)
    assert spilled_and_unspilled(manager) == (gen_df_data_nbytes * 2, 0)


def test_external_memory(manager):
    cupy.cuda.set_allocator()  # uses default allocator
    cpy = cupy.asarray([1, 2, 3])
    s = cudf.Series(cpy)
    # Check that the cupy array is still alive after overwriting `cpy`
    cpy = weakref.ref(cpy)
    assert cpy() is not None
    # Check that the series is spillable and known by the spill manager
    assert len(manager.buffers()) == 1
    assert s._data[None].data.owner.spillable


def test_spilling_df_views(manager):
    df = single_column_df(target="cpu")
    assert single_column_df_data(df).owner.is_spilled
    df_view = df.loc[1:]
    assert single_column_df_data(df_view).owner.spillable
    assert single_column_df_data(df).owner.spillable


@pytest.mark.parametrize("target", ["gpu", "cpu"])
def test_get_ptr(manager: SpillManager, target):
    if target == "gpu":
        mem = rmm.DeviceBuffer(size=10)
    elif target == "cpu":
        mem = np.empty(10, dtype="u1").data
    buf = as_buffer(data=mem)
    assert buf.owner.spillable
    assert len(buf.owner._spill_locks) == 0
    with buf.access(mode="read", scope="internal"):
        buf.ptr
        assert not buf.owner.spillable
        with buf.access(mode="read", scope="internal"):
            buf.ptr
            assert not buf.owner.spillable
        assert not buf.owner.spillable
    assert buf.owner.spillable


@pytest.mark.parametrize("target", ["gpu", "cpu"])
@pytest.mark.parametrize("view", [None, slice(0, 2), slice(1, 3)])
def test_serialize_device(manager, target, view):
    df1 = single_column_df()
    if view is not None:
        df1 = df1.iloc[view]
    single_column_df_data(df1).owner.spill(target=target)

    header, frames = df1.device_serialize()
    assert len(frames) == 1
    if target == "gpu":
        assert isinstance(frames[0], Buffer)
        assert not single_column_df_data(df1).owner.is_spilled
        assert not single_column_df_data(df1).owner.spillable
        frames[0] = cupy.array(frames[0], copy=True)
    else:
        assert isinstance(frames[0], memoryview)
        assert single_column_df_data(df1).owner.is_spilled
        assert single_column_df_data(df1).owner.spillable

    df2 = Serializable.device_deserialize(header, frames)
    assert_eq(df1, df2)


@pytest.mark.parametrize("target", ["gpu", "cpu"])
@pytest.mark.parametrize("view", [None, slice(0, 2), slice(1, 3)])
def test_serialize_host(manager, target, view):
    df1 = single_column_df()
    if view is not None:
        df1 = df1.iloc[view]
    single_column_df_data(df1).owner.spill(target=target)

    # Unspilled df becomes spilled after host serialization
    header, frames = df1.host_serialize()
    assert all(isinstance(f, memoryview) for f in frames)
    df2 = Serializable.host_deserialize(header, frames)
    assert single_column_df_data(df2).owner.is_spilled
    assert_eq(df1, df2)


def test_serialize_dask_dataframe(manager: SpillManager):
    protocol = pytest.importorskip("distributed.protocol")

    df1 = single_column_df(target="gpu")
    header, frames = protocol.serialize(
        df1, serializers=("dask",), on_error="raise"
    )
    buf = single_column_df_data(df1)
    assert len(frames) == 1
    assert isinstance(frames[0], memoryview)
    # Check that the memoryview and frames is the same memory
    assert (
        np.array(buf.memoryview()).__array_interface__["data"]
        == np.array(frames[0]).__array_interface__["data"]
    )

    df2 = protocol.deserialize(header, frames)
    assert single_column_df_data(df2).owner.is_spilled
    assert_eq(df1, df2)


def test_serialize_cuda_dataframe(manager: SpillManager):
    protocol = pytest.importorskip("distributed.protocol")

    df1 = single_column_df(target="gpu")
    header, frames = protocol.serialize(
        df1, serializers=("cuda",), on_error="raise"
    )
    buf: SpillableBuffer = single_column_df_data(df1)
    assert len(buf.owner._spill_locks) == 1
    assert len(frames) == 1
    assert isinstance(frames[0], Buffer)
    assert frames[0].ptr == buf.ptr

    frames[0] = cupy.array(frames[0], copy=True)
    df2 = protocol.deserialize(header, frames)
    assert_eq(df1, df2)


def test_get_rmm_memory_resource_stack():
    mr1 = rmm.mr.CudaMemoryResource()
    assert all(
        not isinstance(m, rmm.mr.FailureCallbackResourceAdaptor)
        for m in get_rmm_memory_resource_stack(mr1)
    )

    mr2 = rmm.mr.FailureCallbackResourceAdaptor(mr1, lambda x: False)
    assert get_rmm_memory_resource_stack(mr2)[0] is mr2
    assert get_rmm_memory_resource_stack(mr2)[1] is mr1

    mr3 = rmm.mr.FixedSizeMemoryResource(mr2)
    assert get_rmm_memory_resource_stack(mr3)[0] is mr3
    assert get_rmm_memory_resource_stack(mr3)[1] is mr2
    assert get_rmm_memory_resource_stack(mr3)[2] is mr1

    mr4 = rmm.mr.FailureCallbackResourceAdaptor(mr3, lambda x: False)
    assert get_rmm_memory_resource_stack(mr4)[0] is mr4
    assert get_rmm_memory_resource_stack(mr4)[1] is mr3
    assert get_rmm_memory_resource_stack(mr4)[2] is mr2
    assert get_rmm_memory_resource_stack(mr4)[3] is mr1


def test_df_transpose(manager: SpillManager):
    df1 = cudf.DataFrame({"a": [1, 2]})
    df2 = df1.transpose()
    # Buffers should not be exposed unless explicitly accessed
    assert not df1._data._data["a"].data.owner.exposed
    assert not df2._data._data[0].data.owner.exposed
    assert not df2._data._data[1].data.owner.exposed
    # Verify they are spillable
    assert df1._data._data["a"].data.owner.spillable
    assert df2._data._data[0].data.owner.spillable
    assert df2._data._data[1].data.owner.spillable


def test_memory_usage_of_sliced_nested_column(manager: SpillManager):
    # Computing memory usage of a sliced list column reads its offsets
    # from device memory, which must spill lock the underlying buffers
    df = cudf.Series([[1, 2], [3, 4], [5, 6]])[1:]
    assert df.memory_usage() == 44
    for child_index in range(2):
        buf = df._column.plc_column.child(child_index).data()
        assert not buf.owner.exposed
        assert buf.owner.spillable


def test_as_buffer_of_spillable_buffer(manager: SpillManager):
    data = cupy.arange(10, dtype="u1")
    b1 = as_buffer(data)
    assert isinstance(b1, SpillableBuffer)
    assert isinstance(b1.owner, SpillableBufferOwner)
    assert b1.owner.owner is data
    b2 = as_buffer(b1)
    assert b1 is b2


@pytest.mark.parametrize("dtype", ["uint8", "uint64"])
def test_memoryview_slice(manager: SpillManager, dtype):
    """Check .memoryview() of a sliced spillable buffer"""

    data = np.arange(10, dtype=dtype).data
    # memoryview of a sliced spillable buffer
    m1 = as_buffer(data=data)[1:-1].memoryview()
    # sliced memoryview of data as bytes
    m2 = memoryview(data).cast("B")[1:-1]
    assert m1 == m2


@pytest.mark.parametrize(
    "manager", [{"statistic_level": 0}, {"statistic_level": 1}], indirect=True
)
def test_statistics(manager: SpillManager):
    assert len(manager.statistics.spill_totals) == 0

    buf: SpillableBuffer = as_buffer(data=rmm.DeviceBuffer(size=10))
    buf.owner.spill(target="cpu")

    if manager.statistics.level == 0:
        assert len(manager.statistics.spill_totals) == 0
        return

    assert len(manager.statistics.spill_totals) == 1
    nbytes, time = manager.statistics.spill_totals[("gpu", "cpu")]
    assert nbytes == buf.size
    assert time > 0

    buf.owner.spill(target="gpu")
    assert len(manager.statistics.spill_totals) == 2
    nbytes, time = manager.statistics.spill_totals[("cpu", "gpu")]
    assert nbytes == buf.size
    assert time > 0


@pytest.mark.parametrize("manager", [{"statistic_level": 2}], indirect=True)
def test_statistics_expose(manager: SpillManager):
    assert len(manager.statistics.spill_totals) == 0

    buffers: list[SpillableBuffer] = [
        as_buffer(data=rmm.DeviceBuffer(size=10)) for _ in range(10)
    ]

    # Expose the first buffer
    buffers[0].owner.mark_exposed()
    assert len(manager.statistics.exposes) == 1
    stat = next(iter(manager.statistics.exposes.values()))
    assert stat.count == 1
    assert stat.total_nbytes == buffers[0].size
    assert stat.spilled_nbytes == 0

    # Expose all 10 buffers
    for i in range(10):
        buffers[i].owner.mark_exposed()

    # The rest of the ptr accesses should accumulate to a single stat
    # because they resolve to the same traceback.
    assert len(manager.statistics.exposes) == 2
    stat = list(manager.statistics.exposes.values())[1]
    assert stat.count == 9
    assert stat.total_nbytes == buffers[0].size * 9
    assert stat.spilled_nbytes == 0

    # Create and spill 10 new buffers
    buffers: list[SpillableBuffer] = [
        as_buffer(data=rmm.DeviceBuffer(size=10)) for _ in range(10)
    ]

    manager.spill_to_device_limit(0)

    # Expose the new buffers and check that they are counted as spilled
    for i in range(10):
        buffers[i].owner.mark_exposed()
    assert len(manager.statistics.exposes) == 3
    stat = list(manager.statistics.exposes.values())[2]
    assert stat.count == 10
    assert stat.total_nbytes == buffers[0].size * 10
    assert stat.spilled_nbytes == buffers[0].size * 10


def test_spill_on_demand(manager: SpillManager):
    with set_rmm_memory_pool(1024):
        a = as_buffer(data=rmm.DeviceBuffer(size=1024))
        assert isinstance(a, SpillableBuffer)
        assert not a.owner.is_spilled

        with pytest.raises(MemoryError, match="Maximum pool size exceeded"):
            as_buffer(data=rmm.DeviceBuffer(size=1024))

        with spill_on_demand_globally():
            b = as_buffer(data=rmm.DeviceBuffer(size=1024))
            assert a.owner.is_spilled
            assert not b.owner.is_spilled

        with pytest.raises(MemoryError, match="Maximum pool size exceeded"):
            as_buffer(data=rmm.DeviceBuffer(size=1024))


def test_spilling_and_copy_on_write(manager: SpillManager):
    a: SpillableBuffer = as_buffer(data=rmm.DeviceBuffer(size=10))

    b = a.copy(deep=False)
    assert a.owner == b.owner
    a.owner.spill(target="cpu")
    assert a.owner.is_spilled
    assert b.owner.is_spilled

    # Write access trigger copy of `a` into `b` but since `a` is spilled
    # the copy is done in host memory and `a` remains spilled.
    with b.access(mode="write"):
        b.ptr
    assert a.owner.is_spilled
    assert not b.owner.is_spilled

    # Deep copy of the spilled buffer `a`
    b = a.copy(deep=True)
    assert a.owner != b.owner
    assert a.owner.is_spilled
    assert b.owner.is_spilled
    a.owner.spill(target="gpu")
    assert not a.owner.is_spilled
    assert b.owner.is_spilled

    # Deep copy of the unspilled buffer `a`
    b = a.copy(deep=True)
    assert a.owner.spillable
    assert not a.owner.is_spilled
    assert not b.owner.is_spilled

    b = a.copy(deep=False)
    assert a.owner == b.owner
    # Write access trigger copy of `a` into `b` in device memory
    with b.access(mode="write"):
        b.ptr
    assert a.owner != b.owner
    assert not a.owner.is_spilled
    assert not b.owner.is_spilled
    # And `a` and `b` is now separated with there one spilling status
    a.owner.spill(target="cpu")
    assert a.owner.is_spilled
    assert not b.owner.is_spilled
    b.owner.spill(target="cpu")
    assert a.owner.is_spilled
    assert b.owner.is_spilled

    # Read access with a spill lock unspill `a` and allows copy-on-write
    with a.access(mode="read", scope="internal"):
        a.ptr
    b = a.copy(deep=False)
    assert a.owner == b.owner
    assert not a.owner.is_spilled

    # Read access without a spill lock exposes `a` and forces a deep copy
    a.ptr
    b = a.copy(deep=False)
    assert a.owner != b.owner
    assert not a.owner.is_spilled
    assert a.owner.exposed
    assert not b.owner.exposed


def test_scatter_by_map():
    data = range(10)
    with cudf.option_context("spill", True):
        df = cudf.DataFrame(data)
        result = df.scatter_by_map(data)
    for i, res in zip(data, result, strict=True):
        assert_eq(res, cudf.DataFrame([i], index=[i]))


def test_spillable_buffer_access_scope_internal(manager: SpillManager):
    """Test internal scope creates temporary spill lock."""
    buf = as_buffer(rmm.DeviceBuffer(size=100))
    assert buf.owner.spillable

    with buf.access(mode="read", scope="internal"):
        # Buffer is spill locked during context
        assert not buf.owner.spillable
        assert len(buf.owner._spill_locks) == 1
        # Ptr access should work
        assert buf.ptr != 0

    # After context, lock is released
    assert buf.owner.spillable
    assert len(buf.owner._spill_locks) == 0


def test_spillable_buffer_access_scope_external(manager: SpillManager):
    """Test external scope marks buffer as exposed."""
    buf = as_buffer(rmm.DeviceBuffer(size=100))
    assert buf.owner.spillable
    assert not buf.owner.exposed

    with buf.access(mode="read", scope="external"):
        assert not buf.owner.spillable
        assert buf.owner.exposed
        assert buf.ptr != 0

    # After context, buffer remains exposed
    assert not buf.owner.spillable
    assert buf.owner.exposed


def test_spillable_buffer_access_nesting(manager: SpillManager):
    """Test nested access contexts work correctly."""
    buf = as_buffer(rmm.DeviceBuffer(size=100))

    with buf.access(mode="read", scope="internal"):
        assert len(buf.owner._spill_locks) == 1

        with buf.access(mode="read", scope="internal"):
            # Two locks active
            assert len(buf.owner._spill_locks) == 2
            assert not buf.owner.spillable

        # One lock released
        assert len(buf.owner._spill_locks) == 1
        assert not buf.owner.spillable

    # All locks released
    assert len(buf.owner._spill_locks) == 0
    assert buf.owner.spillable


def test_spillable_buffer_access_scope_defaults_to_internal(
    manager: SpillManager,
):
    """Test that scope parameter defaults to internal."""
    buf = as_buffer(rmm.DeviceBuffer(size=100))

    # Should default to scope="internal" if not provided
    with buf.access(mode="read"):
        assert not buf.owner.spillable
        assert len(buf.owner._spill_locks) == 1
        buf.ptr  # Should work

    # After context, lock is released
    assert buf.owner.spillable
    assert len(buf.owner._spill_locks) == 0


def test_spillable_buffer_access_invalid_scope(manager: SpillManager):
    """Test that invalid scope values are rejected."""
    buf = as_buffer(rmm.DeviceBuffer(size=100))

    with pytest.raises(ValueError, match="Invalid scope"):
        with buf.access(mode="read", scope="invalid"):
            pass


def test_column_access_propagates_scope(manager: SpillManager):
    """Test Column.access() propagates scope to buffers."""
    from cudf.core.column import as_column

    col = as_column([1, 2, 3, 4, 5])

    with col.access(mode="read", scope="internal"):
        # All buffers should be spill locked
        if col.data:
            assert not col.data.owner.spillable
            assert len(col.data.owner._spill_locks) >= 1

    # After context, all buffers spillable
    if col.data:
        assert col.data.owner.spillable
        assert len(col.data.owner._spill_locks) == 0
