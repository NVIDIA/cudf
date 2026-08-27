# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import rmm

import pylibcudf as plc

VALID_TYPES = [
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
    pa.float32(),
    pa.float64(),
    pa.bool_(),
    pa.timestamp("s"),
    pa.timestamp("ms"),
    pa.timestamp("us"),
    pa.timestamp("ns"),
    pa.duration("s"),
    pa.duration("ms"),
    pa.duration("us"),
    pa.duration("ns"),
]


@pytest.fixture(params=VALID_TYPES, ids=repr)
def valid_type(request):
    return request.param


class DataBuffer:
    def __init__(self, obj, dtype):
        self.obj = rmm.DeviceBuffer.to_device(obj, plc.utils._get_stream())
        self.dtype = dtype
        self.shape = (int(len(self.obj) / self.dtype.itemsize),)
        self.strides = (self.dtype.itemsize,)
        self.typestr = self.dtype.str

    @property
    def __cuda_array_interface__(self):
        return {
            "data": self.obj.__cuda_array_interface__["data"],
            "shape": self.shape,
            "strides": self.strides,
            "typestr": self.typestr,
            "version": 0,
        }


@pytest.fixture
def input_column(valid_type):
    if valid_type == pa.bool_():
        return pa.array([True, False, True], type=valid_type)
    return pa.array([1, 2, 3], type=valid_type)


@pytest.fixture
def iface_obj(input_column):
    data = input_column.to_numpy(zero_copy_only=False)
    return DataBuffer(data.view("uint8"), data.dtype)


@pytest.mark.parametrize("patch_cai", [True, False])
def test_from_cuda_array_interface(
    monkeypatch, input_column, iface_obj, patch_cai
):
    if patch_cai:
        # patch strides to be None to test C-configuous layout
        monkeypatch.setattr(iface_obj, "strides", None)

    res = plc.Column.from_cuda_array_interface(iface_obj)

    assert_column_eq(input_column, res)


def test_cuda_array_interface_with_mask_raises():
    class CudaArrayInterfaceWithMask:
        @property
        def __cuda_array_interface__(self):
            return {
                "data": (0, False),
                "shape": (1,),
                "typestr": "<i4",
                "version": 3,
                "mask": object(),
            }

    with pytest.raises(
        NotImplementedError,
        match="Masked array-interface inputs are not supported",
    ):
        plc.Column.from_cuda_array_interface(CudaArrayInterfaceWithMask())


def test_from_rmm_buffer():
    # the caller is responsible for keeping the source bytes alive until
    # the H2D copy completes (e.g. when the buffer is accessed below).
    # See https://github.com/rapidsai/rmm/issues/2521
    result = pa.array([1, 2, 3], type=pa.int32())
    data_bytes = result.buffers()[1].to_pybytes()
    expected = plc.Column.from_rmm_buffer(
        rmm.DeviceBuffer.to_device(data_bytes, plc.utils._get_stream()),
        plc.DataType.from_arrow(result.type),
        len(result),
        [],
    )
    assert_column_eq(result, expected)

    result = pa.array(["a", "b", "c"], type=pa.string())
    offsets_bytes = result.buffers()[1].to_pybytes()
    data_bytes = result.buffers()[2].to_pybytes()
    expected = plc.Column.from_rmm_buffer(
        rmm.DeviceBuffer.to_device(data_bytes, plc.utils._get_stream()),
        plc.DataType.from_arrow(result.type),
        len(result),
        [
            plc.Column.from_rmm_buffer(
                rmm.DeviceBuffer.to_device(
                    offsets_bytes, plc.utils._get_stream()
                ),
                plc.DataType(plc.TypeId.INT32),
                4,
                [],
            )
        ],
    )
    assert_column_eq(result, expected)


@pytest.mark.parametrize(
    "dtype, children_data",
    [
        (plc.DataType(plc.TypeId.INT32), [[0, 1, 2]]),
        (plc.DataType(plc.TypeId.STRING), []),
        (plc.DataType(plc.TypeId.STRING), [[0, 1], [0, 1]]),
        (plc.DataType(plc.TypeId.LIST), []),
    ],
)
def test_from_rmm_buffer_invalid(dtype, children_data):
    buff = rmm.DeviceBuffer.to_device(b"", plc.utils._get_stream())
    children = [
        plc.Column.from_arrow(pa.array(child_data))
        for child_data in children_data
    ]
    with pytest.raises(ValueError):
        plc.Column.from_rmm_buffer(
            buff,
            dtype,
            0,
            children,
        )
