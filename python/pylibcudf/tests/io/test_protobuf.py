# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa

import pylibcudf as plc


def test_decode_protobuf_wires_python_metadata():
    input_col = plc.Column.from_arrow(
        pa.array([[8, 1], [8, 2]], type=pa.list_(pa.uint8()))
    )
    result = plc.io.protobuf.decode_protobuf(
        input_col,
        schema=[
            (
                1,
                -1,
                0,
                0,  # VARINT
                plc.TypeId.STRING,
                3,  # ENUM_STRING
                False,
                False,
                False,
            )
        ],
        default_ints=[0],
        default_floats=[0.0],
        default_bools=[False],
        default_strings=[b""],
        enum_valid_values=[[1, 2]],
        enum_names=[[b"ONE", b"TWO"]],
        fail_on_errors=True,
    )

    assert result.type().id() == plc.TypeId.STRUCT
    assert result.num_children() == 1
    assert result.children()[0].type().id() == plc.TypeId.STRING
    assert result.children()[0].null_count() == 2
