# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import (
    DEFAULT_STRUCT_TESTING_TYPE,
    NESTED_STRUCT_TESTING_TYPE,
    assert_column_eq,
    assert_table_eq,
    cudf_raises,
    is_nested_list,
    is_nested_struct,
    is_string,
    metadata_from_arrow_type,
)

import pylibcudf as plc


# TODO: consider moving this to conftest and "pairing"
# it with pa_type, so that they don't get out of sync
# TODO: Test nullable data
@pytest.fixture(scope="module")
def input_column(pa_type):
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        pa_array = pa.array([1, 2, 3], type=pa_type)
    elif pa.types.is_string(pa_type):
        pa_array = pa.array(["a", "b", "c"], type=pa_type)
    elif pa.types.is_boolean(pa_type):
        pa_array = pa.array([True, True, False], type=pa_type)
    elif pa.types.is_list(pa_type):
        if pa_type.value_type == pa.int64():
            pa_array = pa.array([[1], [2, 3], [3]], type=pa_type)
        elif (
            isinstance(pa_type.value_type, pa.ListType)
            and pa_type.value_type.value_type == pa.int64()
        ):
            pa_array = pa.array([[[1]], [[2, 3]], [[3]]], type=pa_type)
        else:
            raise ValueError("Unsupported type " + pa_type.value_type)
    elif pa.types.is_struct(pa_type):
        if not is_nested_struct(pa_type):
            pa_array = pa.array([{"v": 1}, {"v": 2}, {"v": 3}], type=pa_type)
        else:
            pa_array = pa.array(
                [
                    {"a": 1, "b_struct": {"b": 1.0}},
                    {"a": 2, "b_struct": {"b": 2.0}},
                    {"a": 3, "b_struct": {"b": 3.0}},
                ],
                type=pa_type,
            )
    else:
        raise ValueError("Unsupported type")
    return pa_array, plc.Column.from_arrow(pa_array)


@pytest.fixture(scope="module")
def index_column():
    # Index column for testing gather/scatter, always integral.
    pa_array = pa.array([1, 2, 3])
    return pa_array, plc.Column.from_arrow(pa_array)


@pytest.fixture(scope="module")
def target_column(pa_type):
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        pa_array = pa.array([4, 5, 6, 7, 8, 9], type=pa_type)
    elif pa.types.is_string(pa_type):
        pa_array = pa.array(["d", "e", "f", "g", "h", "i"], type=pa_type)
    elif pa.types.is_boolean(pa_type):
        pa_array = pa.array(
            [False, True, True, False, True, False], type=pa_type
        )
    elif pa.types.is_list(pa_type):
        if pa_type.value_type == pa.int64():
            pa_array = pa.array(
                [[4], [5, 6], [7], [8], [9], [10]], type=pa_type
            )
        elif (
            isinstance(pa_type.value_type, pa.ListType)
            and pa_type.value_type.value_type == pa.int64()
        ):
            pa_array = pa.array(
                [[[4]], [[5, 6]], [[7]], [[8]], [[9]], [[10]]], type=pa_type
            )
        else:
            raise ValueError("Unsupported type")
    elif pa.types.is_struct(pa_type):
        if not is_nested_struct(pa_type):
            pa_array = pa.array(
                [{"v": 4}, {"v": 5}, {"v": 6}, {"v": 7}, {"v": 8}, {"v": 9}],
                type=pa_type,
            )
        else:
            pa_array = pa.array(
                [
                    {"a": 4, "b_struct": {"b": 4.0}},
                    {"a": 5, "b_struct": {"b": 5.0}},
                    {"a": 6, "b_struct": {"b": 6.0}},
                    {"a": 7, "b_struct": {"b": 7.0}},
                    {"a": 8, "b_struct": {"b": 8.0}},
                    {"a": 9, "b_struct": {"b": 9.0}},
                ],
                type=pa_type,
            )
    else:
        raise ValueError("Unsupported type")
    return pa_array, plc.Column.from_arrow(pa_array)


@pytest.fixture
def mutable_target_column(target_column):
    _, plc_target_column = target_column
    return plc_target_column.copy()


@pytest.fixture(scope="module")
def source_table(input_column):
    pa_input_column, _ = input_column
    pa_table = pa.table([pa_input_column] * 3, [""] * 3)
    return pa_table, plc.Table.from_arrow(pa_table)


@pytest.fixture(scope="module")
def target_table(target_column):
    pa_target_column, _ = target_column
    pa_table = pa.table([pa_target_column] * 3, [""] * 3)
    return pa_table, plc.Table.from_arrow(pa_table)


@pytest.fixture(scope="module")
def source_scalar(pa_type):
    if pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type):
        pa_scalar = pa.scalar(1, type=pa_type)
    elif pa.types.is_string(pa_type):
        pa_scalar = pa.scalar("a", type=pa_type)
    elif pa.types.is_boolean(pa_type):
        pa_scalar = pa.scalar(False, type=pa_type)
    elif pa.types.is_list(pa_type):
        if pa_type.value_type == pa.int64():
            pa_scalar = pa.scalar([1, 2, 3, 4], type=pa_type)
        elif (
            isinstance(pa_type.value_type, pa.ListType)
            and pa_type.value_type.value_type == pa.int64()
        ):
            pa_scalar = pa.scalar([[1, 2, 3, 4]], type=pa_type)
        else:
            raise ValueError("Unsupported type")
    elif pa.types.is_struct(pa_type):
        if not is_nested_struct(pa_type):
            pa_scalar = pa.scalar({"v": 1}, type=pa_type)
        else:
            pa_scalar = pa.scalar(
                {"a": 1, "b_struct": {"b": 1.0}}, type=pa_type
            )
    else:
        raise ValueError("Unsupported type")
    return pa_scalar, plc.Scalar.from_arrow(pa_scalar)


@pytest.fixture(scope="module")
def mask(target_column):
    pa_target_column, _ = target_column
    pa_mask = pa.array([True, False] * (len(pa_target_column) // 2))
    return pa_mask, plc.Column.from_arrow(pa_mask)


def test_gather(target_table, index_column):
    pa_target_table, plc_target_table = target_table
    pa_index_column, plc_index_column = index_column
    result = plc.copying.gather(
        plc_target_table,
        plc_index_column,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    )
    expected = pa_target_table.take(pa_index_column)
    assert_table_eq(expected, result)


def test_gather_map_has_nulls(target_table):
    _, plc_target_table = target_table
    gather_map = plc.Column.from_arrow(pa.array([0, 1, None]))
    with cudf_raises(ValueError):
        plc.copying.gather(
            plc_target_table,
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        )


def _pyarrow_index_to_mask(indices, mask_size):
    # Convert a list of indices to a boolean mask.
    return pc.is_in(pa.array(range(mask_size)), pa.array(indices))


def _pyarrow_boolean_mask_scatter_column(source, mask, target):
    if isinstance(source, pa.Scalar):
        # if_else requires array lengths to match exactly or the replacement must be a
        # scalar, so we use this in the scalar case.
        return pc.if_else(mask, target, source)

    if isinstance(source, pa.ChunkedArray):
        source = source.combine_chunks()
    if isinstance(target, pa.ChunkedArray):
        target = target.combine_chunks()

    # replace_with_mask accepts a column whose size is the number of true values in
    # the mask, so we can use it for columnar scatters.
    return pc.replace_with_mask(target, mask, source)


def _pyarrow_boolean_mask_scatter_table(source, mask, target_table):
    # pyarrow equivalent of cudf's boolean_mask_scatter.
    return pa.table(
        [
            _pyarrow_boolean_mask_scatter_column(r, mask, v)
            for v, r in zip(target_table, source, strict=True)
        ],
        [""] * target_table.num_columns,
    )


def test_scatter_table(
    source_table,
    index_column,
    target_table,
):
    pa_source_table, plc_source_table = source_table
    pa_index_column, plc_index_column = index_column
    pa_target_table, plc_target_table = target_table
    result = plc.copying.scatter(
        plc_source_table,
        plc_index_column,
        plc_target_table,
    )

    if pa.types.is_list(
        dtype := pa_target_table[0].type
    ) or pa.types.is_struct(dtype):
        # pyarrow does not support scattering with list data. If and when they do,
        # replace this hardcoding with their implementation.
        with pytest.raises(pa.ArrowNotImplementedError):
            _pyarrow_boolean_mask_scatter_table(
                pa_source_table,
                _pyarrow_index_to_mask(
                    pa_index_column, pa_target_table.num_rows
                ),
                pa_target_table,
            )

        if pa.types.is_list(dtype := pa_target_table[0].type):
            if is_nested_list(dtype):
                expected = pa.table(
                    [pa.array([[[4]], [[1]], [[2, 3]], [[3]], [[9]], [[10]]])]
                    * 3,
                    [""] * 3,
                )
            else:
                expected = pa.table(
                    [pa.array([[4], [1], [2, 3], [3], [9], [10]])] * 3,
                    [""] * 3,
                )
        elif pa.types.is_struct(dtype):
            if is_nested_struct(dtype):
                expected = pa.table(
                    [
                        pa.array(
                            [
                                {"a": 4, "b_struct": {"b": 4.0}},
                                {"a": 1, "b_struct": {"b": 1.0}},
                                {"a": 2, "b_struct": {"b": 2.0}},
                                {"a": 3, "b_struct": {"b": 3.0}},
                                {"a": 8, "b_struct": {"b": 8.0}},
                                {"a": 9, "b_struct": {"b": 9.0}},
                            ],
                            type=NESTED_STRUCT_TESTING_TYPE,
                        )
                    ]
                    * 3,
                    [""] * 3,
                )
            else:
                expected = pa.table(
                    [
                        pa.array(
                            [
                                {"v": 4},
                                {"v": 1},
                                {"v": 2},
                                {"v": 3},
                                {"v": 8},
                                {"v": 9},
                            ],
                            type=DEFAULT_STRUCT_TESTING_TYPE,
                        )
                    ]
                    * 3,
                    [""] * 3,
                )
    else:
        expected = _pyarrow_boolean_mask_scatter_table(
            pa_source_table,
            _pyarrow_index_to_mask(pa_index_column, pa_target_table.num_rows),
            pa_target_table,
        )

    assert_table_eq(expected, result)


def test_scatter_table_num_col_mismatch(
    source_table, index_column, target_table
):
    # Number of columns in source and target must match.
    _, plc_source_table = source_table
    _, plc_index_column = index_column
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.scatter(
            plc.Table(plc_source_table.columns()[:2]),
            plc_index_column,
            plc_target_table,
        )


def test_scatter_table_num_row_mismatch(source_table, target_table):
    # Number of rows in source and scatter map must match.
    _, plc_source_table = source_table
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.scatter(
            plc_source_table,
            plc.Column.from_arrow(
                pa.array(range(plc_source_table.num_rows() * 2))
            ),
            plc_target_table,
        )


def test_scatter_table_map_has_nulls(source_table, target_table):
    _, plc_source_table = source_table
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.scatter(
            plc_source_table,
            plc.Column.from_arrow(
                pa.array([None] * plc_source_table.num_rows())
            ),
            plc_target_table,
        )


def test_scatter_table_type_mismatch(source_table, index_column, target_table):
    _, plc_source_table = source_table
    _, plc_index_column = index_column
    _, plc_target_table = target_table
    with cudf_raises(TypeError):
        if plc.traits.is_integral_not_bool(
            dtype := plc_target_table.columns()[0].type()
        ) or plc.traits.is_floating_point(dtype):
            pa_array = pa.array([True] * plc_source_table.num_rows())
        else:
            pa_array = pa.array([1] * plc_source_table.num_rows())
        ncol = plc_source_table.num_columns()
        pa_table = pa.table([pa_array] * ncol, [""] * ncol)
        plc.copying.scatter(
            plc.Table.from_arrow(pa_table),
            plc_index_column,
            plc_target_table,
        )


def test_scatter_scalars(
    source_scalar,
    index_column,
    target_table,
):
    pa_source_scalar, plc_source_scalar = source_scalar
    pa_index_column, plc_index_column = index_column
    pa_target_table, plc_target_table = target_table
    result = plc.copying.scatter(
        [plc_source_scalar] * plc_target_table.num_columns(),
        plc_index_column,
        plc_target_table,
    )

    expected = _pyarrow_boolean_mask_scatter_table(
        [pa_source_scalar] * plc_target_table.num_columns(),
        pc.invert(
            _pyarrow_index_to_mask(pa_index_column, pa_target_table.num_rows)
        ),
        pa_target_table,
    )

    assert_table_eq(expected, result)


def test_scatter_scalars_num_scalars_mismatch(
    source_scalar, index_column, target_table
):
    _, plc_source_scalar = source_scalar
    _, plc_index_column = index_column
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.scatter(
            [plc_source_scalar] * (plc_target_table.num_columns() - 1),
            plc_index_column,
            plc_target_table,
        )


def test_scatter_scalars_map_has_nulls(source_scalar, target_table):
    _, plc_source_scalar = source_scalar
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.scatter(
            [plc_source_scalar] * plc_target_table.num_columns(),
            plc.Column.from_arrow(pa.array([None, None])),
            plc_target_table,
        )


def test_scatter_scalars_type_mismatch(index_column, target_table):
    _, plc_index_column = index_column
    _, plc_target_table = target_table
    with cudf_raises(TypeError):
        if plc.traits.is_integral_not_bool(
            dtype := plc_target_table.columns()[0].type()
        ) or plc.traits.is_floating_point(dtype):
            plc_source_scalar = [plc.Scalar.from_arrow(pa.scalar(True))]
        else:
            plc_source_scalar = [plc.Scalar.from_arrow(pa.scalar(1))]
        plc.copying.scatter(
            plc_source_scalar * plc_target_table.num_columns(),
            plc_index_column,
            plc_target_table,
        )


def test_empty_like_column(input_column):
    _, plc_input_column = input_column
    result = plc.copying.empty_like(plc_input_column)
    assert result.type() == plc_input_column.type()


def test_empty_like_table(source_table):
    _, plc_source_table = source_table
    result = plc.copying.empty_like(plc_source_table)
    assert result.num_columns() == plc_source_table.num_columns()
    for icol, rcol in zip(
        plc_source_table.columns(), result.columns(), strict=True
    ):
        assert rcol.type() == icol.type()


def test_reverse_column(target_column):
    pa_target_column, plc_target_column = target_column
    result = plc.copying.reverse(plc_target_column)
    reversed_indices = pa.array(range(len(pa_target_column) - 1, -1, -1))
    expected = pa_target_column.take(reversed_indices)
    assert_column_eq(expected, result)


def test_reverse_table(source_table):
    pa_source_table, plc_source_table = source_table
    result = plc.copying.reverse(plc_source_table)
    reversed_indices = pa.array(range(pa_source_table.num_rows - 1, -1, -1))
    expected = pa_source_table.take(reversed_indices)
    assert_table_eq(expected, result)


@pytest.mark.parametrize("size", [None, 10])
def test_allocate_like(input_column, size):
    _, plc_input_column = input_column
    if plc.traits.is_fixed_width(plc_input_column.type()):
        result = plc.copying.allocate_like(
            plc_input_column,
            plc.copying.MaskAllocationPolicy.RETAIN,
            size=size,
        )
        assert result.type() == plc_input_column.type()
        assert result.size() == (
            plc_input_column.size() if size is None else size
        )
    else:
        with pytest.raises(TypeError):
            plc.copying.allocate_like(
                plc_input_column,
                plc.copying.MaskAllocationPolicy.RETAIN,
                size=size,
            )


def test_copy_range_in_place(
    input_column, mutable_target_column, target_column
):
    pa_input_column, plc_input_column = input_column

    pa_target_column, _ = target_column

    if not plc.traits.is_fixed_width(mutable_target_column.type()):
        with pytest.raises(TypeError):
            plc.copying.copy_range_in_place(
                plc_input_column,
                mutable_target_column,
                0,
                plc_input_column.size(),
                0,
            )
    else:
        plc.copying.copy_range_in_place(
            plc_input_column,
            mutable_target_column,
            0,
            plc_input_column.size(),
            0,
        )
        expected = _pyarrow_boolean_mask_scatter_column(
            pa_input_column,
            _pyarrow_index_to_mask(
                range(len(pa_input_column)), len(pa_target_column)
            ),
            pa_target_column,
        )
        assert_column_eq(expected, mutable_target_column)


def test_copy_range_in_place_out_of_bounds(
    input_column, mutable_target_column
):
    _, plc_input_column = input_column

    if plc.traits.is_fixed_width(mutable_target_column.type()):
        with cudf_raises(IndexError):
            plc.copying.copy_range_in_place(
                plc_input_column,
                mutable_target_column,
                5,
                5 + plc_input_column.size(),
                0,
            )


def test_copy_range_in_place_different_types(mutable_target_column):
    if plc.traits.is_integral_not_bool(
        dtype := mutable_target_column.type()
    ) or plc.traits.is_floating_point(dtype):
        plc_input_column = plc.Column.from_arrow(pa.array(["a", "b", "c"]))
    else:
        plc_input_column = plc.Column.from_arrow(pa.array([1, 2, 3]))

    with cudf_raises(TypeError):
        plc.copying.copy_range_in_place(
            plc_input_column,
            mutable_target_column,
            0,
            plc_input_column.size(),
            0,
        )


def test_copy_range_in_place_null_mismatch(
    input_column, mutable_target_column
):
    pa_input_column, _ = input_column

    if plc.traits.is_fixed_width(mutable_target_column.type()):
        pa_input_column = pc.if_else(
            _pyarrow_index_to_mask([0], len(pa_input_column)),
            pa_input_column,
            pa.scalar(None, type=pa_input_column.type),
        )
        input_column = plc.Column.from_arrow(pa_input_column)
        with cudf_raises(ValueError):
            plc.copying.copy_range_in_place(
                input_column,
                mutable_target_column,
                0,
                input_column.size(),
                0,
            )


def test_copy_range(input_column, target_column):
    pa_input_column, plc_input_column = input_column
    pa_target_column, plc_target_column = target_column
    if plc.traits.is_fixed_width(
        dtype := plc_target_column.type()
    ) or is_string(dtype):
        result = plc.copying.copy_range(
            plc_input_column,
            plc_target_column,
            0,
            plc_input_column.size(),
            0,
        )
        expected = _pyarrow_boolean_mask_scatter_column(
            pa_input_column,
            _pyarrow_index_to_mask(
                range(len(pa_input_column)), len(pa_target_column)
            ),
            pa_target_column,
        )
        assert_column_eq(expected, result)
    else:
        with pytest.raises(TypeError):
            plc.copying.copy_range(
                plc_input_column,
                plc_target_column,
                0,
                plc_input_column.size(),
                0,
            )


def test_copy_range_out_of_bounds(input_column, target_column):
    _, plc_input_column = input_column
    _, plc_target_column = target_column
    with cudf_raises(IndexError):
        plc.copying.copy_range(
            plc_input_column,
            plc_target_column,
            5,
            5 + plc_input_column.size(),
            0,
        )


def test_copy_range_different_types(target_column):
    _, plc_target_column = target_column
    if plc.traits.is_integral_not_bool(
        dtype := plc_target_column.type()
    ) or plc.traits.is_floating_point(dtype):
        plc_input_column = plc.Column.from_arrow(pa.array(["a", "b", "c"]))
    else:
        plc_input_column = plc.Column.from_arrow(pa.array([1, 2, 3]))

    with cudf_raises(TypeError):
        plc.copying.copy_range(
            plc_input_column,
            plc_target_column,
            0,
            plc_input_column.size(),
            0,
        )


def test_shift(target_column, source_scalar):
    pa_source_scalar, plc_source_scalar = source_scalar
    pa_target_column, plc_target_column = target_column
    shift = 2
    if plc.traits.is_fixed_width(
        dtype := plc_target_column.type()
    ) or is_string(dtype):
        result = plc.copying.shift(plc_target_column, shift, plc_source_scalar)
        expected = pa.concat_arrays(
            [pa.array([pa_source_scalar] * shift), pa_target_column[:-shift]]
        )
        assert_column_eq(expected, result)
    else:
        with pytest.raises(TypeError):
            plc.copying.shift(plc_target_column, shift, source_scalar)


def test_shift_type_mismatch(target_column):
    _, plc_target_column = target_column
    if plc.traits.is_integral_not_bool(
        dtype := plc_target_column.type()
    ) or plc.traits.is_floating_point(dtype):
        fill_value = plc.Scalar.from_arrow(pa.scalar("a"))
    else:
        fill_value = plc.Scalar.from_arrow(pa.scalar(1))

    with cudf_raises(TypeError):
        plc.copying.shift(plc_target_column, 2, fill_value)


def test_slice_column(target_column):
    pa_target_column, plc_target_column = target_column
    bounds = list(range(6))
    upper_bounds = bounds[1::2]
    lower_bounds = bounds[::2]
    result = plc.copying.slice(plc_target_column, bounds)
    for lb, ub, slice_ in zip(lower_bounds, upper_bounds, result, strict=True):
        assert_column_eq(pa_target_column[lb:ub], slice_)


def test_slice_column_wrong_length(target_column):
    _, plc_target_column = target_column
    with cudf_raises(ValueError):
        plc.copying.slice(plc_target_column, list(range(5)))


def test_slice_column_decreasing(target_column):
    _, plc_target_column = target_column
    with cudf_raises(ValueError):
        plc.copying.slice(plc_target_column, list(range(5, -1, -1)))


def test_slice_column_out_of_bounds(target_column):
    _, plc_target_column = target_column
    with cudf_raises(IndexError):
        plc.copying.slice(plc_target_column, list(range(2, 8)))


def test_slice_table(target_table):
    pa_target_table, plc_target_table = target_table
    bounds = list(range(6))
    upper_bounds = bounds[1::2]
    lower_bounds = bounds[::2]
    result = plc.copying.slice(plc_target_table, bounds)
    for lb, ub, slice_ in zip(lower_bounds, upper_bounds, result, strict=True):
        assert_table_eq(pa_target_table[lb:ub], slice_)


def test_split_column(target_column):
    upper_bounds = [1, 3, 5]
    lower_bounds = [0, *upper_bounds]
    upper_bounds_extended = [*upper_bounds, None]  # None means to the end
    pa_target_column, plc_target_column = target_column
    result = plc.copying.split(plc_target_column, upper_bounds)
    for lb, ub, split in zip(
        lower_bounds, upper_bounds_extended, result, strict=True
    ):
        if ub is None:
            assert_column_eq(pa_target_column[lb:], split)
        else:
            assert_column_eq(pa_target_column[lb:ub], split)


def test_split_column_decreasing(target_column):
    _, plc_target_column = target_column
    with cudf_raises(ValueError):
        plc.copying.split(plc_target_column, list(range(5, -1, -1)))


def test_split_column_out_of_bounds(target_column):
    _, plc_target_column = target_column
    with cudf_raises(IndexError):
        plc.copying.split(plc_target_column, list(range(5, 8)))


def test_split_table(target_table):
    pa_target_table, plc_target_table = target_table

    upper_bounds = [1, 3, 5]
    lower_bounds = [0, *upper_bounds]
    upper_bounds_extended = [*upper_bounds, None]  # None means to the end
    result = plc.copying.split(plc_target_table, upper_bounds)
    for lb, ub, split in zip(
        lower_bounds, upper_bounds_extended, result, strict=True
    ):
        if ub is None:
            assert_table_eq(pa_target_table[lb:], split)
        else:
            assert_table_eq(pa_target_table[lb:ub], split)


def test_copy_if_else_column_column(target_column, mask, source_scalar):
    pa_target_column, plc_target_column = target_column
    pa_source_scalar, _ = source_scalar
    pa_mask, plc_mask = mask

    pa_other_column = pa.concat_arrays(
        [pa.array([pa_source_scalar] * 2), pa_target_column[:-2]]
    )
    plc_other_column = plc.Column.from_arrow(pa_other_column)

    result = plc.copying.copy_if_else(
        plc_target_column,
        plc_other_column,
        plc_mask,
    )

    expected = pc.if_else(
        pa_mask,
        pa_target_column,
        pa_other_column,
    )
    assert_column_eq(expected, result)


def test_copy_if_else_wrong_type(target_column, mask):
    _, plc_target_column = target_column
    _, plc_mask = mask
    if plc.traits.is_integral_not_bool(
        dtype := plc_target_column.type()
    ) or plc.traits.is_floating_point(dtype):
        plc_input_column = plc.Column.from_arrow(
            pa.array(["a"] * plc_target_column.size())
        )
    else:
        plc_input_column = plc.Column.from_arrow(
            pa.array([1] * plc_target_column.size())
        )

    with cudf_raises(TypeError):
        plc.copying.copy_if_else(plc_input_column, plc_target_column, plc_mask)


def test_copy_if_else_wrong_type_mask(target_column):
    _, plc_target_column = target_column
    with cudf_raises(TypeError):
        plc.copying.copy_if_else(
            plc_target_column,
            plc_target_column,
            plc.Column.from_arrow(
                pa.array([1.0, 2.0] * (plc_target_column.size() // 2))
            ),
        )


def test_copy_if_else_wrong_size(target_column):
    _, plc_target_column = target_column
    with cudf_raises(ValueError):
        plc.copying.copy_if_else(
            plc.Column.from_arrow(pa.array([1])),
            plc_target_column,
            plc.Column.from_arrow(
                pa.array([True, False] * (plc_target_column.size() // 2))
            ),
        )


def test_copy_if_else_wrong_size_mask(target_column):
    _, plc_target_column = target_column
    with cudf_raises(ValueError):
        plc.copying.copy_if_else(
            plc_target_column,
            plc_target_column,
            plc.Column.from_arrow(pa.array([True])),
        )


@pytest.mark.parametrize("array_left", [True, False])
def test_copy_if_else_column_scalar(
    target_column,
    source_scalar,
    array_left,
    mask,
):
    pa_target_column, plc_target_column = target_column
    pa_source_scalar, plc_source_scalar = source_scalar
    pa_mask, plc_mask = mask

    args = (
        (plc_target_column, plc_source_scalar)
        if array_left
        else (plc_source_scalar, plc_target_column)
    )
    result = plc.copying.copy_if_else(
        *args,
        plc_mask,
    )

    pa_args = (
        (pa_target_column, pa_source_scalar)
        if array_left
        else (pa_source_scalar, pa_target_column)
    )
    expected = pc.if_else(
        pa_mask,
        *pa_args,
    )
    assert_column_eq(expected, result)


def test_boolean_mask_scatter_from_table(
    source_table,
    target_table,
    mask,
):
    pa_source_table, plc_source_table = source_table
    pa_target_table, plc_target_table = target_table
    pa_mask, plc_mask = mask

    result = plc.copying.boolean_mask_scatter(
        plc_source_table,
        plc_target_table,
        plc_mask,
    )

    if pa.types.is_list(
        dtype := pa_target_table[0].type
    ) or pa.types.is_struct(dtype):
        # pyarrow does not support scattering with list data. If and when they do,
        # replace this hardcoding with their implementation.
        with pytest.raises(pa.ArrowNotImplementedError):
            _pyarrow_boolean_mask_scatter_table(
                pa_source_table, pa_mask, pa_target_table
            )

        if pa.types.is_list(dtype := pa_target_table[0].type):
            if is_nested_list(dtype):
                expected = pa.table(
                    [
                        pa.array(
                            [[[1]], [[5, 6]], [[2, 3]], [[8]], [[3]], [[10]]]
                        )
                    ]
                    * 3,
                    [""] * 3,
                )
            else:
                expected = pa.table(
                    [pa.array([[1], [5, 6], [2, 3], [8], [3], [10]])] * 3,
                    [""] * 3,
                )
        elif pa.types.is_struct(dtype):
            if is_nested_struct(dtype):
                expected = pa.table(
                    [
                        pa.array(
                            [
                                {"a": 1, "b_struct": {"b": 1.0}},
                                {"a": 5, "b_struct": {"b": 5.0}},
                                {"a": 2, "b_struct": {"b": 2.0}},
                                {"a": 7, "b_struct": {"b": 7.0}},
                                {"a": 3, "b_struct": {"b": 3.0}},
                                {"a": 9, "b_struct": {"b": 9.0}},
                            ],
                            type=NESTED_STRUCT_TESTING_TYPE,
                        )
                    ]
                    * 3,
                    [""] * 3,
                )
            else:
                expected = pa.table(
                    [
                        pa.array(
                            [
                                {"v": 1},
                                {"v": 5},
                                {"v": 2},
                                {"v": 7},
                                {"v": 3},
                                {"v": 9},
                            ],
                            type=DEFAULT_STRUCT_TESTING_TYPE,
                        )
                    ]
                    * 3,
                    [""] * 3,
                )
    else:
        expected = _pyarrow_boolean_mask_scatter_table(
            pa_source_table, pa_mask, pa_target_table
        )

    assert_table_eq(expected, result)


def test_boolean_mask_scatter_from_wrong_num_cols(source_table, target_table):
    _, plc_source_table = source_table
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.boolean_mask_scatter(
            plc.Table(plc_source_table.columns()[:2]),
            plc_target_table,
            plc.Column.from_arrow(pa.array([True, False] * 3)),
        )


def test_boolean_mask_scatter_from_wrong_mask_size(source_table, target_table):
    _, plc_source_table = source_table
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.boolean_mask_scatter(
            plc_source_table,
            plc_target_table,
            plc.Column.from_arrow(pa.array([True, False] * 2)),
        )


def test_boolean_mask_scatter_from_wrong_num_true(source_table, target_table):
    _, plc_source_table = source_table
    _, plc_target_table = target_table
    with cudf_raises(ValueError):
        plc.copying.boolean_mask_scatter(
            plc.Table(plc_source_table.columns()[:2]),
            plc_target_table,
            plc.Column.from_arrow(
                pa.array([True, False] * 2 + [False, False])
            ),
        )


def test_boolean_mask_scatter_from_wrong_col_type(target_table, mask):
    _, plc_target_table = target_table
    _, plc_mask = mask
    if plc.traits.is_integral_not_bool(
        dtype := plc_target_table.columns()[0].type()
    ) or plc.traits.is_floating_point(dtype):
        input_column = plc.Column.from_arrow(pa.array(["a", "b", "c"]))
    else:
        input_column = plc.Column.from_arrow(pa.array([1, 2, 3]))

    with cudf_raises(TypeError):
        plc.copying.boolean_mask_scatter(
            plc.Table([input_column] * 3), plc_target_table, plc_mask
        )


def test_boolean_mask_scatter_from_wrong_mask_type(source_table, target_table):
    _, plc_source_table = source_table
    _, plc_target_table = target_table
    with cudf_raises(TypeError):
        plc.copying.boolean_mask_scatter(
            plc_source_table,
            plc_target_table,
            plc.Column.from_arrow(pa.array([1.0, 2.0] * 3)),
        )


def test_boolean_mask_scatter_from_scalars(
    source_scalar,
    target_table,
    mask,
):
    pa_source_scalar, plc_source_scalar = source_scalar
    pa_target_table, plc_target_table = target_table
    pa_mask, plc_mask = mask
    result = plc.copying.boolean_mask_scatter(
        [plc_source_scalar] * 3,
        plc_target_table,
        plc_mask,
    )

    expected = _pyarrow_boolean_mask_scatter_table(
        [pa_source_scalar] * plc_target_table.num_columns(),
        pc.invert(pa_mask),
        pa_target_table,
    )

    assert_table_eq(expected, result)


def test_get_element(input_column):
    index = 1
    pa_input_column, plc_input_column = input_column
    result = plc.copying.get_element(plc_input_column, index)
    assert (
        result.to_arrow(
            metadata=metadata_from_arrow_type(pa_input_column.type)
        ).as_py()
        == pa_input_column[index].as_py()
    )


def test_get_element_out_of_bounds(input_column):
    _, plc_input_column = input_column
    with cudf_raises(IndexError):
        plc.copying.get_element(plc_input_column, 100)


def test_gather_zero_columns_preserves_num_rows():
    source = plc.Table([], num_rows=5)
    gather_map = plc.Column.from_arrow(pa.array([0, 2, 4, 1], type=pa.int32()))
    result = plc.copying.gather(
        source, gather_map, plc.copying.OutOfBoundsPolicy.DONT_CHECK
    )
    assert result.num_columns() == 0
    assert result.num_rows() == 4


def test_scatter_zero_columns_preserves_num_rows():
    source = plc.Table([], num_rows=2)
    target = plc.Table([], num_rows=5)
    scatter_map = plc.Column.from_arrow(pa.array([0, 3], type=pa.int32()))
    result = plc.copying.scatter(source, scatter_map, target)
    assert result.num_columns() == 0
    assert result.num_rows() == 5


def test_slice_zero_columns_preserves_num_rows():
    result = plc.copying.slice(plc.Table([], num_rows=10), [1, 4, 5, 9])
    assert len(result) == 2
    assert result[0].num_columns() == 0
    assert result[0].num_rows() == 3
    assert result[1].num_rows() == 4


def test_split_zero_columns_preserves_num_rows():
    result = plc.copying.split(plc.Table([], num_rows=10), [4])
    assert len(result) == 2
    assert result[0].num_columns() == 0
    assert result[0].num_rows() == 4
    assert result[1].num_rows() == 6
