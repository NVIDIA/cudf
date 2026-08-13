# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator import dereference

from libcpp.functional cimport reference_wrapper
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
# TODO: We want to make cpp a more full-featured package so that we can access
# directly from that. It will make namespacing much cleaner in pylibcudf. What
# we really want here would be
# cimport libcudf... libcudf.copying.algo(...)
from pylibcudf.libcudf cimport copying as cpp_copying
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from pylibcudf.libcudf.copying cimport (
    mask_allocation_policy,
    out_of_bounds_policy,
)
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type

from pylibcudf.libcudf.copying import \
    mask_allocation_policy as MaskAllocationPolicy  # no-cython-lint
from pylibcudf.libcudf.copying import \
    out_of_bounds_policy as OutOfBoundsPolicy  # no-cython-lint
from pylibcudf.libcudf.copying import \
    sample_with_replacement as SampleWithReplacement  # no-cython-lint

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .utils cimport _as_vector, _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t


__all__ = [
    "MaskAllocationPolicy",
    "OutOfBoundsPolicy",
    "SampleWithReplacement",
    "allocate_like",
    "boolean_mask_scatter",
    "copy_if_else",
    "copy_range",
    "copy_range_in_place",
    "empty_like",
    "gather",
    "get_element",
    "scatter",
    "shift",
    "slice",
    "split",
]

cpdef Table gather(
    Table source_table,
    Column gather_map,
    out_of_bounds_policy bounds_policy,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Select rows from source_table according to the provided gather_map.

    For details, see :cpp:func:`gather`.

    Parameters
    ----------
    source_table : Table
        The table object from which to pull data.
    gather_map : Column
        The list of row indices to pull out of the source table.
    bounds_policy : out_of_bounds_policy
        Controls whether out of bounds indices are checked and nullified in the
        output or if indices are assumed to be in bounds.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Table
        The result of the gather

    Raises
    ------
    ValueError
        If the gather_map contains nulls.
    """
    cdef unique_ptr[table] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_source_table = source_table.view()
    cdef column_view c_gather_map = gather_map.view()
    with nogil:
        c_result = cpp_copying.gather(
            c_source_table,
            c_gather_map,
            bounds_policy,
            _cs,
            mr.get_mr()
        )

    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef Table scatter(
    TableOrListOfScalars source,
    Column scatter_map,
    Table target_table,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Scatter from source into target_table according to scatter_map.

    If source is a table, it specifies rows to scatter. If source is a list,
    each scalar is scattered into the corresponding column in the ``target_table``.

    For details, see :cpp:func:`scatter`.

    Parameters
    ----------
    source : Union[Table, List[Scalar]]
        The table object or list of scalars from which to pull data.
    scatter_map : Column
        A mapping from rows in source to rows in target_table.
    target_table : Table
        The table object into which to scatter data.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        The result of the scatter

    Raises
    ------
    ValueError
        If any of the following occur:
            - scatter_map contains null values.
            - source is a Table and the number of columns in source does not match the
              number of columns in target.
            - source is a Table and the number of rows in source does not match the
              number of elements in scatter_map.
            - source is a List[Scalar] and the number of scalars does not match the
              number of columns in target.
    TypeError
        If data types of the source and target columns do not match.
    """
    cdef unique_ptr[table] c_result
    cdef vector[reference_wrapper[const scalar]] source_scalars
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef table_view c_source_table
    cdef column_view c_scatter_map
    cdef table_view c_target_table

    mr = _get_memory_resource(mr)

    if TableOrListOfScalars is Table:
        c_source_table = source.view()
        c_scatter_map = scatter_map.view()
        c_target_table = target_table.view()
        with nogil:
            c_result = cpp_copying.scatter(
                c_source_table,
                c_scatter_map,
                c_target_table,
                _cs,
                mr.get_mr()
            )
    else:
        source_scalars = _as_vector(source)
        c_scatter_map = scatter_map.view()
        c_target_table = target_table.view()
        with nogil:
            c_result = cpp_copying.scatter(
                source_scalars,
                c_scatter_map,
                c_target_table,
                _cs,
                mr.get_mr()
            )
    return Table.from_libcudf(move(c_result), _stream, mr)


cpdef ColumnOrTable empty_like(
    ColumnOrTable input, object stream=None, DeviceMemoryResource mr=None
):
    """Create an empty column or table with the same type as ``input``.

    For details, see :cpp:func:`empty_like`.

    Parameters
    ----------
    input : Union[Column, Table]
        The column or table to use as a template for the output.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Union[Column, Table]
        An empty column or table with the same type(s) as ``input``.
    """
    cdef unique_ptr[table] c_tbl_result
    cdef unique_ptr[column] c_col_result
    cdef Stream _stream = _get_stream(stream)
    cdef column_view c_input_column
    cdef table_view c_input_table

    mr = _get_memory_resource(mr)
    if ColumnOrTable is Column:
        c_input_column = input.view()
        with nogil:
            c_col_result = cpp_copying.empty_like(c_input_column)
        return Column.from_libcudf(move(c_col_result), _stream, mr)
    else:
        c_input_table = input.view()
        with nogil:
            c_tbl_result = cpp_copying.empty_like(c_input_table)
        return Table.from_libcudf(move(c_tbl_result), _stream, mr)


cpdef Column allocate_like(
    Column input_column,
    mask_allocation_policy policy,
    size=None,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Allocate a column with the same type as input_column.

    For details, see :cpp:func:`allocate_like`.

    Parameters
    ----------
    input_column : Column
        The column to use as a template for the output.
    policy : mask_allocation_policy
        Controls whether the output column has a valid mask.
    size : int, optional
        The number of elements to allocate in the output column. If not
        specified, the size of the input column is used.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        A column with the same type and size as input.
    """

    cdef unique_ptr[column] c_result
    cdef size_type c_size = size if size is not None else input_column.size()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input_column = input_column.view()
    with nogil:
        c_result = cpp_copying.allocate_like(
                c_input_column,
                c_size,
                policy,
                _cs,
                mr.get_mr()
            )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column copy_range_in_place(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
    object stream=None
):
    """Copy a range of elements from input_column to target_column.

    The target_column is overwritten in place.

    For details on the implementation, see :cpp:func:`copy_range_in_place`.

    Parameters
    ----------
    input_column : Column
        The column from which to copy elements.
    target_column : Column
        The column into which to copy elements.
    input_begin : int
        The index of the first element in input_column to copy.
    input_end : int
        The index of the last element in input_column to copy.
    target_begin : int
        The index of the first element in target_column to overwrite.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Raises
    ------
    TypeError
        If the operation is attempted on non-fixed width types since those would require
        memory reallocations, or if the input and target columns have different types.
    IndexError
        If the indices accessed by the ranges implied by input_begin, input_end, and
        target_begin are out of bounds.
    ValueError
        If source has null values and target is not nullable.
    """

    cdef mutable_column_view target_view = target_column.mutable_view()
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()

    cdef column_view c_input_column = input_column.view()
    with nogil:
        cpp_copying.copy_range_in_place(
            c_input_column,
            target_view,
            input_begin,
            input_end,
            target_begin,
            _cs
        )
    target_column.set_null_count(target_view.null_count())


cpdef Column copy_range(
    Column input_column,
    Column target_column,
    size_type input_begin,
    size_type input_end,
    size_type target_begin,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Copy a range of elements from input_column to target_column.

    For details on the implementation, see :cpp:func:`copy_range`.

    Parameters
    ----------
    input_column : Column
        The column from which to copy elements.
    target_column : Column
        The column into which to copy elements.
    input_begin : int
        The index of the first element in input_column to copy.
    input_end : int
        The index of the last element in input_column to copy.
    target_begin : int
        The index of the first element in target_column to overwrite.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        A copy of target_column with the specified range overwritten.

    Raises
    ------
    IndexError
        If the indices accessed by the ranges implied by input_begin, input_end, and
        target_begin are out of bounds.
    TypeError
        If target and source have different types.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input_column = input_column.view()
    cdef column_view c_target_column = target_column.view()
    with nogil:
        c_result = cpp_copying.copy_range(
            c_input_column,
            c_target_column,
            input_begin,
            input_end,
            target_begin,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column shift(
    Column input,
    size_type offset,
    Scalar fill_value,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Shift the elements of input by offset.

    For details on the implementation, see :cpp:func:`shift`.

    Parameters
    ----------
    input : Column
        The column to shift.
    offset : int
        The number of elements to shift by.
    fill_values : Scalar
        The value to use for elements that are shifted in from outside the
        bounds of the input column.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        A copy of input shifted by offset.

    Raises
    ------
    TypeError
        If the fill_value is not of the same type as input, or if the input type is not
        of fixed width or string type.
    """
    cdef unique_ptr[column] c_result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_copying.shift(
                c_input,
                offset,
                dereference(fill_value.c_obj),
                _cs,
                mr.get_mr()
            )
    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef list slice(ColumnOrTable input, list indices, object stream=None):
    """Slice input according to indices.

    For details on the implementation, see :cpp:func:`slice`.

    Parameters
    ----------
    input_column : Union[Column, Table]
        The column or table to slice.
    indices : List[int]
        The indices to select from input.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    List[Union[Column, Table]]
        The result of slicing ``input``.

    Raises
    ------
    ValueError
        If indices size is not even or the values in any pair of lower/upper bounds are
        strictly decreasing.
    IndexError
        When any of the indices don't belong to the range ``[0, input_column.size())``.
    """
    cdef vector[size_type] c_indices = indices
    cdef vector[column_view] c_col_result
    cdef vector[table_view] c_tbl_result
    cdef int i
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()

    cdef column_view c_input_column
    cdef table_view c_input_table

    if ColumnOrTable is Column:
        c_input_column = input.view()
        with nogil:
            c_col_result = cpp_copying.slice(c_input_column, c_indices, _cs)

        return [
            Column.from_column_view(c_col_result[i], input)
            for i in range(c_col_result.size())
        ]
    else:
        c_input_table = input.view()
        with nogil:
            c_tbl_result = cpp_copying.slice(c_input_table, c_indices, _cs)

        return [
            Table.from_table_view(c_tbl_result[i], input)
            for i in range(c_tbl_result.size())
        ]


cpdef list split(ColumnOrTable input, list splits, object stream=None):
    """Split input into multiple.

    For details on the implementation, see :cpp:func:`split`.

    Parameters
    ----------
    input : Union[Column, Table]
        The column to split.
    splits : List[int]
        The indices at which to split the column.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    List[Union[Column, Table]]
        The result of splitting input.
    """
    cdef vector[size_type] c_splits = splits
    cdef vector[column_view] c_col_result
    cdef vector[table_view] c_tbl_result
    cdef int i
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()

    cdef column_view c_input_column
    cdef table_view c_input_table

    if ColumnOrTable is Column:
        c_input_column = input.view()
        with nogil:
            c_col_result = cpp_copying.split(c_input_column, c_splits, _cs)

        return [
            Column.from_column_view(c_col_result[i], input)
            for i in range(c_col_result.size())
        ]
    else:
        c_input_table = input.view()
        with nogil:
            c_tbl_result = cpp_copying.split(c_input_table, c_splits, _cs)

        return [
            Table.from_table_view(c_tbl_result[i], input)
            for i in range(c_tbl_result.size())
        ]


cpdef Column copy_if_else(
    LeftCopyIfElseOperand lhs,
    RightCopyIfElseOperand rhs,
    Column boolean_mask,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Copy elements from lhs or rhs into a new column according to boolean_mask.

    For details on the implementation, see :cpp:func:`copy_if_else`.

    Parameters
    ----------
    lhs : Union[Column, Scalar]
        The column or scalar to copy from if the corresponding element in
        boolean_mask is True.
    rhs : Union[Column, Scalar]
        The column or scalar to copy from if the corresponding element in
        boolean_mask is False.
    boolean_mask : Column
        The boolean mask to use to select elements from lhs and rhs.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Column
        The result of copying elements from lhs and rhs according to boolean_mask.

    Raises
    ------
    TypeError
        If lhs and rhs are not of the same type or if the boolean mask is not of type
        bool.
    ValueError
        If boolean mask is not of the same length as lhs and rhs (whichever are
        columns), or if lhs and rhs are not of the same length (if both are columns).
    """
    cdef unique_ptr[column] result
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef column_view c_lhs_column
    cdef column_view c_rhs_column
    cdef column_view c_boolean_mask

    mr = _get_memory_resource(mr)

    if LeftCopyIfElseOperand is Column and RightCopyIfElseOperand is Column:
        c_lhs_column = lhs.view()
        c_rhs_column = rhs.view()
        c_boolean_mask = boolean_mask.view()
        with nogil:
            result = cpp_copying.copy_if_else(
                c_lhs_column,
                c_rhs_column,
                c_boolean_mask,
                _cs,
                mr.get_mr()
            )
    elif LeftCopyIfElseOperand is Column and RightCopyIfElseOperand is Scalar:
        c_lhs_column = lhs.view()
        c_boolean_mask = boolean_mask.view()
        with nogil:
            result = cpp_copying.copy_if_else(
                c_lhs_column,
                dereference(rhs.c_obj),
                c_boolean_mask,
                _cs,
                mr.get_mr()
            )
    elif LeftCopyIfElseOperand is Scalar and RightCopyIfElseOperand is Column:
        c_rhs_column = rhs.view()
        c_boolean_mask = boolean_mask.view()
        with nogil:
            result = cpp_copying.copy_if_else(
                dereference(lhs.c_obj),
                c_rhs_column,
                c_boolean_mask,
                _cs,
                mr.get_mr()
            )
    else:
        c_boolean_mask = boolean_mask.view()
        with nogil:
            result = cpp_copying.copy_if_else(
                dereference(lhs.c_obj),
                dereference(rhs.c_obj),
                c_boolean_mask,
                _cs,
                mr.get_mr()
            )

    return Column.from_libcudf(move(result), _stream, mr)


cpdef Table boolean_mask_scatter(
    TableOrListOfScalars input,
    Table target,
    Column boolean_mask,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Scatter rows from input into target according to boolean_mask.

    If source is a table, it specifies rows to scatter. If source is a list,
    each scalar is scattered into the corresponding column in the ``target_table``.

    For details on the implementation, see :cpp:func:`boolean_mask_scatter`.

    Parameters
    ----------
    input : Union[Table, List[Scalar]]
        The table object from which to pull data.
    target : Table
        The table object into which to scatter data.
    boolean_mask : Column
        A mapping from rows in input to rows in target.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Table
        The result of the scatter

    Raises
    ------
    ValueError
        If input.num_columns() != target.num_columns(), boolean_mask.size() !=
        target.num_rows(), or if input is a Table and the number of `true` in
        `boolean_mask` > input.num_rows().
    TypeError
        If any input type does not match the corresponding target column's type, or
        if boolean_mask.type() is not bool.
    """
    cdef unique_ptr[table] result
    cdef vector[reference_wrapper[const scalar]] source_scalars
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    cdef table_view c_input_table
    cdef table_view c_target
    cdef column_view c_boolean_mask

    mr = _get_memory_resource(mr)

    if TableOrListOfScalars is Table:
        c_input_table = input.view()
        c_target = target.view()
        c_boolean_mask = boolean_mask.view()
        with nogil:
            result = cpp_copying.boolean_mask_scatter(
                c_input_table,
                c_target,
                c_boolean_mask,
                _cs,
                mr.get_mr()
            )
    else:
        source_scalars = _as_vector(input)
        c_target = target.view()
        c_boolean_mask = boolean_mask.view()
        with nogil:
            result = cpp_copying.boolean_mask_scatter(
                source_scalars,
                c_target,
                c_boolean_mask,
                _cs,
                mr.get_mr()
            )

    return Table.from_libcudf(move(result), _stream, mr)


cpdef Scalar get_element(
    Column input_column,
    size_type index,
    object stream=None,
    DeviceMemoryResource mr=None
):
    """Get the element at index from input_column.

    For details on the implementation, see :cpp:func:`get_element`.

    Parameters
    ----------
    input_column : Column
        The column from which to get the element.
    index : int
        The index of the element to get.
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    pylibcudf.Scalar
        The element at index from input_column.

    Raises
    ------
    IndexError
        If index is out of bounds.
    """
    cdef unique_ptr[scalar] c_output
    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input_column = input_column.view()
    with nogil:
        c_output = cpp_copying.get_element(
            c_input_column, index, _cs, mr.get_mr()
        )

    return Scalar.from_libcudf(move(c_output))

OutOfBoundsPolicy.__str__ = OutOfBoundsPolicy.__repr__
MaskAllocationPolicy.__str__ = MaskAllocationPolicy.__repr__
SampleWithReplacement.__str__ = SampleWithReplacement.__repr__
