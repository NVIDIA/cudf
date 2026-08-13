# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference

from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.span cimport span
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move, pair

from pylibcudf.libcudf cimport transform as cpp_transform
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport (
    bitmask_type,
    size_type,
    udf_source_type,
)

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .gpumemoryview cimport gpumemoryview
from .types cimport DataType, null_aware, output_nullability
from .utils cimport _get_stream, _get_memory_resource
from cuda.bindings.cyruntime cimport cudaStream_t

ctypedef const cpp_transform.transform_input const_transform_input
ctypedef const cpp_transform.transform_output const_transform_output

__all__ = [
    "bools_to_mask",
    "column_nans_to_nulls",
    "compute_column",
    "compute_column_jit",
    "encode",
    "mask_to_bools",
    "nans_to_nulls",
    "one_hot_encode",
    "transform",
]

cpdef tuple[gpumemoryview, int] nans_to_nulls(
    Column input,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Create a null mask preserving existing nulls and converting nans to null.

    For details, see :cpp:func:`nans_to_nulls`.

    Parameters
    ----------
    input : Column
        Column to produce new mask from.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned mask's device memory.

    Returns
    -------
    Two-tuple of a gpumemoryview wrapping the null mask and the new null count.
    """
    cdef pair[unique_ptr[device_buffer], size_type] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_transform.nans_to_nulls(
            c_input, _cs, mr.get_mr()
        )

    return (
        gpumemoryview(
            DeviceBuffer.c_from_unique_ptr(move(c_result.first), _stream, mr)
        ),
        c_result.second
    )


cpdef Column column_nans_to_nulls(
    Column input,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Create a column with nans converted to nulls.

    For details, see :cpp:func:`column_nans_to_nulls`.

    Parameters
    ----------
    input : Column
        Column to convert nans to nulls.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        New column with nans converted to nulls.
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_transform.column_nans_to_nulls(
            c_input, _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column compute_column(
    Table input, Expression expr, object stream=None, DeviceMemoryResource mr=None
):
    """Create a column by evaluating an expression on a table.

    For details see :cpp:func:`compute_column`.

    Parameters
    ----------
    input : Table
        Table used for expression evaluation
    expr : Expression
        Expression to evaluate
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column of the evaluated expression
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_transform.compute_column(
            c_input, dereference(expr.c_obj.get()), _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column compute_column_jit(
    Table input, Expression expr, object stream=None, DeviceMemoryResource mr=None
):
    """
    Create a column by evaluating an expression on a table
    using a JIT-compiled kernel.

    For details see :cpp:func:`compute_column_jit`.

    Parameters
    ----------
    input : Table
        Table used for expression evaluation
    expr : Expression
        Expression to evaluate
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column of the evaluated expression
    """
    cdef unique_ptr[column] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_transform.compute_column_jit(
            c_input, dereference(expr.c_obj.get()), _cs, mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef tuple[gpumemoryview, int] bools_to_mask(
    Column input,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Create a bitmask from a column of boolean elements

    Parameters
    ----------
    input : Column
        Column to produce new mask from.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned mask's device memory.

    Returns
    -------
    tuple[gpumemoryview, int]
        Two-tuple of a gpumemoryview wrapping the bitmask and the null count.
    """
    cdef pair[unique_ptr[device_buffer], size_type] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    with nogil:
        c_result = cpp_transform.bools_to_mask(
            c_input, _cs, mr.get_mr()
        )

    return (
        gpumemoryview(
            DeviceBuffer.c_from_unique_ptr(move(c_result.first), _stream, mr)
        ),
        c_result.second
    )


cpdef Column mask_to_bools(
    Py_ssize_t bitmask,
    int begin_bit,
    int end_bit,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Creates a boolean column from given bitmask.

    Parameters
    ----------
    bitmask : int
        Pointer to the bitmask which needs to be converted
    begin_bit : int
        Position of the bit from which the conversion should start
    end_bit : int
        Position of the bit before which the conversion should stop
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        Boolean column of the bitmask from [begin_bit, end_bit]
    """
    cdef unique_ptr[column] c_result
    cdef bitmask_type * bitmask_ptr = <bitmask_type*>bitmask

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_transform.mask_to_bools(
            bitmask_ptr,
            begin_bit,
            end_bit,
            _cs,
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), _stream, mr)


cpdef Column transform(
    inputs,
    str transform_udf,
    DataType output_type,
    bool is_ptx,
    null_aware is_null_aware,
    output_nullability null_policy,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Create a new column by applying a transform function against
       multiple input columns.

    Parameters
    ----------
    inputs : Sequence[Column]
        Columns to transform.
    transform_udf : str
        The PTX/CUDA string of the transform function to apply.
    output_type : DataType
        The output type that is compatible with the output type in the unary_udf.
    is_ptx : bool
        If `True`, the UDF is treated as PTX code.
        If `False`, the UDF is treated as CUDA code.
    is_null_aware: NullAware
        If `NO`, the UDF gets non-nullable parameters
        If `YES`, the UDF gets nullable parameters
    null_policy: OutputNullability
        If `PRESERVE`, null-masks are produced if necessary.
        If `ALL_VALID`, null-masks are not produced.
        `ALL_VALID` has undefined behavior if the UDF can produce nulls.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    Column
        The transformed column having the UDF applied to each element.
    """
    cdef vector[cpp_transform.transform_input] c_inputs
    cdef unique_ptr[column] c_result
    cdef unique_ptr[table] c_table_result
    cdef vector[unique_ptr[column]] c_columns
    cdef vector[unique_ptr[column]] string_offsets
    cdef vector[cpp_transform.transform_output] c_outputs
    cdef cpp_transform.transform_output c_output
    cdef unique_ptr[cpp_transform.scalar_column_view] c_scalar_input
    cdef string c_transform_udf = transform_udf.encode()
    cdef udf_source_type source_type = (
        udf_source_type.PTX if is_ptx else udf_source_type.CUDA
    )
    cdef null_aware c_is_null_aware = is_null_aware
    cdef output_nullability c_null_policy = null_policy
    cdef optional[void *] user_data
    cdef optional[size_type] row_size
    cdef column_view input_view
    cdef size_type smallest_size
    cdef size_type largest_size
    cdef size_type base_size

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    if inputs:
        sizes = [(<Column?>input).view().size() for input in inputs]
        smallest_size = min(sizes)
        largest_size = max(sizes)

        base_size = 0 if largest_size == 1 and smallest_size == 0 else largest_size
        row_size = base_size

    for input in inputs:
        input_view = (<Column?>input).view()
        if input_view.size() == 1 and input_view.size() != base_size:
            c_scalar_input.reset(
                new cpp_transform.scalar_column_view(input_view)
            )
            c_inputs.push_back(
                cpp_transform.transform_input(dereference(c_scalar_input))
            )
        else:
            c_inputs.push_back(cpp_transform.transform_input(input_view))

    c_output.type = output_type.c_obj
    c_output.nullability = c_null_policy
    c_outputs.push_back(c_output)

    with nogil:
        c_table_result = cpp_transform.transform(
            c_transform_udf,
            source_type,
            c_is_null_aware,
            user_data,
            span[const_transform_input](c_inputs.data(), c_inputs.size()),
            span[const_transform_output](c_outputs.data(), c_outputs.size()),
            move(string_offsets),
            row_size,
            _cs,
            mr.get_mr()
        )
        c_columns = dereference(c_table_result).release()
        c_result = move(c_columns[0])

    return Column.from_libcudf(move(c_result), _stream, mr)

cpdef tuple[Table, Column] encode(
    Table input, object stream=None, DeviceMemoryResource mr=None
):
    """Encode the rows of the given table as integers.

    Parameters
    ----------
    input : Table
        Table containing values to be encoded
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned objects' device memory.

    Returns
    -------
    tuple[Table, Column]
        The distinct row of the input table in sorted order,
        and a column of integer indices representing the encoded rows.
    """
    cdef pair[unique_ptr[table], unique_ptr[column]] c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef table_view c_input = input.view()
    with nogil:
        c_result = cpp_transform.encode(c_input, _cs, mr.get_mr())

    return (
        Table.from_libcudf(move(c_result.first), _stream, mr),
        Column.from_libcudf(move(c_result.second), _stream, mr)
    )

cpdef Table one_hot_encode(
    Column input,
    Column categories,
    object stream=None,
    DeviceMemoryResource mr=None,
):
    """Encodes `input` by generating a new column
    for each value in `categories` indicating the presence
    of that value in `input`.

    Parameters
    ----------
    input : Column
        Column containing values to be encoded.
    categories : Column
        Column containing categories
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    Column
        A table of the encoded values.
    """
    cdef pair[unique_ptr[column], table_view] c_result
    cdef Table owner_table

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().value()
    mr = _get_memory_resource(mr)

    cdef column_view c_input = input.view()
    cdef column_view c_categories = categories.view()
    with nogil:
        c_result = cpp_transform.one_hot_encode(
            c_input,
            c_categories,
            _cs,
            mr.get_mr()
        )

    owner_table = Table(
        [Column.from_libcudf(move(c_result.first), _stream, mr)]
        * c_result.second.num_columns()
    )

    return Table.from_table_view(c_result.second, owner_table)
