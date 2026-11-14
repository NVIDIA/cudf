# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.types cimport bitmask_type
from cython.operator import dereference

from libc.stddef cimport size_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.optional cimport optional
from libcpp.utility cimport move
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf cimport asof_join as cpp_asof_join
from pylibcudf.libcudf.asof_join cimport (
    asof_join_exact_matches,
    asof_join_strategy,
)
from pylibcudf.libcudf cimport join as cpp_join
from pylibcudf.libcudf cimport null_mask as cpp_null_mask
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport mask_state, null_equality
from pylibcudf.libcudf.utilities.device_buffer cimport device_buffer
from pylibcudf.experimental._join_streams cimport join_streams
from rmm.librmm.cuda_stream_ref cimport stream_ref
from rmm.librmm.memory_resource cimport device_async_resource_ref

from pylibcudf.libcudf.asof_join import (
    asof_join_exact_matches as AsofJoinExactMatches,  # no-cython-lint
    asof_join_strategy as AsofJoinStrategy,  # no-cython-lint
)

from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

from .column cimport Column
from .expressions cimport Expression
from .table cimport Table
from .utils cimport _get_stream, _get_memory_resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylibcudf.typing import CudaStreamLike

from cuda.bindings.cyruntime cimport cudaStream_t

__all__ = [
    "AsofJoin",
    "AsofJoinExactMatches",
    "AsofJoinStrategy",
    "conditional_full_join",
    "conditional_inner_join",
    "conditional_left_anti_join",
    "conditional_left_join",
    "conditional_left_semi_join",
    "cross_join",
    "FilteredJoin",
    "full_join",
    "inner_join",
    "left_anti_join",
    "left_join",
    "left_semi_join",
    "mixed_full_join",
    "mixed_inner_join",
    "mixed_left_anti_join",
    "mixed_left_join",
    "mixed_left_semi_join",
]


cdef extern from * nogil:
    """
    #include <cudf/join/asof_join.hpp>
    #include <cudf/column/column.hpp>
    #include <cudf/column/column_view.hpp>
    #include <cudf/types.hpp>
    #include <cuda/stream>
    #include <memory>

    std::unique_ptr<cudf::column> asof_join_indices(
      cudf::experimental::asof_join const& joiner,
      cudf::table_view const& left_by, cudf::column_view const& left_on,
      cudf::experimental::asof_join_strategy strategy,
      cudf::experimental::asof_join_exact_matches exact_matches,
      cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    {
      auto indices = joiner.join(
        left_by, left_on, strategy, exact_matches, stream, mr);
      // cudf::column stores an rmm::device_buffer, so copy the returned CUDA buffer.
      cudf::column_view view{cudf::data_type{cudf::type_id::INT32},
                             left_on.size(), indices.data(), nullptr, 0};
      return std::make_unique<cudf::column>(view, stream, mr);
    }
    """
    cdef unique_ptr[column] asof_join_indices(
        const cpp_asof_join.asof_join& joiner,
        const table_view& left_by,
        const column_view& left_on,
        asof_join_strategy strategy,
        asof_join_exact_matches exact_matches,
        stream_ref stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler


cdef class AsofJoin:
    """As-of join with reusable right-side preprocessing.

    The right keys must be sorted ascending by ``(right_by..., right_on)``
    with nulls first. This precondition is not checked. Left rows may be
    unordered. The right input is retained and must not be modified while in
    use.
    """

    def __cinit__(
        self,
        Table right_by,
        Column right_on,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None,
    ) -> None:
        """Prepare the right-side keys.

        Parameters
        ----------
        right_by : Table
            Equality keys. An empty table selects an ungrouped join.
        right_on : Column
            Ordered key.
        stream : Stream, optional
            Construction stream. Defaults to the configured default stream.
        mr : DeviceMemoryResource, optional
            Memory resource for preprocessing. Defaults to the current resource.
        """
        self._right_by = right_by
        self._right_on = right_on
        self._stream = _get_stream(stream)
        self._mr = _get_memory_resource(mr)
        cdef table_view c_right_by = right_by.view()
        cdef column_view c_right_on = right_on.view()
        cdef stream_ref _cs = self._stream.view()
        with nogil:
            self.c_obj.reset(
                new cpp_asof_join.asof_join(
                    c_right_by, c_right_on, _cs, self._mr.get_mr()
                )
            )

    def __dealloc__(self):
        if self._stream is not None:
            self._stream.synchronize()
        self.c_obj.reset()

    def join(
        self,
        Table left_by,
        Column left_on,
        asof_join_strategy strategy=asof_join_strategy.BACKWARD,
        asof_join_exact_matches exact_matches=asof_join_exact_matches.YES,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None,
    ) -> Column:
        """Return right-row indices in left-row order.

        Unmatched rows contain ``JoinNoMatch`` and must be gathered with
        ``OutOfBoundsPolicy.NULLIFY``.

        Parameters
        ----------
        left_by : Table
            Equality keys. An empty table selects an ungrouped join.
        left_on : Column
            Ordered key.
        strategy : AsofJoinStrategy, optional
            Match direction. Defaults to ``BACKWARD``.
        exact_matches : AsofJoinExactMatches, optional
            Whether equal ordered keys may match. Defaults to ``YES``.
        stream : Stream, optional
            Probe stream. Defaults to the construction stream.
        mr : DeviceMemoryResource, optional
            Memory resource for the result. Defaults to the current resource.

        Returns
        -------
        Column
            Right-row indices in left-row order.
        """
        cdef Stream _stream = self._stream if stream is None else _get_stream(stream)
        cdef stream_ref _cs = _stream.view()
        mr = _get_memory_resource(mr)
        cdef table_view c_left_by = left_by.view()
        cdef column_view c_left_on = left_on.view()
        cdef unique_ptr[column] result
        if _stream is not self._stream:
            join_streams([self._stream], _stream)
        try:
            with nogil:
                result = asof_join_indices(
                    dereference(self.c_obj), c_left_by, c_left_on,
                    strategy, exact_matches, _cs, mr.get_mr()
                )
        finally:
            if _stream is not self._stream:
                join_streams([_stream], self._stream)
        return Column.from_libcudf(move(result), _stream, mr)


cdef Column _column_from_gather_map(
    cpp_join.gather_map_type gather_map, object stream, DeviceMemoryResource mr
):
    # helper to convert a gather map to a Column
    cdef Stream _stream = _get_stream(stream)
    cdef unique_ptr[device_buffer[bitmask_type]] mask = (
        cpp_null_mask.create_null_mask_unique_ptr(
            0,
            mask_state.UNALLOCATED,
            _stream.view().get(),
            mr.get_mr(),
        )
    )
    return Column.from_libcudf(
        move(
            make_unique[column](
                move(dereference(gather_map.get())),
                move(dereference(mask)),
                0
            )
        ), _stream, mr
    )


cpdef tuple[Column, Column] inner_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform an inner join between two tables.

    For details, see :cpp:func:`inner_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    with nogil:
        c_result = cpp_join.inner_join(
            c_left_keys,
            c_right_keys,
            nulls_equal,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef tuple[Column, Column] left_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a left join between two tables.

    For details, see :cpp:func:`left_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    with nogil:
        c_result = cpp_join.left_join(
            c_left_keys,
            c_right_keys,
            nulls_equal,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef tuple[Column, Column] full_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a full join between two tables.

    For details, see :cpp:func:`full_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    with nogil:
        c_result = cpp_join.full_join(
            c_left_keys,
            c_right_keys,
            nulls_equal,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef Column left_semi_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a left semi join between two tables.

    For details, see :cpp:class:`cudf::filtered_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef unique_ptr[cpp_join.filtered_join] join_obj

    cdef table_view c_right_keys = right_keys.view()
    cdef table_view c_left_keys = left_keys.view()
    with nogil:
        join_obj.reset(
            new cpp_join.filtered_join(
                c_right_keys,
                nulls_equal,
                _cs
            )
        )
        c_result = join_obj.get()[0].semi_join(
            c_left_keys,
            _cs,
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), _stream, mr)


cpdef Column left_anti_join(
    Table left_keys,
    Table right_keys,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a left anti join between two tables.

    For details, see :cpp:class:`cudf::filtered_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to join.
    right_keys : Table
        The right table to join.
    nulls_equal : NullEquality
        Should nulls compare equal?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef unique_ptr[cpp_join.filtered_join] join_obj

    cdef table_view c_right_keys = right_keys.view()
    cdef table_view c_left_keys = left_keys.view()
    with nogil:
        join_obj.reset(
            new cpp_join.filtered_join(
                c_right_keys,
                nulls_equal,
                _cs
            )
        )
        c_result = join_obj.get()[0].anti_join(
            c_left_keys,
            _cs,
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), _stream, mr)


cpdef Table cross_join(
    Table left, Table right, object stream: CudaStreamLike | None = None, DeviceMemoryResource mr=None
):
    """Perform a cross join on two tables.

    For details see :cpp:func:`cross_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right: Table
        The right table to join.
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned table's device memory.

    Returns
    -------
    Table
        The result of cross joining the two inputs.
    """
    cdef unique_ptr[table] result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left = left.view()
    cdef table_view c_right = right.view()
    with nogil:
        result = cpp_join.cross_join(
            c_left, c_right, _cs, mr.get_mr()
        )
    return Table.from_libcudf(move(result), _stream, mr)


cpdef tuple[Column, Column] conditional_inner_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional inner join between two tables.

    For details, see :cpp:func:`conditional_inner_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef optional[size_t] output_size

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left = left.view()
    cdef table_view c_right = right.view()
    with nogil:
        c_result = cpp_join.conditional_inner_join(
            c_left,
            c_right,
            dereference(binary_predicate.c_obj.get()),
            output_size,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef tuple[Column, Column] conditional_left_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional left join between two tables.

    For details, see :cpp:func:`conditional_left_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef optional[size_t] output_size

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left = left.view()
    cdef table_view c_right = right.view()
    with nogil:
        c_result = cpp_join.conditional_left_join(
            c_left,
            c_right,
            dereference(binary_predicate.c_obj.get()),
            output_size,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef tuple[Column, Column] conditional_full_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional full join between two tables.

    For details, see :cpp:func:`conditional_full_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left = left.view()
    cdef table_view c_right = right.view()
    with nogil:
        c_result = cpp_join.conditional_full_join(
            c_left,
            c_right,
            dereference(binary_predicate.c_obj.get()),
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef Column conditional_left_semi_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional left semi join between two tables.

    For details, see :cpp:func:`conditional_left_semi_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result
    cdef optional[size_t] output_size

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left = left.view()
    cdef table_view c_right = right.view()
    with nogil:
        c_result = cpp_join.conditional_left_semi_join(
            c_left,
            c_right,
            dereference(binary_predicate.c_obj.get()),
            output_size,
            _cs,
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), _stream, mr)


cpdef Column conditional_left_anti_join(
    Table left,
    Table right,
    Expression binary_predicate,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a conditional left anti join between two tables.

    For details, see :cpp:func:`conditional_left_anti_join`.

    Parameters
    ----------
    left : Table
        The left table to join.
    right : Table
        The right table to join.
    binary_predicate : Expression
        Condition to join on.

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result
    cdef optional[size_t] output_size

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left = left.view()
    cdef table_view c_right = right.view()
    with nogil:
        c_result = cpp_join.conditional_left_anti_join(
            c_left,
            c_right,
            dereference(binary_predicate.c_obj.get()),
            output_size,
            _cs,
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), _stream, mr)


cpdef tuple[Column, Column] mixed_inner_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed inner join between two tables.

    For details, see :cpp:func:`mixed_inner_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef cpp_join.output_size_data_type empty_optional

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    cdef table_view c_left_conditional = left_conditional.view()
    cdef table_view c_right_conditional = right_conditional.view()
    with nogil:
        c_result = cpp_join.mixed_inner_join(
            c_left_keys,
            c_right_keys,
            c_left_conditional,
            c_right_conditional,
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            empty_optional,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef tuple[Column, Column] mixed_left_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed left join between two tables.

    For details, see :cpp:func:`mixed_left_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef cpp_join.output_size_data_type empty_optional

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    cdef table_view c_left_conditional = left_conditional.view()
    cdef table_view c_right_conditional = right_conditional.view()
    with nogil:
        c_result = cpp_join.mixed_left_join(
            c_left_keys,
            c_right_keys,
            c_left_conditional,
            c_right_conditional,
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            empty_optional,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef tuple[Column, Column] mixed_full_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed full join between two tables.

    For details, see :cpp:func:`mixed_full_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Tuple[Column, Column]
        A tuple containing the row indices from the left and right tables after the
        join.
    """
    cdef cpp_join.gather_map_pair_type c_result
    cdef cpp_join.output_size_data_type empty_optional

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    cdef table_view c_left_conditional = left_conditional.view()
    cdef table_view c_right_conditional = right_conditional.view()
    with nogil:
        c_result = cpp_join.mixed_full_join(
            c_left_keys,
            c_right_keys,
            c_left_conditional,
            c_right_conditional,
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            empty_optional,
            _cs,
            mr.get_mr()
        )
    return (
        _column_from_gather_map(move(c_result.first), _stream, mr),
        _column_from_gather_map(move(c_result.second), _stream, mr),
    )


cpdef Column mixed_left_semi_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed left semi join between two tables.

    For details, see :cpp:func:`mixed_left_semi_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    cdef table_view c_left_conditional = left_conditional.view()
    cdef table_view c_right_conditional = right_conditional.view()
    with nogil:
        c_result = cpp_join.mixed_left_semi_join(
            c_left_keys,
            c_right_keys,
            c_left_conditional,
            c_right_conditional,
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            _cs,
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), _stream, mr)


cpdef Column mixed_left_anti_join(
    Table left_keys,
    Table right_keys,
    Table left_conditional,
    Table right_conditional,
    Expression binary_predicate,
    null_equality nulls_equal,
    object stream: CudaStreamLike | None = None,
    DeviceMemoryResource mr=None,
):
    """Perform a mixed left anti join between two tables.

    For details, see :cpp:func:`mixed_left_anti_join`.

    Parameters
    ----------
    left_keys : Table
        The left table to use for the equality join.
    right_keys : Table
        The right table to use for the equality join.
    left_conditional : Table
        The left table to use for the conditional join.
    right_conditional : Table
        The right table to use for the conditional join.
    binary_predicate : Expression
        Condition to join on.
    nulls_equal : NullEquality
        Should nulls compare equal in the equality join?

    Returns
    -------
    Column
        A column containing the row indices from the left table after the join.
    """
    cdef cpp_join.gather_map_type c_result

    cdef Stream _stream = _get_stream(stream)
    cdef cudaStream_t _cs = _stream.view().get()
    mr = _get_memory_resource(mr)

    cdef table_view c_left_keys = left_keys.view()
    cdef table_view c_right_keys = right_keys.view()
    cdef table_view c_left_conditional = left_conditional.view()
    cdef table_view c_right_conditional = right_conditional.view()
    with nogil:
        c_result = cpp_join.mixed_left_anti_join(
            c_left_keys,
            c_right_keys,
            c_left_conditional,
            c_right_conditional,
            dereference(binary_predicate.c_obj.get()),
            nulls_equal,
            _cs,
            mr.get_mr()
        )
    return _column_from_gather_map(move(c_result), _stream, mr)


cdef class FilteredJoin:
    """
    Filtered hash join that builds a hash table from the right (filter) table
    on creation and probes results in subsequent join member functions.

    The right table is used as the filter applied to multiple left tables in
    subsequent ``semi_join`` or ``anti_join`` calls. For use cases where the
    left table should be reused with multiple right tables, use ``MarkJoin``
    instead.

    For details, see :cpp:class:`cudf::filtered_join`.
    """

    def __cinit__(
        self,
        Table right,
        null_equality compare_nulls=null_equality.EQUAL,
        double load_factor=0.5,
        object stream: CudaStreamLike | None = None,
    ) -> None:
        """
        Construct a filtered hash join object for subsequent probe calls.

        Parameters
        ----------
        right : Table
            The right (filter) table used to build the hash table.
        compare_nulls : NullEquality
            Controls whether null join-key values should match or not.
        load_factor : float, optional
            The desired ratio of filled slots to total slots in the hash table,
            must be in range (0,1]. Defaults to 0.5.
        stream : Stream, optional
            CUDA stream used for device memory operations and kernel launches.
        """
        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().get()

        cdef table_view c_right = right.view()
        with nogil:
            self.c_obj.reset(
                new cpp_join.filtered_join(
                    c_right,
                    compare_nulls,
                    load_factor,
                    _cs
                )
            )

    def semi_join(
        self,
        Table left,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None,
    ) -> Column:
        """
        Returns a column of row indices corresponding to a semi-join
        between the right (filter) table and left table.

        For details, see :cpp:func:`cudf::filtered_join::semi_join`.

        Parameters
        ----------
        left : Table
            The left table.
        stream : Stream, optional
            CUDA stream used for device memory operations and kernel launches.
        mr : DeviceMemoryResource, optional
            Device memory resource used to allocate the returned column's device memory.

        Returns
        -------
        Column
            A column containing the row indices from the left table after the join.
        """
        cdef cpp_join.gather_map_type c_result

        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().get()
        mr = _get_memory_resource(mr)

        cdef table_view c_left = left.view()
        with nogil:
            c_result = self.c_obj.get()[0].semi_join(
                c_left,
                _cs,
                mr.get_mr()
            )
        return _column_from_gather_map(move(c_result), _stream, mr)

    def anti_join(
        self,
        Table left,
        object stream: CudaStreamLike | None = None,
        DeviceMemoryResource mr=None,
    ) -> Column:
        """
        Returns a column of row indices corresponding to an anti-join
        between the right (filter) table and left table.

        For details, see :cpp:func:`cudf::filtered_join::anti_join`.

        Parameters
        ----------
        left : Table
            The left table.
        stream : Stream, optional
            CUDA stream used for device memory operations and kernel launches.
        mr : DeviceMemoryResource, optional
            Device memory resource used to allocate the returned column's device memory.

        Returns
        -------
        Column
            A column containing the row indices from the left table after the join.
        """
        cdef cpp_join.gather_map_type c_result

        cdef Stream _stream = _get_stream(stream)
        cdef cudaStream_t _cs = _stream.view().get()
        mr = _get_memory_resource(mr)

        cdef table_view c_left = left.view()
        with nogil:
            c_result = self.c_obj.get()[0].anti_join(
                c_left,
                _cs,
                mr.get_mr()
            )
        return _column_from_gather_map(move(c_result), _stream, mr)
