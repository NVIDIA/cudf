# Copyright (c) 2024-2025, NVIDIA CORPORATION.
# Implementation of decimal division with scale preservation

from enum import IntEnum

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from .column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.decimal.decimal_ops cimport (
    decimal_rounding_mode as cpp_decimal_rounding_mode,
    divide_decimal as cpp_divide_decimal
)
from pylibcudf.types cimport type_id
from pylibcudf.scalar cimport Scalar
from pylibcudf.libcudf.scalar.scalar cimport scalar as cpp_scalar
from cython.operator cimport dereference

__all__ = [
    "DecimalRoundingMode",
    "divide_decimal",
    "divide_decimal_column_scalar",
    "divide_decimal_scalar_column"
]


class DecimalRoundingMode(IntEnum):
    """
    Rounding modes for decimal division.

    Attributes
    ----------
    HALF_UP : int
        Round half away from zero (0.5 rounds to 1, -0.5 rounds to -1)
    HALF_EVEN : int
        Round half to even (banker's rounding)
    """
    HALF_UP = 0
    HALF_EVEN = 1


cpdef Column divide_decimal(
    Column lhs,
    Column rhs,
    rounding_mode=DecimalRoundingMode.HALF_UP
):
    """
    Perform decimal division preserving the dividend's scale.

    This function divides two decimal columns while maintaining the scale
    of the dividend (left operand), similar to Java's BigDecimal.divide()
    with a specified rounding mode.

    Parameters
    ----------
    lhs : Column
        The dividend (left operand) - a decimal column
    rhs : Column
        The divisor (right operand) - a decimal column or scalar
    rounding_mode : DecimalRoundingMode, optional
        The rounding mode to use (default: HALF_UP)

    Returns
    -------
    Column
        Result column with the same scale as the dividend

    Raises
    ------
    TypeError
        If input columns are not decimal types
    ValueError
        If division by zero is attempted

    Examples
    --------
    >>> import pylibcudf
    >>> # Create decimal columns
    >>> lhs = pylibcudf.column_from_decimal_values([1.23, 4.56], scale=-2)
    >>> rhs = pylibcudf.column_from_decimal_values([2.0, 3.0], scale=-1)
    >>> # Divide preserving lhs scale
    >>> result = divide_decimal(lhs, rhs, DecimalRoundingMode.HALF_UP)
    >>> # Result has scale -2 (same as lhs)
    """
    cdef cpp_decimal_rounding_mode cpp_mode

    # Convert Python enum to C++ enum
    if rounding_mode == DecimalRoundingMode.HALF_UP:
        cpp_mode = cpp_decimal_rounding_mode.HALF_UP
    elif rounding_mode == DecimalRoundingMode.HALF_EVEN:
        cpp_mode = cpp_decimal_rounding_mode.HALF_EVEN
    else:
        raise ValueError(f"Invalid rounding mode: {rounding_mode}")

    # Check that both columns are decimal types
    if lhs.type().id() not in [
        type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128
    ]:
        raise TypeError(f"Left operand must be a decimal type, got {lhs.type()}")
    if rhs.type().id() not in [
        type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128
    ]:
        raise TypeError(f"Right operand must be a decimal type, got {rhs.type()}")

    # Call the C++ divide_decimal function
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_divide_decimal(
            lhs.view(),
            rhs.view(),
            cpp_mode
        )

    return Column.from_libcudf(move(result))


cpdef Column divide_decimal_column_scalar(
    Column lhs,
    Scalar rhs,
    rounding_mode=DecimalRoundingMode.HALF_UP
):
    """
    Perform decimal division of a column by a scalar, preserving the column's scale.

    This function divides a decimal column by a decimal scalar while maintaining
    the scale of the dividend (column), similar to Java's BigDecimal.divide()
    with a specified rounding mode.

    Parameters
    ----------
    lhs : Column
        The dividend column (must be a decimal type)
    rhs : Scalar
        The divisor scalar (must be a decimal type)
    rounding_mode : DecimalRoundingMode, default DecimalRoundingMode.HALF_UP
        The rounding mode to use:
        - HALF_UP: Round half away from zero
        - HALF_EVEN: Round half to even (banker's rounding)

    Returns
    -------
    Column
        Result column with the same scale as the dividend

    Raises
    ------
    TypeError
        If either operand is not a decimal type
    ValueError
        If an invalid rounding mode is provided

    Examples
    --------
    >>> import pylibcudf
    >>> # Column [10.50, 20.25] / Scalar 2.50
    >>> # Result: [4.20, 8.10] (scale preserved)
    """
    cdef cpp_decimal_rounding_mode cpp_mode

    # Convert rounding mode
    if rounding_mode == DecimalRoundingMode.HALF_UP:
        cpp_mode = cpp_decimal_rounding_mode.HALF_UP
    elif rounding_mode == DecimalRoundingMode.HALF_EVEN:
        cpp_mode = cpp_decimal_rounding_mode.HALF_EVEN
    else:
        raise ValueError(f"Invalid rounding mode: {rounding_mode}")

    # Check that column is decimal type
    if lhs.type().id() not in [
        type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128
    ]:
        raise TypeError(f"Left operand must be a decimal type, got {lhs.type()}")

    # Check that scalar is decimal type
    if rhs.type().id() not in [
        type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128
    ]:
        raise TypeError(f"Right operand must be a decimal type, got {rhs.type()}")

    # Call the C++ divide_decimal function for column/scalar
    cdef unique_ptr[column] result
    cdef const cpp_scalar* scalar_ptr = rhs.get()

    with nogil:
        result = cpp_divide_decimal(
            lhs.view(),
            dereference(scalar_ptr),
            cpp_mode
        )

    return Column.from_libcudf(move(result))


cpdef Column divide_decimal_scalar_column(
    Scalar lhs,
    Column rhs,
    rounding_mode=DecimalRoundingMode.HALF_UP
):
    """
    Perform decimal division of a scalar by a column, preserving the scalar's scale.

    This function divides a decimal scalar by a decimal column while maintaining
    the scale of the dividend (scalar), similar to Java's BigDecimal.divide()
    with a specified rounding mode.

    Parameters
    ----------
    lhs : Scalar
        The dividend scalar (must be a decimal type)
    rhs : Column
        The divisor column (must be a decimal type)
    rounding_mode : DecimalRoundingMode, default DecimalRoundingMode.HALF_UP
        The rounding mode to use:
        - HALF_UP: Round half away from zero
        - HALF_EVEN: Round half to even (banker's rounding)

    Returns
    -------
    Column
        Result column with the same scale as the dividend scalar

    Raises
    ------
    TypeError
        If either operand is not a decimal type
    ValueError
        If an invalid rounding mode is provided

    Examples
    --------
    >>> import pylibcudf
    >>> # Scalar 100.00 / Column [4.00, 5.00, 10.00]
    >>> # Result: [25.00, 20.00, 10.00] (scale preserved)
    """
    cdef cpp_decimal_rounding_mode cpp_mode

    # Convert rounding mode
    if rounding_mode == DecimalRoundingMode.HALF_UP:
        cpp_mode = cpp_decimal_rounding_mode.HALF_UP
    elif rounding_mode == DecimalRoundingMode.HALF_EVEN:
        cpp_mode = cpp_decimal_rounding_mode.HALF_EVEN
    else:
        raise ValueError(f"Invalid rounding mode: {rounding_mode}")

    # Check that scalar is decimal type
    if lhs.type().id() not in [
        type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128
    ]:
        raise TypeError(f"Left operand must be a decimal type, got {lhs.type()}")

    # Check that column is decimal type
    if rhs.type().id() not in [
        type_id.DECIMAL32, type_id.DECIMAL64, type_id.DECIMAL128
    ]:
        raise TypeError(f"Right operand must be a decimal type, got {rhs.type()}")

    # Call the C++ divide_decimal function for scalar/column
    cdef unique_ptr[column] result
    cdef const cpp_scalar* scalar_ptr = lhs.get()

    with nogil:
        result = cpp_divide_decimal(
            dereference(scalar_ptr),
            rhs.view(),
            cpp_mode
        )

    return Column.from_libcudf(move(result))
