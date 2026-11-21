# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiled device functions and borrowed device pointers for row transforms."""

from dataclasses import dataclass
from enum import IntEnum


class LTOBinaryType(IntEnum):
    """Binary format of a device module to link into a JIT kernel."""

    #: LTO-IR binary.
    LTO_IR = 0
    #: FATBIN binary.
    FATBIN = 1
    #: PTX source.
    PTX = 2


@dataclass(frozen=True)
class LTOFragment:
    """A compiled device module to link into a JIT kernel.

    A module may define multiple functions and reference functions in other
    supplied modules. The fragment retains its immutable module bytes.

    Parameters
    ----------
    type : LTOBinaryType
        Binary format of this module.
    data : bytes
        Nonempty compiled module bytes. PTX need not include a null terminator.

    Raises
    ------
    TypeError
        If ``type`` is not an :class:`LTOBinaryType`, or ``data`` is not
        a nonempty ``bytes`` object.
    """

    type: LTOBinaryType
    data: bytes

    def __post_init__(self):
        if not isinstance(self.type, LTOBinaryType):
            raise TypeError("type must be an LTOBinaryType")
        if not isinstance(self.data, bytes) or not self.data:
            raise TypeError(
                "data must contain nonempty LTO IR, fatbin, or PTX bytes"
            )


@dataclass(frozen=True)
class UserData:
    """A borrowed device pointer, passed as a raw ``void*`` UDF argument.

    Keep the allocation alive until work on the transform's stream completes.
    ``owner`` can retain the object owning that allocation. Zero is a valid pointer.
    """

    ptr: int = 0
    owner: object = None


@dataclass(frozen=True)
class TransformUDF:
    """An explicitly named transform function and the modules defining it.

    The CUDF ABI is::

        cudf::errc symbol([size_type index,]
                         Outputs*... outputs, Inputs... inputs);

    Transform supplies one logical row index before the output pointers.
    Output pointers precede all inputs. Indices are never nullable. The function
    returns a signed 32-bit status matching ``cudf::errc``.

    Null-aware calls wrap column and output values in ``cuda::std::optional``;
    :class:`UserData` inputs remain raw ``void*`` values. Modules may export
    multiple symbols and reference symbols in other supplied modules.
    This descriptor retains the module bytes. CUDA C++ source is not accepted.

    Parameters
    ----------
    symbol : str
        Exact exported device symbol, expressible as a CUDA identifier.
    fragments : tuple[LTOFragment, ...]
        Compiled device modules defining the function and its dependencies,
        in link order. Each fragment specifies its own binary format, so LTO IR,
        fatbin, and PTX modules may be mixed.
    is_null_aware : bool, default False
        Whether column and output values use ``cuda::std::optional``.
    indexed : bool, default False
        Whether a logical row index precedes all other arguments.

    Raises
    ------
    TypeError
        If ``fragments`` is empty or contains values other than
        :class:`LTOFragment` objects.
    """

    symbol: str
    fragments: tuple[LTOFragment, ...]
    is_null_aware: bool = False
    indexed: bool = False

    def __post_init__(self):
        if not self.fragments or any(
            not isinstance(fragment, LTOFragment)
            for fragment in self.fragments
        ):
            raise TypeError("fragments must contain LTOFragment objects")
