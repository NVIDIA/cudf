# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiled device functions and borrowed device pointers for row transforms."""

from dataclasses import dataclass
from enum import IntEnum


class LTOBinaryType(IntEnum):
    """Binary format of a device fragment to link into a JIT kernel."""

    #: LTO-IR binary.
    LTO_IR = 0
    #: FATBIN binary.
    FATBIN = 1
    #: PTX source.
    PTX = 2


@dataclass(frozen=True)
class LTOFragment:
    """A compiled device fragment to link into a JIT kernel.

    A fragment may define multiple functions and reference functions in other
    supplied fragments. The fragment retains its immutable fragment bytes.

    Parameters
    ----------
    type : LTOBinaryType
        Binary format of this fragment.
    data : bytes
        Nonempty compiled fragment bytes. PTX need not include a null terminator.

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
    """A borrowed device pointer, passed as a raw ``void*`` UDF argument."""

    ptr: int = 0
    owner: object = None


@dataclass(frozen=True)
class TransformUDF:
    """An explicitly named transform function and the fragments defining it.

    The UDF must conform to the CUDF transform ABI.

    Parameters
    ----------
    symbol : str
        Exact exported mangled device symbol of the function.
    fragments : tuple[LTOFragment, ...]
        Compiled device fragments defining the function and its dependencies.
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
