# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.libcudf.strings.regex_flags cimport regex_flags
from pylibcudf.libcudf.strings.regex_program cimport regex_program

__all__ = ["RegexProgram"]

cdef class RegexProgram:
    """Regex program class.

    This is the Cython representation of
    :cpp:class:`cudf::strings::regex_program`.

    Do not instantiate this class directly, use the `create` method.

    """
    def __init__(self, *args, **kwargs):
        raise ValueError("Do not instantiate RegexProgram directly, use create")

    __hash__ = None

    @staticmethod
    def create(str pattern, regex_flags flags) -> RegexProgram:
        """Create a program from a pattern.

        For detils, see :cpp:func:`create`.

        Parameters
        ----------
        pattern : str
            Regex pattern
        flags : RegexFlags
            Regex flags for interpreting special characters in the pattern

        Returns
        -------
        RegexProgram
            A new RegexProgram
        """
        cdef unique_ptr[regex_program] c_prog
        cdef string c_pattern = pattern.encode()

        cdef RegexProgram ret = RegexProgram.__new__(RegexProgram)
        with nogil:
            c_prog = regex_program.create(c_pattern, flags)

        ret.c_obj = move(c_prog)
        return ret
