# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/strings/regex/flags.hpp" namespace "cudf::strings" nogil:

    # Note that unlike most libcudf enums, this one is an enum and not an enum class.
    # That allows it to be used as a bitmask with bitwise operators.
    cpdef enum regex_flags:
        DEFAULT "cudf::strings::regex_flags::DEFAULT"
        IGNORECASE "cudf::strings::regex_flags::IGNORECASE"
        MULTILINE "cudf::strings::regex_flags::MULTILINE"
        DOTALL "cudf::strings::regex_flags::DOTALL"
