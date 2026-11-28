# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint64_t
from libcpp.string cimport string


# Only the parts of the header that are bound are declared. Declare more here, mirroring the
# header, when a Python consumer needs it.
cdef extern from "cucascade/io/column_admission.hpp" namespace "cucascade::io" nogil:

    cdef cppclass column_stats:
        string name
        uint64_t registered
        uint64_t requested
        uint64_t admitted
        uint64_t refused
        uint64_t admitted_footprint
        double rank
