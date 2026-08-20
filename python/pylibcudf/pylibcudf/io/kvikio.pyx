# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.io.config_utils cimport set_up_kvikio as cpp_set_up_kvikio

__all__ = ["set_up_kvikio"]


cpdef void set_up_kvikio():
    """Set KvikIO parameters.

    Parameters include:

    - Compatibility mode, according to the environment variable ``KVIKIO_COMPAT_MODE``. If
      ``KVIKIO_COMPAT_MODE`` is not set, enable it by default, which enforces the use of POSIX I/O.
    - Thread pool size, according to the environment variable ``KVIKIO_NTHREADS``. If
      ``KVIKIO_NTHREADS`` is not set, use 4 threads by default.

    Returns
    -------
    None
    """
    with nogil:
        cpp_set_up_kvikio()
