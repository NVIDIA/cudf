# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import ctypes
import os

# Loading with RTLD_LOCAL adds the library itself to the loader's
# loaded library cache without loading any symbols into the global
# namespace. This allows libraries that express a dependency on
# this library to be loaded later and successfully satisfy this dependency
# without polluting the global symbol table with symbols from
# libcudf that could conflict with symbols from other DSOs.
PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _load_system_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``
    Raises ``OSError`` if library cannot be loaded.
    """
    return ctypes.CDLL(soname, PREFERRED_LOAD_FLAG)


def _load_wheel_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``

    Returns ``None`` if the library cannot be loaded.
    """
    if os.path.isfile(
        lib := os.path.join(os.path.dirname(__file__), "lib64", soname)
    ):
        return ctypes.CDLL(lib, PREFERRED_LOAD_FLAG)
    return None


def load_library():
    """Dynamically load libcudf.so and its dependencies"""

    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib

        load_nvidia_dynamic_lib("nvJitLink")
        load_nvidia_dynamic_lib("nvrtc")
    except ModuleNotFoundError:
        pass

    try:
        # librmm and libkvikio must be loaded before libcudf because libcudf references
        # them.
        import libkvikio
        import librmm
        import nvidia.libnvcomp
        import rapids_logger

        rapids_logger.load_library()
        librmm.load_library()
        libkvikio.load_library()
        nvidia.libnvcomp.load_library()
    except ModuleNotFoundError:
        # libcudf's runtime dependency on libkvikio may be satisfied by a natively
        # installed library or a conda package, in which case the import will fail and
        # we assume the library is discoverable on system paths.
        pass

    return _load_library("libcudf.so")


def _load_library(soname):
    prefer_system_installation = (
        os.getenv("RAPIDS_LIBCUDF_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    found_lib = None
    if prefer_system_installation:
        # Prefer a system library if one is present to
        # avoid clobbering symbols that other packages might expect, but if no
        # other library is present use the one in the wheel.
        try:
            found_lib = _load_system_installation(soname)
        except OSError:
            found_lib = _load_wheel_installation(soname)
    else:
        # Prefer the libraries bundled in this package. If they aren't found
        # (which might be the case in builds where the library was prebuilt before
        # packaging the wheel), look for a system installation.
        try:
            found_lib = _load_wheel_installation(soname)
            if found_lib is None:
                found_lib = _load_system_installation(soname)
        except OSError:
            # If none of the searches above succeed, just silently return None
            # and rely on other mechanisms (like RPATHs on other DSOs) to
            # help the loader find the library.
            pass

    # The caller almost never needs to do anything with this library, but no
    # harm in offering the option since this object at least provides a handle
    # to inspect where the library was loaded from.
    return found_lib
