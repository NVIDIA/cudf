# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os

# Loading with RTLD_LOCAL adds the library itself to the loader's
# loaded library cache without loading any symbols into the global
# namespace. This allows libraries that express a dependency on
# this library to be loaded later and successfully satisfy this dependency
# without polluting the global symbol table with symbols from
# libcudf_streaming that could conflict with symbols from other DSOs.
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
    """Dynamically load libcudf_streaming.so and its dependencies"""
    try:
        # These libraries must be loaded before libcudf_streaming because
        # libcudf_streaming references their symbols.
        import libcudf
        import librapidsmpf
        import librmm

        librmm.load_library()
        libcudf.load_library()
        librapidsmpf.load_library()
    except ModuleNotFoundError:
        # libcudf_streaming's runtime dependencies may be satisfied by
        # natively installed libraries or conda packages, in which case
        # the imports will fail and we assume the libraries are
        # discoverable on system paths.
        pass

    return _load_library("libcudf_streaming.so")


def _load_library(soname):
    prefer_system_installation = (
        os.getenv(
            "RAPIDS_LIBCUDF_STREAMING_PREFER_SYSTEM_LIBRARY", "false"
        ).lower()
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
        # (which might be the case in builds where the library was prebuilt
        # before packaging the wheel), look for a system installation.
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
