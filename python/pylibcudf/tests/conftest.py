# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# Tell ruff it's OK that some imports occur after the sys.path.insert
# ruff: noqa: E402
import io
import os
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pytest

import pylibcudf as plc
from pylibcudf.io.types import CompressionType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))

# Since we assume that the stream identification library is preloaded when tests are run
# in this mode, it is fine to load the library after pylibcudf inside conftest.
# Using conftest itself as the first point of loading is not viable since there is no
# guarantee that something else won't load CUDA libraries on startup (e.g. pth files)
if os.getenv("PYLIBCUDF_STREAM_TESTING", "0") == "1":

    def _get_default_stream():
        """Load the C++ stream identification library and fetch its default stream.

        This function leverages subprocess and ctypes to avoid needing to build and ship a
        testing-specific submodule in pylibcudf for this purpose. The testing module would
        also exhibit missing symbols by default without suitable library preloading in the
        test suite, so this approach is cleaner. Moreover, this approach avoids bloating the
        shipped wheels.
        """
        import ctypes
        import subprocess

        from rmm.pylibrmm.stream import Stream

        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudf_identify_stream_usage_mode_testing.so" in line:
                    stream_identification_lib_path = line.split()[-1]
                    break
            else:
                raise RuntimeError(
                    "The stream identification library must be preloaded when PYLIBCUDF_STREAM_TESTING=1"
                )

        # Get the symbol name for the default stream function
        symbols = subprocess.check_output(
            ["nm", "-D", "--defined-only", stream_identification_lib_path],
            text=True,
        )
        default_stream_func_name = next(
            line
            for line in symbols.splitlines()
            if "get_default_stream" in line
        ).split(" ")[2]

        # The library must already be preloaded at this point so neither RTLD_GLOBAL nor the
        # full path are strictly necessary, but we use them for clarity.
        stream_identification_lib = ctypes.CDLL(
            stream_identification_lib_path, mode=ctypes.RTLD_GLOBAL
        )
        stream_func = getattr(
            stream_identification_lib, default_stream_func_name
        )
        # Make sure the return type is not treated as a signed int
        stream_func.restype = ctypes.c_void_p
        stream_val = stream_func()

        # The CUDA stream protocol is currently the only way to produce an rmm Stream
        # pointing to an arbitrary cudaStream_t from Python (not Cython) code.
        class _Stream:
            def __init__(self, value):
                self._value = value

            def __cuda_stream__(self):
                return (0, self._value)

        return Stream(_Stream(stream_val))

    plc.utils.CUDF_DEFAULT_STREAM = _get_default_stream()

# This ensures that assertions from expressions like `left == right` in the
# common/utils.py are rewritten so that we get nice tracebacks.
# We need to register this before the module is imported.
pytest.register_assert_rewrite("utils")


from utils import (
    ALL_PA_TYPES,
    DEFAULT_PA_TYPES,
    NON_NESTED_PA_TYPES,
    NUMERIC_PA_TYPES,
)


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on stream parameter values.

    Tests that use streams internally without parametrization may also use the mark
    directly to indicate this.
    """
    for item in items:
        if hasattr(item, "callspec") and "stream" in item.callspec.params:
            stream_value = item.callspec.params["stream"]
            # Add marker based on stream parameter value being non-None
            if stream_value is not None:
                item.add_marker(pytest.mark.uses_custom_stream)


def _type_to_str(typ):
    if isinstance(typ, pa.ListType):
        return f"list[{_type_to_str(typ.value_type)}]"
    elif isinstance(typ, pa.StructType):
        return f"struct[{', '.join(_type_to_str(typ.field(i).type) for i in range(typ.num_fields))}]"
    else:
        return str(typ)


# This fixture defines [the standard set of types that all tests should default to
# running on. If there is a need for some tests to run on a different set of types, that
# type list fixture should also be defined below here if it is likely to be reused
# across modules. Otherwise it may be defined on a per-module basis.
@pytest.fixture(
    scope="session",
    params=DEFAULT_PA_TYPES,
    ids=_type_to_str,
)
def pa_type(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=NUMERIC_PA_TYPES,
)
def numeric_pa_type(request):
    return request.param


def _get_vals_of_type(pa_type, length, seed):
    """
    Returns an list-like of random values of that type
    """
    rng = np.random.default_rng(seed=seed)
    if pa_type == pa.int64():
        half = length // 2
        negs = rng.integers(-length, 0, half, dtype=np.int64)
        pos = rng.integers(0, length, length - half, dtype=np.int64)
        return np.concatenate([negs, pos])
    elif pa_type == pa.uint64():
        return rng.integers(0, length, length, dtype=np.uint64)
    elif pa_type == pa.float64():
        # Round to 6 decimal places or else we have problems comparing our
        # output to pandas due to floating point/rounding differences
        return rng.uniform(-length, length, length).round(6)
    elif pa_type == pa.bool_():
        return rng.integers(0, 2, length, dtype=bool)
    elif pa_type == pa.string():
        # Generate random ASCII strings
        strs = []
        for _ in range(length):
            chrs = rng.integers(33, 128, length)
            strs.append("".join(chr(x) for x in chrs))
        return strs
    else:
        raise NotImplementedError(
            f"random data generation not implemented for {pa_type}"
        )


# TODO: Consider adapting this helper function
# to consider nullability
def _generate_table_data(types, nrows, seed=42):
    table_dict = {}
    colnames = []

    for typ in types:
        child_colnames = []

        def _generate_nested_data(typ):
            child_colnames = []

            # recurse to get vals for children
            rand_arrs = []
            for i in range(typ.num_fields):
                rand_arr, grandchild_colnames = _generate_nested_data(
                    typ.field(i).type
                )
                rand_arrs.append(rand_arr)
                child_colnames.append((typ.field(i).name, grandchild_colnames))

            if isinstance(typ, pa.StructType):
                pa_array = pa.StructArray.from_arrays(
                    [rand_arr for rand_arr in rand_arrs],
                    names=[typ.field(i).name for i in range(typ.num_fields)],
                )
            elif isinstance(typ, pa.ListType):
                pa_array = pa.array(
                    [
                        list(row_vals)
                        for row_vals in zip(rand_arrs[0], strict=True)
                    ],
                    type=typ,
                )
                child_colnames.append(("", grandchild_colnames))
            else:
                # typ is scalar type
                pa_array = pa.array(
                    _get_vals_of_type(typ, nrows, seed=seed), type=typ
                )
            return pa_array, child_colnames

        if isinstance(typ, (pa.ListType, pa.StructType)):
            rand_arr, child_colnames = _generate_nested_data(typ)
        else:
            rand_arr = pa.array(
                _get_vals_of_type(typ, nrows, seed=seed), type=typ
            )

        table_dict[f"col_{typ}"] = rand_arr
        colnames.append((f"col_{typ}", child_colnames))

    pa_table = pa.Table.from_pydict(table_dict)

    # TODO: Once interop APIs support a stream we should pass one and synchronize on it
    # to avoid syncing the default stream here.
    plc_table = plc.io.TableWithMetadata(
        plc.Table.from_arrow(pa_table), column_names=colnames
    )
    plc.utils.DEFAULT_STREAM.synchronize()
    return plc_table, pa_table


@pytest.fixture(scope="session", params=[0, 100])
def table_data(
    request: pytest.FixtureRequest,
) -> tuple[plc.io.types.TableWithMetadata, pa.Table]:
    """
    Returns (TableWithMetadata, pa_table).

    This is the default fixture you should be using for testing
    pylibcudf I/O writers.

    Contains one of each category (e.g. int, bool, list, struct)
    of dtypes.
    """
    nrows = request.param
    return _generate_table_data(ALL_PA_TYPES, nrows)


@pytest.fixture(scope="session", params=[0, 100])
def table_data_with_non_nested_pa_types(request):
    """
    Returns (TableWithMetadata, pa_table).

    This fixture is for testing with non-nested PyArrow types.
    """
    nrows = request.param
    return _generate_table_data(NON_NESTED_PA_TYPES, nrows)


@pytest.fixture(params=[(0, 0), ("half", 0), (-1, "half")])
def nrows_skiprows(table_data, request):
    """
    Parametrized nrows fixture that accompanies table_data
    """
    _, pa_table = table_data
    nrows, skiprows = request.param
    if nrows == "half":
        nrows = len(pa_table) // 2
    if skiprows == "half":
        skiprows = (len(pa_table) - nrows) // 2
    return nrows, skiprows


@pytest.fixture(
    params=["a.txt", pathlib.Path("a.txt"), io.BytesIO, io.StringIO],
)
def source_or_sink(request, tmp_path):
    fp_or_buf = request.param
    if isinstance(fp_or_buf, str):
        return f"{tmp_path}/{fp_or_buf}"
    elif isinstance(fp_or_buf, os.PathLike):
        return tmp_path.joinpath(fp_or_buf)
    elif issubclass(fp_or_buf, io.IOBase):
        # Must construct io.StringIO/io.BytesIO inside
        # fixture, or we'll end up re-using it
        return fp_or_buf()


@pytest.fixture(
    params=["a.txt", pathlib.Path("a.txt"), io.BytesIO],
)
def binary_source_or_sink(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> str | pathlib.Path | io.BytesIO:
    fp_or_buf = request.param
    if isinstance(fp_or_buf, str):
        return f"{tmp_path}/{fp_or_buf}"
    elif isinstance(fp_or_buf, os.PathLike):
        return tmp_path.joinpath(fp_or_buf)
    elif issubclass(fp_or_buf, io.IOBase):
        # Must construct io.StringIO/io.BytesIO inside
        # fixture, or we'll end up re-using it
        return fp_or_buf()


unsupported_types = {
    # Not supported by pandas
    # TODO: find a way to test these
    CompressionType.SNAPPY,
    CompressionType.BROTLI,
    CompressionType.LZ4,
    CompressionType.LZO,
    CompressionType.ZLIB,
}

unsupported_text_compression_types = unsupported_types.union(
    {
        # compressions not supported by libcudf
        # for csv/json
        CompressionType.XZ,
        CompressionType.ZSTD,
    }
)


@pytest.fixture(
    params=set(CompressionType).difference(unsupported_text_compression_types)
)
def text_compression_type(request):
    return request.param


@pytest.fixture(params=[opt for opt in plc.io.types.CompressionType])
def compression_type(request):
    return request.param


@pytest.fixture(
    scope="session", params=[opt for opt in plc.types.Interpolation]
)
def interp_opt(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[opt for opt in plc.types.Sorted],
)
def sorted_opt(request):
    return request.param


@pytest.fixture(
    scope="session", params=[False, True], ids=["without_nulls", "with_nulls"]
)
def has_nulls(request):
    return request.param


@pytest.fixture(
    scope="session", params=[False, True], ids=["without_nans", "with_nans"]
)
def has_nans(request):
    return request.param


@pytest.fixture(scope="session")
def patch_cupy_stream(request):
    import cupy as cp

    # TODO: Remove this version conditional once we require CuPy 14
    if hasattr(cp.cuda.Stream, "from_external"):
        return cp.cuda.Stream.from_external(plc.utils.CUDF_DEFAULT_STREAM)
    else:
        version, stream_ptr = plc.utils.CUDF_DEFAULT_STREAM.__cuda_stream__()
        assert version == 0
        return cp.cuda.ExternalStream(stream_ptr)
