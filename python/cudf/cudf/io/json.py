# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from collections.abc import Collection, Mapping
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

import pylibcudf as plc

from cudf.core.column import access_columns
from cudf.core.dataframe import DataFrame
from cudf.core.dtypes import (
    CategoricalDtype,
    ListDtype,
    StructDtype,
    dtype as cudf_dtype,
)
from cudf.core.series import Series
from cudf.options import get_option
from cudf.utils import ioutils
from cudf.utils.dtypes import (
    _maybe_convert_to_default_type,
    dtype_to_pylibcudf_type,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cudf._typing import DtypeObj


def _get_cudf_schema_element_from_dtype(
    dtype,
) -> tuple[plc.DataType, list[tuple[str, plc.DataType, Any]]]:
    dtype = cudf_dtype(dtype)
    if isinstance(dtype, CategoricalDtype):
        raise NotImplementedError(
            "CategoricalDtype as dtype is not yet supported in JSON reader"
        )

    lib_type = dtype_to_pylibcudf_type(dtype)
    child_types = []

    if isinstance(dtype, StructDtype):
        for name, child_type in dtype.fields.items():
            child_lib_type, grandchild_types = (
                _get_cudf_schema_element_from_dtype(child_type)
            )
            child_types.append((name, child_lib_type, grandchild_types))
    elif isinstance(dtype, ListDtype):
        child_lib_type, grandchild_types = _get_cudf_schema_element_from_dtype(
            dtype.element_type
        )
        child_types = [
            ("offsets", plc.DataType(plc.TypeId.INT32), []),
            ("element", child_lib_type, grandchild_types),
        ]

    return lib_type, child_types


def _to_plc_compression(
    compression: Literal[
        "brotli",
        "bz2",
        "gzip",
        "infer",
        "lz4",
        "lzo",
        "snappy",
        "xz",
        "zip",
        "zlib",
        "zstd",
    ]
    | None,
) -> plc.io.types.CompressionType:
    compression_map = {
        None: plc.io.types.CompressionType.NONE,
        "brotli": plc.io.types.CompressionType.BROTLI,
        "bz2": plc.io.types.CompressionType.BZIP2,
        "gzip": plc.io.types.CompressionType.GZIP,
        "infer": plc.io.types.CompressionType.AUTO,
        "lz4": plc.io.types.CompressionType.LZ4,
        "lzo": plc.io.types.CompressionType.LZO,
        "snappy": plc.io.types.CompressionType.SNAPPY,
        "xz": plc.io.types.CompressionType.XZ,
        "zip": plc.io.types.CompressionType.ZIP,
        "zlib": plc.io.types.CompressionType.ZLIB,
        "zstd": plc.io.types.CompressionType.ZSTD,
    }
    try:
        return compression_map[compression]
    except KeyError:
        raise ValueError(f"Unsupported compression type: {compression}")


@ioutils.doc_read_json()
def read_json(
    path_or_buf,
    engine: Literal["auto", "pandas", "cudf"] = "auto",
    orient=None,
    dtype=None,
    lines: bool = False,
    compression: Literal[
        "bz2",
        "gzip",
        "infer",
        "snappy",
        "zip",
        "zstd",
    ]
    | None = "infer",
    byte_range: None | list[int] = None,
    keep_quotes: bool = False,
    storage_options=None,
    mixed_types_as_string: bool = False,
    prune_columns: bool = False,
    on_bad_lines: Literal["error", "recover"] = "error",
    *args,
    **kwargs,
) -> DataFrame:
    """{docstring}"""

    if dtype is not None and not isinstance(dtype, (Mapping, bool)):
        raise TypeError(
            "'dtype' parameter only supports "
            "a dict of column names and types as key-value pairs, "
            f"or a bool, or None. Got {type(dtype)}"
        )

    if engine == "auto":
        engine = "cudf" if lines else "pandas"
    if engine != "cudf" and keep_quotes:
        raise ValueError(
            "keep_quotes='True' is supported only with engine='cudf'"
        )

    if engine == "cudf":
        if dtype is None:
            dtype = True

        if args:
            raise ValueError(
                "cudf engine doesn't support the "
                f"following positional arguments: {list(args)}"
            )

        filepaths_or_buffers = ioutils.get_reader_filepath_or_buffer(
            path_or_buf,
            iotypes=(BytesIO, StringIO),
            storage_options=storage_options,
            warn_meta=("json", "read_json"),
            expand_dir_pattern="*.json",
        )

        c_compression = _to_plc_compression(compression)

        if on_bad_lines.lower() == "error":
            c_on_bad_lines = plc.io.types.JSONRecoveryMode.FAIL
        elif on_bad_lines.lower() == "recover":
            c_on_bad_lines = plc.io.types.JSONRecoveryMode.RECOVER_WITH_NULL
        else:
            raise TypeError(f"Invalid parameter for {on_bad_lines=}")

        processed_dtypes = None

        if dtype is False:
            raise ValueError("False value is unsupported for `dtype`")
        elif dtype is not True:
            processed_dtypes = []
            if isinstance(dtype, Mapping):
                for k, v in dtype.items():
                    # Make sure keys are string
                    k = str(k)
                    lib_type, child_types = (
                        _get_cudf_schema_element_from_dtype(v)
                    )
                    processed_dtypes.append((k, lib_type, child_types))
            elif isinstance(dtype, Collection):
                for col_dtype in dtype:
                    # Ignore child columns since we cannot specify their dtypes
                    # when passing a list
                    processed_dtypes.append(
                        _get_cudf_schema_element_from_dtype(col_dtype)[0]  # type: ignore[arg-type]
                    )
            else:
                raise TypeError("`dtype` must be 'list like' or 'dict'")

        if get_option("io.json.low_memory") and lines:
            res_cols, res_col_names, res_child_names = (
                plc.io.json.chunked_read_json(
                    plc.io.json._setup_json_reader_options(
                        plc.io.SourceInfo(filepaths_or_buffers),
                        processed_dtypes,
                        c_compression,
                        keep_quotes=keep_quotes,
                        mixed_types_as_string=mixed_types_as_string,
                        prune_columns=prune_columns,
                        recovery_mode=c_on_bad_lines,
                    )
                )
            )
            return DataFrame.from_pylibcudf(
                plc.Table(res_cols),
                metadata={
                    "columns": res_col_names,
                    "child_names": res_child_names,
                },
            )
        else:
            table_w_meta = plc.io.json.read_json(
                plc.io.json._setup_json_reader_options(
                    plc.io.SourceInfo(filepaths_or_buffers),
                    processed_dtypes,
                    c_compression,
                    lines,
                    byte_range_offset=byte_range[0]
                    if byte_range is not None
                    else 0,
                    byte_range_size=byte_range[1]
                    if byte_range is not None
                    else 0,
                    keep_quotes=keep_quotes,
                    mixed_types_as_string=mixed_types_as_string,
                    prune_columns=prune_columns,
                    recovery_mode=c_on_bad_lines,
                    extra_parameters=kwargs,
                )
            )
            df = DataFrame.from_pylibcudf(table_w_meta)
    else:
        warnings.warn(
            "Using CPU via Pandas to read JSON dataset, this may "
            "be GPU accelerated in the future"
        )

        filepath_or_buffer = ioutils.get_reader_filepath_or_buffer(
            path_or_data=path_or_buf,
            iotypes=(BytesIO, StringIO),
            storage_options=storage_options,
        )
        filepath_or_buffer = ioutils._select_single_source(
            filepath_or_buffer, "read_json (via pandas)"
        )
        if isinstance(filepath_or_buffer, bytes):
            # TODO: Remove once pandas 3.0 is minimum version
            # get_reader_filepath_or_buffer may have encoded raw data to bytes for libcudf
            filepath_or_buffer = filepath_or_buffer.decode()

        pd_value = pd.read_json(
            filepath_or_buffer,
            lines=lines,
            dtype=dtype,
            compression=compression,
            storage_options=storage_options,
            orient=orient,
            *args,
            **kwargs,
        )
        if isinstance(pd_value, pd.DataFrame):
            df = DataFrame(pd_value)
        else:
            df = Series(pd_value)

    if dtype is None:
        dtype = True

    if dtype is True or isinstance(dtype, Mapping):
        # There exists some dtypes in the result columns that is inferred.
        # Find them and map them to the default dtypes.
        specified_dtypes = {} if dtype is True else dtype
        unspecified_dtypes = {
            name: dtype
            for name, dtype in df._dtypes
            if name not in specified_dtypes
        }
        default_dtypes = {}

        for name, dt in unspecified_dtypes.items():
            if dt == np.dtype("i1"):
                # csv reader reads all null column as int8.
                # The dtype should remain int8.
                default_dtypes[name] = dt
            else:
                default_dtypes[name] = _maybe_convert_to_default_type(dt)
        df = df.astype(default_dtypes)

    return df


def _maybe_return_nullable_pd_obj(
    cudf_obj: DataFrame | Series,
) -> pd.DataFrame | pd.Series:
    try:
        return cudf_obj.to_pandas(nullable=True)
    except NotImplementedError:
        return cudf_obj.to_pandas(nullable=False)


def _dtype_to_names_list(dtype: DtypeObj) -> list[tuple[Hashable, Any]]:
    if isinstance(dtype, StructDtype):
        return [
            (name, _dtype_to_names_list(field_dtype))
            for name, field_dtype in dtype.fields.items()
        ]
    elif isinstance(dtype, ListDtype):
        # List columns have two children: offsets and values
        return [("", []), ("", _dtype_to_names_list(dtype.element_type))]
    return []


def _plc_write_json(
    table: Series | DataFrame,
    colnames: list[tuple[Hashable, Any]],
    path_or_buf,
    compression: Literal[
        "gzip",
        "snappy",
        "zstd",
    ]
    | None = None,
    na_rep: str = "null",
    include_nulls: bool = True,
    lines: bool = False,
    rows_per_chunk: int = 1024 * 64,  # 64K rows
    force_ascii: bool = True,
) -> None:
    with access_columns(*table._columns, mode="read", scope="internal"):
        try:
            # TODO: TableWithMetadata expects list[ColumnNameSpec] but receives list[tuple[Hashable, Any]]
            tbl_w_meta = plc.io.TableWithMetadata(
                plc.Table([col.plc_column for col in table._columns]),
                colnames,  # type: ignore[arg-type]
            )
            options = (
                plc.io.json.JsonWriterOptions.builder(
                    plc.io.SinkInfo([path_or_buf]), tbl_w_meta.tbl
                )
                .metadata(tbl_w_meta)
                .na_rep(na_rep)
                .include_nulls(include_nulls)
                .lines(lines)
                .compression(_to_plc_compression(compression))
                .utf8_escaped(force_ascii)
                .build()
            )
            if rows_per_chunk != np.iinfo(np.int32).max:
                options.set_rows_per_chunk(rows_per_chunk)
            plc.io.json.write_json(options)
        except OverflowError as err:
            raise OverflowError(
                f"Writing JSON file with rows_per_chunk={rows_per_chunk} failed. "
                "Consider providing a smaller rows_per_chunk argument."
            ) from err


@ioutils.doc_to_json()
def to_json(
    cudf_val: DataFrame | Series,
    path_or_buf=None,
    compression: Literal["gzip", "snappy", "zstd"] | None = None,
    engine: Literal["auto", "pandas", "cudf"] = "auto",
    orient=None,
    storage_options=None,
    *args,
    **kwargs,
):
    """{docstring}"""

    if engine == "auto":
        engine = "pandas"

    if engine == "cudf":
        if orient not in {"records", None}:
            raise ValueError(
                f"Only the `orient='records'` is supported for JSON writer"
                f" with `engine='cudf'`, got {orient}"
            )

        if path_or_buf is None:
            path_or_buf = StringIO()
            return_as_string = True
        else:
            path_or_buf = ioutils.get_writer_filepath_or_buffer(
                path_or_data=path_or_buf,
                mode="w",
                storage_options=storage_options,
            )
            return_as_string = False

        colnames = [
            (name, _dtype_to_names_list(col.dtype))
            for name, col in cudf_val._column_labels_and_values
        ]

        if ioutils.is_fsspec_open_file(path_or_buf):
            with path_or_buf as file_obj:
                file_obj = ioutils.get_IOBase_writer(file_obj)
                _plc_write_json(
                    cudf_val,
                    colnames,
                    path_or_buf,
                    compression,
                    *args,
                    **kwargs,
                )
        else:
            _plc_write_json(
                cudf_val, colnames, path_or_buf, compression, *args, **kwargs
            )

        if return_as_string:
            path_or_buf.seek(0)
            return path_or_buf.read()
    elif engine == "pandas":
        warnings.warn("Using CPU via Pandas to write JSON dataset")
        if isinstance(cudf_val, DataFrame):
            pd_data = {
                col: _maybe_return_nullable_pd_obj(series)
                for col, series in cudf_val.items()
            }
            pd_value = pd.DataFrame(pd_data)
        else:
            pd_value = _maybe_return_nullable_pd_obj(cudf_val)
        return pd_value.to_json(
            path_or_buf,
            orient=orient,
            storage_options=storage_options,
            *args,
            **kwargs,
        )
    else:
        raise ValueError(
            f"`engine` only support {{'auto', 'cudf', 'pandas'}}, "
            f"got: {engine}"
        )
