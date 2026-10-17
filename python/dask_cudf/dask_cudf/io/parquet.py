# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import itertools
import math
import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from dask._task_spec import List as TaskList, Task
from dask.dataframe.io.parquet.arrow import _filters_to_expression
from dask.dataframe.io.parquet.core import ParquetFunctionWrapper
from dask.tokenize import tokenize
from dask.utils import parse_bytes

import cudf

from dask_cudf._expr import (
    Elemwise,
    FragmentWrapper,
    FusedIO,
    FusedParquetIO,
    ReadParquetFSSpec,
    ReadParquetPyarrowFS,
    _convert_to_list,
    new_collection,
)

# Dask-expr imports CudfEngine from this module
from dask_cudf._legacy.io.parquet import CudfEngine
from dask_cudf.core import _deprecated_api

if TYPE_CHECKING:
    from collections.abc import MutableMapping


_DEVICE_SIZE_CACHE: int | None = None


def _get_device_size():
    try:
        # Use NVML to find the worker device size.
        from cuda.core import system

        index_or_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[
            0
        ]
        if index_or_uuid and not index_or_uuid.isnumeric():
            # This means index is UUID. This works for both MIG and non-MIG device UUIDs.
            device = system.Device(uuid=index_or_uuid)
            if device.mig.is_mig_device:
                device = device.mig.parent
        else:
            # This is a device index
            device = system.Device(index=int(index_or_uuid))
        return device.memory_info.total

    except (ValueError, system.NotSupportedError):
        # Fall back to a conservative 8GiB default
        return 8 * 1024**3


def _normalize_blocksize(fraction: float = 0.03125):
    # Set the blocksize to fraction * <device-size>.
    # We use the smallest worker device to set <device-size>.
    # (Default blocksize is 1/32 * <device-size>)
    global _DEVICE_SIZE_CACHE

    if _DEVICE_SIZE_CACHE is None:
        try:
            # Check distributed workers (if a client exists)
            from distributed import get_client

            client = get_client()
            # TODO: Check "GPU" worker resources only.
            # Depends on (https://github.com/rapidsai/dask-cuda/pull/1401)
            device_size = min(client.run(_get_device_size).values())
        except (ImportError, ValueError):
            device_size = _get_device_size()
        _DEVICE_SIZE_CACHE = device_size

    return int(_DEVICE_SIZE_CACHE * fraction)


class NoOp(Elemwise):
    # Workaround - Always wrap read_parquet operations
    # in a NoOp to trigger tune_up optimizations.
    _parameters = ["frame"]
    _is_length_preserving = True
    _projection_passthrough = True
    _filter_passthrough = True
    _preserves_partitioning_information = True

    @staticmethod
    def operation(x):
        return x


class CudfReadParquetFSSpec(ReadParquetFSSpec):
    _STATS_CACHE: MutableMapping[str, Any] = {}

    def approx_statistics(self):
        # Use a few files to approximate column-size statistics
        key = tokenize(self._dataset_info["ds"].files[:10], self.filters)
        try:
            return self._STATS_CACHE[key]

        except KeyError:
            # Account for filters
            ds_filters = None
            if self.filters is not None:
                ds_filters = _filters_to_expression(self.filters)

            # Use average total_uncompressed_size of three files
            n_sample = 3
            column_sizes = {}
            for i, frag in enumerate(
                self._dataset_info["ds"].get_fragments(ds_filters)
            ):
                md = frag.metadata
                for rg in range(md.num_row_groups):
                    row_group = md.row_group(rg)
                    for col in range(row_group.num_columns):
                        column = row_group.column(col)
                        name = column.path_in_schema
                        if name not in column_sizes:
                            column_sizes[name] = np.zeros(
                                n_sample, dtype="int64"
                            )
                        column_sizes[name][i] += column.total_uncompressed_size
                if (i + 1) >= n_sample:
                    break

            # Reorganize stats to look like arrow-fs version
            self._STATS_CACHE[key] = {
                "columns": [
                    {
                        "path_in_schema": name,
                        "total_uncompressed_size": np.mean(sizes),
                    }
                    for name, sizes in column_sizes.items()
                ]
            }
            return self._STATS_CACHE[key]

    @functools.cached_property
    def _fusion_compression_factor(self):
        # Disable fusion when blocksize=None
        if self.blocksize is None:
            return 1

        # At this point, we *may* have used `blockwise`
        # already to split or aggregate files. We don't
        # *know* if the current partitions correspond to
        # individual/full files, multiple/aggregated files
        # or partial/split files.
        #
        # Therefore, we need to use the statistics from
        # a few files to estimate the current partition
        # size. This size should be similar to `blocksize`
        # *if* aggregate_files is True or if the files
        # are *smaller* than `blocksize`.

        # Step 1: Sample statistics
        approx_stats = self.approx_statistics()
        projected_size, original_size = 0, 0
        col_op = self.operand("columns") or self.columns
        for col in approx_stats["columns"]:
            original_size += col["total_uncompressed_size"]
            if col["path_in_schema"] in col_op or (
                (split_name := col["path_in_schema"].split("."))
                and split_name[0] in col_op
            ):
                projected_size += col["total_uncompressed_size"]
        if original_size < 1 or projected_size < 1:
            return 1

        # Step 2: Estimate the correction factor
        # (Correct for possible pre-optimization fusion/splitting)
        blocksize = parse_bytes(self.blocksize)
        if original_size > blocksize:
            # Input files are bigger than blocksize
            # and we already split these large files.
            # (correction_factor > 1)
            correction_factor = original_size / blocksize
        elif self.aggregate_files:
            # Input files are smaller than blocksize
            # and we already aggregate small files.
            # (correction_factor == 1)
            correction_factor = 1
        else:
            # Input files are smaller than blocksize
            # but we haven't aggregate small files yet.
            # (correction_factor < 1)
            correction_factor = original_size / blocksize

        # Step 3. Estimate column-projection factor
        if self.operand("columns") is None:
            projection_factor = 1
        else:
            projection_factor = projected_size / original_size

        return max(projection_factor * correction_factor, 0.001)

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        if isinstance(parent, FusedIO):
            return
        return parent.substitute(self, CudfFusedIO(self))


class CudfReadParquetPyarrowFS(ReadParquetPyarrowFS):
    _parameters = [
        "path",
        "columns",
        "filters",
        "categories",
        "index",
        "storage_options",
        "filesystem",
        "blocksize",
        "ignore_metadata_file",
        "calculate_divisions",
        "arrow_to_pandas",
        "pyarrow_strings_enabled",
        "kwargs",
        "_partitions",
        "_series",
        "_dataset_info_cache",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "categories": None,
        "index": None,
        "storage_options": None,
        "filesystem": None,
        "blocksize": "256 MiB",
        "ignore_metadata_file": True,
        "calculate_divisions": False,
        "arrow_to_pandas": None,
        "pyarrow_strings_enabled": True,
        "kwargs": None,
        "_partitions": None,
        "_series": False,
        "_dataset_info_cache": None,
    }

    @functools.cached_property
    def _dataset_info(self):
        from dask_cudf._legacy.io.parquet import (
            set_object_dtypes_from_pa_schema,
        )

        dataset_info = super()._dataset_info
        meta_pd = dataset_info["base_meta"]
        if isinstance(meta_pd, cudf.DataFrame):
            return dataset_info

        # Convert to cudf
        # (drop unsupported timezone information)
        for k, v in meta_pd.dtypes.items():
            if isinstance(v, pd.DatetimeTZDtype) and v.tz is not None:
                meta_pd[k] = meta_pd[k].dt.tz_localize(None)
        meta_cudf = cudf.from_pandas(meta_pd)

        # Re-set "object" dtypes to align with pa schema
        kwargs = dataset_info.get("kwargs", {})
        set_object_dtypes_from_pa_schema(
            meta_cudf,
            kwargs.get("schema", None),
        )

        dataset_info["base_meta"] = meta_cudf
        self.operands[type(self)._parameters.index("_dataset_info_cache")] = (
            dataset_info
        )
        return dataset_info

    @staticmethod
    def _table_to_pandas(table, index_name):
        if isinstance(table, cudf.DataFrame):
            df = table
        else:
            df = cudf.DataFrame.from_arrow(table)
            if index_name is not None:
                return df.set_index(index_name)
        return df

    @staticmethod
    def _fragments_to_cudf_dataframe(
        fragment_wrappers,
        filters,
        columns,
        schema,
    ):
        from dask.dataframe.io.utils import _is_local_fs

        from cudf.io.parquet import _apply_post_filters, _normalize_filters

        if not isinstance(fragment_wrappers, list):
            fragment_wrappers = [fragment_wrappers]

        filesystem = None
        paths, row_groups = [], []
        for fw in fragment_wrappers:
            frag = fw.fragment if isinstance(fw, FragmentWrapper) else fw
            paths.append(frag.path)
            row_groups.append(
                [rg.id for rg in frag.row_groups] if frag.row_groups else None
            )
            if filesystem is None:
                filesystem = frag.filesystem

        if _is_local_fs(filesystem):
            filesystem = None
        else:
            from fsspec.implementations.arrow import ArrowFSWrapper

            filesystem = ArrowFSWrapper(filesystem)
            protocol = filesystem.protocol
            paths = [f"{protocol}://{path}" for path in paths]

        filters = _normalize_filters(filters)
        projected_columns = None
        if columns and filters:
            projected_columns = [c for c in columns if c is not None]
            columns = sorted(
                set(v[0] for v in itertools.chain.from_iterable(filters))
                | set(projected_columns)
            )

        if row_groups == [None for path in paths]:
            row_groups = None

        df = cudf.read_parquet(
            paths,
            columns=columns,
            filters=filters,
            row_groups=row_groups,
            dataset_kwargs={"schema": schema},
        )

        # Apply filters (if any are defined)
        df = _apply_post_filters(df, filters)
        if projected_columns:
            # Elements of `projected_columns` may now be in the index.
            # We must filter these names from our projection
            projected_columns = [
                col for col in projected_columns if col in df._column_names
            ]
            df = df[projected_columns]

        # TODO: Deal with hive partitioning.
        # Note that ReadParquetPyarrowFS does NOT support this yet anyway.
        return df

    @functools.cached_property
    def _use_device_io(self):
        from dask.dataframe.io.utils import _is_local_fs

        # Use host for remote filesystem only
        # (Unless we are using kvikio-S3)
        return _is_local_fs(self.fs) or (
            self.fs.type_name == "s3" and cudf.get_option("kvikio_remote_io")
        )

    def _filtered_task(self, name, index: int):
        columns = self.columns.copy()
        index_name = self.index.name
        if self.index is not None:
            index_name = self.index.name
        schema = self._dataset_info["schema"].remove_metadata()
        if index_name:
            if columns is None:
                columns = list(schema.names)
            columns.append(index_name)

        frag_to_table = self._fragment_to_table
        if self._use_device_io:
            frag_to_table = self._fragments_to_cudf_dataframe

        return Task(
            name,
            self._table_to_pandas,
            Task(
                None,
                frag_to_table,
                fragment_wrapper=FragmentWrapper(
                    self.fragments[index], filesystem=self.fs
                ),
                filters=self.filters,
                columns=columns,
                schema=schema,
            ),
            index_name=index_name,
        )

    @property
    def _fusion_compression_factor(self):
        blocksize = self.blocksize
        if blocksize is None:
            return 1
        elif blocksize == "default":
            blocksize = "256MiB"

        projected_size = 0
        approx_stats = self.approx_statistics()
        col_op = self.operand("columns") or self.columns
        for col in approx_stats["columns"]:
            if col["path_in_schema"] in col_op or (
                (split_name := col["path_in_schema"].split("."))
                and split_name[0] in col_op
            ):
                projected_size += col["total_uncompressed_size"]

        if projected_size < 1:
            return 1

        aggregate_files = max(1, int(parse_bytes(blocksize) / projected_size))
        return max(1 / aggregate_files, 0.001)

    def _tune_up(self, parent):
        if self._fusion_compression_factor >= 1:
            return
        fused_cls = (
            CudfFusedParquetIO
            if self._use_device_io
            else CudfFusedParquetIOHost
        )
        if isinstance(parent, fused_cls):
            return
        return parent.substitute(self, fused_cls(self))


class CudfFusedIO(FusedIO):
    def _task(self, name, index: int):
        expr = self.operand("_expr")
        bucket = self._fusion_buckets[index]
        io_func = expr._filtered_task(name, 0).func
        if not isinstance(
            io_func, ParquetFunctionWrapper
        ) or io_func.common_kwargs.get("partitions", None):
            # Just use "simple" fusion if we have an unexpected
            # callable, or we are dealing with hive partitioning.
            return Task(
                name,
                cudf.concat,
                TaskList(*(expr._filtered_task(name, i) for i in bucket)),
            )

        pieces = []
        for i in bucket:
            piece = expr._filtered_task(name, i).args[0]
            if isinstance(piece, list):
                pieces.extend(piece)
            else:
                pieces.append(piece)
        return Task(name, io_func, pieces)


class CudfFusedParquetIO(FusedParquetIO):
    @functools.cached_property
    def _fusion_buckets(self):
        partitions = self.operand("_expr")._partitions
        npartitions = len(partitions)

        step = math.ceil(1 / self.operand("_expr")._fusion_compression_factor)

        # TODO: Heuristic to limit fusion should probably
        # account for the number of workers. For now, just
        # limiting fusion to 100 partitions at once.
        step = min(step, 100)

        buckets = [
            partitions[i : i + step] for i in range(0, npartitions, step)
        ]
        return buckets

    @classmethod
    def _load_multiple_files(
        cls,
        frag_filters,
        columns,
        schema,
        **to_pandas_kwargs,
    ):
        frag_to_table = CudfReadParquetPyarrowFS._fragments_to_cudf_dataframe
        return CudfReadParquetPyarrowFS._table_to_pandas(
            frag_to_table(
                [frag[0] for frag in frag_filters],
                frag_filters[0][1],  # TODO: Check for consistent filters?
                columns,
                schema,
            ),
            **to_pandas_kwargs,
        )


class CudfFusedParquetIOHost(CudfFusedParquetIO):
    @classmethod
    def _load_multiple_files(
        cls,
        frag_filters,
        columns,
        schema,
        **to_pandas_kwargs,
    ):
        import pyarrow as pa

        from dask.threaded import get
        from dask.tokenize import tokenize
        from dask.utils import apply

        token = tokenize(frag_filters, columns, schema)
        name = f"pq-file-{token}"
        dsk = {
            (name, i): (
                CudfReadParquetPyarrowFS._fragment_to_table,
                frag,
                filter,
                columns,
                schema,
            )
            for i, (frag, filter) in enumerate(frag_filters)
        }
        dsk[name] = (
            apply,
            pa.concat_tables,
            [list(dsk.keys())],
            {"promote_options": "permissive"},
        )

        return CudfReadParquetPyarrowFS._table_to_pandas(
            get(dsk, name),
            **to_pandas_kwargs,
        )


def read_parquet_expr(
    path,
    *args,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    dtype_backend=None,
    calculate_divisions=False,
    ignore_metadata_file=False,
    metadata_task_size=None,
    split_row_groups="infer",
    blocksize="default",
    aggregate_files=None,
    parquet_file_extension=(".parq", ".parquet", ".pq"),
    filesystem="fsspec",
    engine=None,
    arrow_to_pandas=None,
    open_file_options=None,
    **kwargs,
):
    """
    Read a Parquet file into a Dask-cuDF DataFrame.

    This reads a directory of Parquet data into a DataFrame collection.
    Partitioning behavior mostly depends on the ``blocksize`` argument.

    .. note::
        Dask may automatically resize partitions at optimization time.
        Please set ``blocksize=None`` to disable this behavior in Dask cuDF.
        (NOTE: This will not disable fusion for the "pandas" backend)

    .. note::
        Specifying ``filesystem="arrow"`` leverages a complete reimplementation of
        the Parquet reader that is solely based on PyArrow. It is faster than the
        legacy implementation in some cases, but doesn't yet support all features.

    Parameters
    ----------
    path : str or list
        Source directory for data, or path(s) to individual parquet files.
        Prefix with a protocol like ``s3://`` to read from alternative
        filesystems. To read from multiple files you can pass a globstring or a
        list of paths, with the caveat that they must all have the same
        protocol.
    columns : str or list, default None
        Field name(s) to read in as columns in the output. By default all
        non-index fields will be read (as determined by the pandas parquet
        metadata, if present). Provide a single field name instead of a list to
        read in the data as a Series.
    filters : Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]]], default None
        List of filters to apply, like ``[[('col1', '==', 0), ...], ...]``.
        Using this argument will result in row-wise filtering of the final partitions.

        Predicates can be expressed in disjunctive normal form (DNF). This means that
        the inner-most tuple describes a single column predicate. These inner predicates
        are combined with an AND conjunction into a larger predicate. The outer-most
        list then combines all of the combined filters with an OR disjunction.

        Predicates can also be expressed as a ``List[Tuple]``. These are evaluated
        as an AND conjunction. To express OR in predicates, one must use the
        (preferred for "pyarrow") ``List[List[Tuple]]`` notation.
    index : str, list or False, default None
        Field name(s) to use as the output frame index. By default will be
        inferred from the pandas parquet file metadata, if present. Use ``False``
        to read all fields as columns.
    categories : list or dict, default None
        For any fields listed here, if the parquet encoding is Dictionary,
        the column will be created with dtype category. Use only if it is
        guaranteed that the column is encoded as dictionary in all row-groups.
        If a list, assumes up to 2**16-1 labels; if a dict, specify the number
        of labels expected; if None, will load categories automatically for
        data written by dask, not otherwise.
    storage_options : dict, default None
        Key/value pairs to be passed on to the file-system backend, if any.
        Note that the default file-system backend can be configured with the
        ``filesystem`` argument, described below.
    calculate_divisions : bool, default False
        Whether to use min/max statistics from the footer metadata (or global
        ``_metadata`` file) to calculate divisions for the output DataFrame
        collection. Divisions will not be calculated if statistics are missing.
        This option will be ignored if ``index`` is not specified and there is
        no physical index column specified in the custom "pandas" Parquet
        metadata. Note that ``calculate_divisions=True`` may be extremely slow
        when no global ``_metadata`` file is present, especially when reading
        from remote storage. Set this to ``True`` only when known divisions
        are needed for your workload (see :ref:`dataframe-design-partitions`).
    ignore_metadata_file : bool, default False
        Whether to ignore the global ``_metadata`` file (when one is present).
        If ``True``, or if the global ``_metadata`` file is missing, the parquet
        metadata may be gathered and processed in parallel. Parallel metadata
        processing is currently supported for ``ArrowDatasetEngine`` only.
    metadata_task_size : int, default configurable
        If parquet metadata is processed in parallel (see ``ignore_metadata_file``
        description above), this argument can be used to specify the number of
        dataset files to be processed by each task in the Dask graph.  If this
        argument is set to ``0``, parallel metadata processing will be disabled.
        The default values for local and remote filesystems can be specified
        with the "metadata-task-size-local" and "metadata-task-size-remote"
        config fields, respectively (see "dataframe.parquet").
    split_row_groups : 'infer', 'adaptive', bool, or int, default 'infer'
        WARNING: The ``split_row_groups`` argument is now deprecated, please use
        ``blocksize`` instead.

    blocksize : int, float or str, default 'default'
        The desired size of each output ``DataFrame`` partition in terms of total
        (uncompressed) parquet storage space. This argument may be used to split
        large files or aggregate small files into the same partition. Use ``None``
        for a simple 1:1 mapping between files and partitions. Use a float value
        less than 1.0 to specify the fractional size of the partitions with
        respect to the total memory of the first NVIDIA GPU on your machine.
        Default is 1/32 the total memory of a single GPU.
    aggregate_files : bool or str, default None
        WARNING: The behavior of ``aggregate_files=True`` is now obsolete
        when query-planning is enabled (the default). Small files are now
        aggregated automatically according to the ``blocksize`` setting.
        Please expect this argument to be deprecated in a future release.

        WARNING: Passing a string argument to ``aggregate_files`` will result
        in experimental behavior that may be removed at any time.

    parquet_file_extension: str, tuple[str], or None, default (".parq", ".parquet", ".pq")
        A file extension or an iterable of extensions to use when discovering
        parquet files in a directory. Files that don't match these extensions
        will be ignored. This argument only applies when ``paths`` corresponds
        to a directory and no ``_metadata`` file is present (or
        ``ignore_metadata_file=True``). Passing in ``parquet_file_extension=None``
        will treat all files in the directory as parquet files.

        The purpose of this argument is to ensure that the engine will ignore
        unsupported metadata files (like Spark's '_SUCCESS' and 'crc' files).
        It may be necessary to change this argument if the data files in your
        parquet dataset do not end in ".parq", ".parquet", or ".pq".
    filesystem: "fsspec", "arrow", or fsspec.AbstractFileSystem backend to use.
    dataset: dict, default None
        Dictionary of options to use when creating a ``pyarrow.dataset.Dataset`` object.
        These options may include a "filesystem" key to configure the desired
        file-system backend. However, the top-level ``filesystem`` argument will always
        take precedence.

        **Note**: The ``dataset`` options may include a "partitioning" key.
        However, since ``pyarrow.dataset.Partitioning``
        objects cannot be serialized, the value can be a dict of key-word
        arguments for the ``pyarrow.dataset.partitioning`` API
        (e.g. ``dataset={"partitioning": {"flavor": "hive", "schema": ...}}``).
        Note that partitioned columns will not be converted to categorical
        dtypes when a custom partitioning schema is specified in this way.
    read: dict, default None
        Dictionary of options to pass through to ``CudfEngine.read_partitions``
        using the ``read`` key-word argument.
    """

    from fsspec.utils import stringify_path
    from pyarrow import fs as pa_fs

    from dask.core import flatten
    from dask.dataframe.utils import pyarrow_strings_enabled

    if args:
        raise ValueError(f"Unexpected positional arguments: {args}")

    if open_file_options is not None:
        raise ValueError(
            "The open_file_options argument is no longer supported "
            "by the 'cudf' backend."
        )
    if dtype_backend is not None:
        raise NotImplementedError(
            "dtype_backend is not supported by the 'cudf' backend."
        )
    if arrow_to_pandas is not None:
        raise NotImplementedError(
            "arrow_to_pandas is not supported by the 'cudf' backend."
        )
    if engine not in (None, "cudf", CudfEngine):
        raise NotImplementedError(
            "engine={engine} is not supported by the 'cudf' backend."
        )

    if not isinstance(path, str):
        path = stringify_path(path)

    kwargs["dtype_backend"] = None
    if arrow_to_pandas:
        kwargs["arrow_to_pandas"] = None

    if filters is not None:
        for filter in flatten(filters, container=list):
            _, op, val = filter
            if op == "in" and not isinstance(val, (set, list, tuple)):
                raise TypeError(
                    "Value of 'in' filter must be a list, set or tuple."
                )

    # Normalize blocksize input
    if blocksize == "default":
        blocksize = _normalize_blocksize()
    elif isinstance(blocksize, float) and blocksize < 1:
        blocksize = _normalize_blocksize(blocksize)

    if (
        isinstance(filesystem, pa_fs.FileSystem)
        or isinstance(filesystem, str)
        and filesystem.lower() in ("arrow", "pyarrow")
    ):
        # EXPERIMENTAL filesystem="arrow" support.
        # This code path may use PyArrow for remote IO.

        # CudfReadParquetPyarrowFS requires import of distributed beforehand
        # (See: https://github.com/dask/dask/issues/11352)
        import distributed  # noqa: F401

        if metadata_task_size is not None:
            warnings.warn(
                "metadata_task_size is not supported when using the pyarrow filesystem."
                " This argument will be ignored!"
            )
        if aggregate_files is not None:
            warnings.warn(
                "aggregate_files is not supported when using the pyarrow filesystem."
                " This argument will be ignored!"
            )
        if split_row_groups != "infer":
            warnings.warn(
                "split_row_groups is not supported when using the pyarrow filesystem."
                " This argument will be ignored!"
            )
        if parquet_file_extension != (".parq", ".parquet", ".pq"):
            raise NotImplementedError(
                "parquet_file_extension is not supported when using the pyarrow filesystem."
            )

        return new_collection(
            NoOp(
                CudfReadParquetPyarrowFS(
                    path,
                    columns=_convert_to_list(columns),
                    filters=filters,
                    categories=categories,
                    index=index,
                    calculate_divisions=calculate_divisions,
                    storage_options=storage_options,
                    filesystem=filesystem,
                    blocksize=blocksize,
                    ignore_metadata_file=ignore_metadata_file,
                    arrow_to_pandas=None,
                    pyarrow_strings_enabled=pyarrow_strings_enabled(),
                    kwargs=kwargs,
                    _series=isinstance(columns, str),
                ),
            )
        )

    return new_collection(
        NoOp(
            CudfReadParquetFSSpec(
                path,
                columns=_convert_to_list(columns),
                filters=filters,
                categories=categories,
                index=index,
                blocksize=blocksize,
                storage_options=storage_options,
                calculate_divisions=calculate_divisions,
                ignore_metadata_file=ignore_metadata_file,
                metadata_task_size=metadata_task_size,
                split_row_groups=split_row_groups,
                aggregate_files=aggregate_files,
                parquet_file_extension=parquet_file_extension,
                filesystem=filesystem,
                engine=CudfEngine,
                kwargs=kwargs,
                _series=isinstance(columns, str),
            ),
        )
    )


read_parquet = read_parquet_expr
read_parquet.__doc__ = read_parquet_expr.__doc__
to_parquet = _deprecated_api(
    "dask_cudf.io.parquet.to_parquet",
    new_api="dask_cudf._legacy.io.parquet.to_parquet",
    rec="Please use the DataFrame.to_parquet method instead.",
)
create_metadata_file = _deprecated_api(
    "dask_cudf.io.parquet.create_metadata_file",
    new_api="dask_cudf._legacy.io.parquet.create_metadata_file",
    rec="Please raise an issue if this feature is needed.",
)
