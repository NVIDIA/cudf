# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import itertools
import warnings
from functools import partial
from io import BufferedWriter, BytesIO, IOBase

import numpy as np
import pandas as pd
from pyarrow import dataset as pa_ds, parquet as pq

import dask.dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowDatasetEngine

try:
    from dask.dataframe.io.parquet import (
        create_metadata_file as create_metadata_file_dd,
    )
except ImportError:
    create_metadata_file_dd = None

import cudf
from cudf.core.column import as_column
from cudf.io import write_to_dataset
from cudf.io.parquet import _apply_post_filters, _normalize_filters
from cudf.utils import ioutils
from cudf.utils.dtypes import cudf_dtype_from_pa_type


class CudfEngine(ArrowDatasetEngine):
    @classmethod
    def _create_dd_meta(cls, dataset_info, **kwargs):
        # Start with pandas-version of meta
        meta_pd = super()._create_dd_meta(dataset_info, **kwargs)

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

        return meta_cudf

    @classmethod
    def multi_support(cls):
        # Assert that this class is CudfEngine
        # and that multi-part reading is supported
        return cls == CudfEngine

    @classmethod
    def _read_paths(
        cls,
        paths,
        fs,
        columns=None,
        row_groups=None,
        filters=None,
        partitions=None,
        partitioning=None,
        partition_keys=None,
        open_file_options=None,
        dataset_kwargs=None,
        **kwargs,
    ):
        # Simplify row_groups if all None
        if row_groups == [None for path in paths]:
            row_groups = None

        # Make sure we read in the columns needed for row-wise
        # filtering after IO. This means that one or more columns
        # will be dropped almost immediately after IO. However,
        # we do NEED these columns for accurate filtering.
        filters = _normalize_filters(filters)
        projected_columns = None
        if columns and filters:
            projected_columns = [c for c in columns if c is not None]
            columns = sorted(
                set(v[0] for v in itertools.chain.from_iterable(filters))
                | set(projected_columns)
            )

        dataset_kwargs = dataset_kwargs or {}
        if partitions:
            dataset_kwargs["partitioning"] = partitioning or "hive"

        # Use cudf to read in data
        try:
            df = cudf.read_parquet(
                paths,
                engine="cudf",
                columns=columns,
                row_groups=row_groups if row_groups else None,
                dataset_kwargs=dataset_kwargs,
                categorical_partitions=False,
                filesystem=fs,
                **kwargs,
            )
        except RuntimeError as err:
            # TODO: Remove try/except after null-schema issue is resolved
            # (See: https://github.com/NVIDIA/cudf/issues/12702)
            if len(paths) > 1:
                df = cudf.concat(
                    [
                        cudf.read_parquet(
                            path,
                            engine="cudf",
                            columns=columns,
                            row_groups=row_groups[i] if row_groups else None,
                            dataset_kwargs=dataset_kwargs,
                            categorical_partitions=False,
                            filesystem=fs,
                            **kwargs,
                        )
                        for i, path in enumerate(paths)
                    ]
                )
            else:
                raise err

        # Apply filters (if any are defined)
        df = _apply_post_filters(df, filters)

        if projected_columns:
            # Elements of `projected_columns` may now be in the index.
            # We must filter these names from our projection
            projected_columns = [
                col for col in projected_columns if col in df._column_names
            ]
            df = df[projected_columns]

        if partitions and partition_keys is None:
            # Use `HivePartitioning` by default
            ds = pa_ds.dataset(
                paths,
                filesystem=fs,
                **dataset_kwargs,
            )
            frag = next(ds.get_fragments())
            if frag:
                # Extract hive-partition keys, and make sure they
                # are ordered the same as they are in `partitions`
                raw_keys = pa_ds._get_partition_keys(frag.partition_expression)
                partition_keys = [
                    (hive_part.name, raw_keys[hive_part.name])
                    for hive_part in partitions
                ]

        if partition_keys:
            if partitions is None:
                raise ValueError("Must pass partition sets")

            for i, (name, index2) in enumerate(partition_keys):
                if len(partitions[i].keys):
                    # Build a categorical column from `codes` directly
                    # (since the category is often a larger dtype)
                    df[name] = cudf.CategoricalIndex.from_codes(
                        [partitions[i].keys.get_loc(index2)] * len(df),
                        categories=partitions[i].keys,
                        ordered=False,
                    )
                elif name not in df.columns:
                    # Add non-categorical partition column
                    df[name] = as_column(index2, length=len(df))

        return df

    @classmethod
    def read_partition(
        cls,
        fs,
        pieces,
        columns,
        index,
        categories=(),
        partitions=(),
        filters=None,
        partitioning=None,
        schema=None,
        open_file_options=None,
        **kwargs,
    ):
        if columns is not None:
            columns = [c for c in columns]
        if isinstance(index, list):
            columns += index

        dataset_kwargs = kwargs.get("dataset", {})
        partitioning = partitioning or dataset_kwargs.get("partitioning", None)
        if isinstance(partitioning, dict):
            partitioning = pa_ds.partitioning(**partitioning)

        # Check if we are actually selecting any columns
        read_columns = columns
        if schema and columns:
            ignored = set(schema.names) - set(columns)
            if not ignored:
                read_columns = None

        if not isinstance(pieces, list):
            pieces = [pieces]

        # Extract supported kwargs from `kwargs`
        read_kwargs = kwargs.get("read", {})
        read_kwargs.update(open_file_options or {})
        check_file_size = read_kwargs.pop("check_file_size", None)

        # Wrap reading logic in a `try` block so that we can
        # inform the user that the `read_parquet` partition
        # size is too large for the available memory
        try:
            # Assume multi-piece read
            paths = []
            rgs = []
            last_partition_keys = None
            dfs = []

            for i, piece in enumerate(pieces):
                (path, row_group, partition_keys) = piece
                row_group = None if row_group == [None] else row_group

                # File-size check to help "protect" users from change
                # to up-stream `split_row_groups` default. We only
                # check the file size if this partition corresponds
                # to a full file, and `check_file_size` is defined
                if check_file_size and len(pieces) == 1 and row_group is None:
                    file_size = fs.size(path)
                    if file_size > check_file_size:
                        warnings.warn(
                            f"A large parquet file ({file_size}B) is being "
                            f"used to create a DataFrame partition in "
                            f"read_parquet. This may cause out of memory "
                            f"exceptions in operations downstream. See the "
                            f"notes on split_row_groups in the read_parquet "
                            f"documentation. Setting split_row_groups "
                            f"explicitly will silence this warning."
                        )

                if i > 0 and partition_keys != last_partition_keys:
                    dfs.append(
                        cls._read_paths(
                            paths,
                            fs,
                            columns=read_columns,
                            row_groups=rgs if rgs else None,
                            filters=filters,
                            partitions=partitions,
                            partitioning=partitioning,
                            partition_keys=last_partition_keys,
                            dataset_kwargs=dataset_kwargs,
                            **read_kwargs,
                        )
                    )
                    paths = []
                    rgs = []
                    last_partition_keys = None
                paths.append(path)
                rgs.append(
                    [row_group]
                    if not isinstance(row_group, list)
                    and row_group is not None
                    else row_group
                )
                last_partition_keys = partition_keys

            dfs.append(
                cls._read_paths(
                    paths,
                    fs,
                    columns=read_columns,
                    row_groups=rgs if rgs else None,
                    filters=filters,
                    partitions=partitions,
                    partitioning=partitioning,
                    partition_keys=last_partition_keys,
                    dataset_kwargs=dataset_kwargs,
                    **read_kwargs,
                )
            )
            df = cudf.concat(dfs) if len(dfs) > 1 else dfs[0]

            # Re-set "object" dtypes align with pa schema
            set_object_dtypes_from_pa_schema(df, schema)

            if index and (index[0] in df.columns):
                df = df.set_index(index[0])
            elif index is False and df.index.names != [None]:
                # If index=False, we shouldn't have a named index
                df.reset_index(inplace=True)

        except MemoryError as err:
            raise MemoryError(
                "Parquet data was larger than the available GPU memory!\n\n"
                "See the notes on split_row_groups in the read_parquet "
                "documentation.\n\n"
                "Original Error: " + str(err)
            )
            raise err

        return df

    @staticmethod
    def write_partition(
        df,
        path,
        fs,
        filename,
        partition_on,
        return_metadata,
        fmd=None,
        compression="snappy",
        index_cols=None,
        **kwargs,
    ):
        preserve_index = False
        if len(index_cols) and set(index_cols).issubset(set(df.columns)):
            df.set_index(index_cols, drop=True, inplace=True)
            preserve_index = True
        if partition_on:
            md = write_to_dataset(
                df=df,
                root_path=path,
                compression=compression,
                filename=filename,
                partition_cols=partition_on,
                fs=fs,
                preserve_index=preserve_index,
                return_metadata=return_metadata,
                statistics=kwargs.get("statistics", "ROWGROUP"),
                int96_timestamps=kwargs.get("int96_timestamps", False),
                row_group_size_bytes=kwargs.get("row_group_size_bytes", None),
                row_group_size_rows=kwargs.get("row_group_size_rows", None),
                max_page_size_bytes=kwargs.get("max_page_size_bytes", None),
                max_page_size_rows=kwargs.get("max_page_size_rows", None),
                storage_options=kwargs.get("storage_options", None),
            )
        else:
            with (
                contextlib.nullcontext()
                if ioutils._is_local_filesystem(fs)
                else fs.open(fs.sep.join([path, filename]), mode="wb")
            ) as out_file:
                if out_file is None:
                    out_file = fs.sep.join([path, filename])
                elif not isinstance(out_file, IOBase):
                    out_file = BufferedWriter(out_file)
                md = df.to_parquet(
                    path=out_file,
                    engine=kwargs.get("engine", "cudf"),
                    index=kwargs.get("index", None),
                    partition_cols=kwargs.get("partition_cols", None),
                    partition_file_name=kwargs.get(
                        "partition_file_name", None
                    ),
                    partition_offsets=kwargs.get("partition_offsets", None),
                    statistics=kwargs.get("statistics", "ROWGROUP"),
                    int96_timestamps=kwargs.get("int96_timestamps", False),
                    row_group_size_bytes=kwargs.get(
                        "row_group_size_bytes", None
                    ),
                    row_group_size_rows=kwargs.get(
                        "row_group_size_rows", None
                    ),
                    storage_options=kwargs.get("storage_options", None),
                    metadata_file_path=filename if return_metadata else None,
                )
        # Return the schema needed to write the metadata
        if return_metadata:
            return [{"meta": md}]
        else:
            return []

    @staticmethod
    def write_metadata(parts, fmd, fs, path, append=False, **kwargs):
        if parts:
            # Aggregate metadata and write to _metadata file
            metadata_path = fs.sep.join([path, "_metadata"])
            _meta = []
            if append and fmd is not None:
                # Convert to bytes: <https://github.com/NVIDIA/cudf/issues/17177>
                if isinstance(fmd, pq.FileMetaData):
                    with BytesIO() as myio:
                        fmd.write_metadata_file(myio)
                        myio.seek(0)
                        fmd = np.frombuffer(myio.read(), dtype="uint8")
                _meta = [fmd]
            _meta.extend([parts[i][0]["meta"] for i in range(len(parts))])
            _meta = (
                cudf.io.merge_parquet_filemetadata(_meta)
                if len(_meta) > 1
                else _meta[0]
            )
            with fs.open(metadata_path, "wb") as fil:
                fil.write(memoryview(_meta))

    @classmethod
    def collect_file_metadata(cls, path, fs, file_path):
        with fs.open(path, "rb") as f:
            meta = pq.ParquetFile(f).metadata
        if file_path:
            meta.set_file_path(file_path)
        with BytesIO() as myio:
            meta.write_metadata_file(myio)
            myio.seek(0)
            meta = np.frombuffer(myio.read(), dtype="uint8")
        return meta

    @classmethod
    def aggregate_metadata(cls, meta_list, fs, out_path):
        meta = (
            cudf.io.merge_parquet_filemetadata(meta_list)
            if len(meta_list) > 1
            else meta_list[0]
        )
        if out_path:
            metadata_path = fs.sep.join([out_path, "_metadata"])
            with fs.open(metadata_path, "wb") as fil:
                fil.write(memoryview(meta))
            return None
        else:
            return meta


def set_object_dtypes_from_pa_schema(df, schema):
    # Simple utility to modify cudf DataFrame
    # "object" dtypes to agree with a specific
    # pyarrow schema.
    if schema:
        for col_name, col in df._data.items():
            if col_name in schema.names:
                typ = cudf_dtype_from_pa_type(schema.field(col_name).type)
                if not isinstance(
                    typ, (cudf.ListDtype, cudf.StructDtype)
                ) and isinstance(col, cudf.core.column.StringColumn):
                    df._data[col_name] = col.astype(typ)


to_parquet = dd.to_parquet

if create_metadata_file_dd is None:
    create_metadata_file = create_metadata_file_dd
else:
    create_metadata_file = partial(create_metadata_file_dd, engine=CudfEngine)
