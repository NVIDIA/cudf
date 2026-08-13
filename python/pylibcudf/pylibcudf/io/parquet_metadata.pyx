# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.datasource cimport datasource, make_datasources
from pylibcudf.libcudf.io.hybrid_scan cimport (
    const_uint8_t,
    hybrid_scan_reader as cpp_hybrid_scan_reader,
)
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.io cimport parquet_metadata as cpp_parquet_metadata
from pylibcudf.libcudf.io.parquet_schema cimport (
    ColumnChunk as cpp_ColumnChunk,
    ColumnChunkMetaData as cpp_ColumnChunkMetaData,
    FileMetaData as cpp_FileMetaData,
    RowGroup as cpp_RowGroup,
    SortingColumn as cpp_SortingColumn,
)
from pylibcudf.libcudf.utilities.span cimport host_span
from pylibcudf.types cimport DataType

ctypedef const unique_ptr[datasource] const_unique_ptr_datasource


__all__ = [
    "ColumnChunk",
    "ColumnChunkMetaData",
    "FileMetaData",
    "ParquetColumnSchema",
    "ParquetMetadata",
    "ParquetSchema",
    "RowGroup",
    "SortingColumn",
    "read_parquet_footers",
    "read_parquet_metadata",
]

cdef class ParquetColumnSchema:
    """
    Schema of a parquet column, including the nested columns.

    Parameters
    ----------
    parquet_column_schema
    """
    def __init__(self):
        raise ValueError("Construct ParquetColumnSchema with from_column_schema.")

    @staticmethod
    cdef from_column_schema(cpp_parquet_metadata.parquet_column_schema column_schema):
        cdef ParquetColumnSchema result = ParquetColumnSchema.__new__(
            ParquetColumnSchema
        )
        result.column_schema = column_schema
        return result

    cpdef str name(self):
        """
        Returns parquet column name; can be empty.

        Returns
        -------
        str
            Column name
        """
        return self.column_schema.name().decode()

    cpdef int num_children(self):
        """
        Returns the number of child columns.

        Returns
        -------
        int
            Children count
        """
        return self.column_schema.num_children()

    cpdef ParquetColumnSchema child(self, int idx):
        """
        Returns schema of the child with the given index.

        Parameters
        ----------
        idx : int
            Child Index

        Returns
        -------
        ParquetColumnSchema
            Child schema
        """
        return ParquetColumnSchema.from_column_schema(self.column_schema.child(idx))

    cpdef list children(self):
        """
        Returns schemas of all child columns.

        Returns
        -------
        list[ParquetColumnSchema]
            Child schemas.
        """
        cdef cpp_parquet_metadata.parquet_column_schema child
        return [
            ParquetColumnSchema.from_column_schema(child)
            for child in self.column_schema.children()
        ]

    cpdef DataType cudf_type(self):
        """
        Returns the cudf data type for this column.

        This is the resolved cudf data type mapped from the parquet
        physical/logical types.

        Returns
        -------
        DataType
            cudf data type
        """
        return DataType.from_libcudf(self.column_schema.cudf_type())


cdef class ParquetSchema:
    """
    Schema of a parquet file.

    Parameters
    ----------
    parquet_schema
    """

    def __init__(self):
        raise ValueError("Construct ParquetSchema with from_schema.")

    @staticmethod
    cdef from_schema(cpp_parquet_metadata.parquet_schema schema):
        cdef ParquetSchema result = ParquetSchema.__new__(ParquetSchema)
        result.schema = schema
        return result

    cpdef ParquetColumnSchema root(self):
        """
        Returns the schema of the struct column that contains all columns as fields.

        Returns
        -------
        ParquetColumnSchema
            Root column schema
        """
        return ParquetColumnSchema.from_column_schema(self.schema.root())

    cpdef dict column_types(self):
        """
        Returns a dictionary mapping column names to their cudf data types.

        Returns
        -------
        dict[str, DataType]
            Dictionary mapping column names to DataType objects
        """
        cdef ParquetColumnSchema root_schema = self.root()
        return {
            root_schema.child(i).name(): root_schema.child(i).cudf_type()
            for i in range(root_schema.num_children())
        }


cdef class ParquetMetadata:
    """
    Information about content of a parquet file.

    Parameters
    ----------
    parquet_metadata
    """

    def __init__(self):
        raise ValueError("Construct ParquetMetadata with from_metadata.")

    @staticmethod
    cdef from_metadata(cpp_parquet_metadata.parquet_metadata meta):
        cdef ParquetMetadata result = ParquetMetadata.__new__(ParquetMetadata)
        result.meta = meta
        return result

    cpdef ParquetSchema schema(self):
        """
        Returns the parquet schema.

        Returns
        -------
        ParquetSchema
            Parquet schema
        """
        return ParquetSchema.from_schema(self.meta.schema())

    cpdef int num_rows(self):
        """
        Returns the number of rows of the root column.

        Returns
        -------
        int
            Number of rows
        """
        return self.meta.num_rows()

    cpdef int num_rowgroups(self):
        """
        Returns the total number of rowgroups in the file.

        Returns
        -------
        int
            Number of row groups.
        """
        return self.meta.num_rowgroups()

    cpdef list num_rowgroups_per_file(self):
        """
        Returns the number of rowgroups in each file.
        """
        return self.meta.num_rowgroups_per_file()

    cpdef dict metadata(self):
        """
        Returns the key-value metadata in the file footer.

        Returns
        -------
        dict[str, str]
            Key value metadata as a map.
        """
        return {key.decode(): val.decode() for key, val in self.meta.metadata()}

    cpdef list rowgroup_metadata(self):
        """
        Returns the row group metadata in the file footer.

        Returns
        -------
        list[dict[str, int]]
            Vector of row group metadata as maps.
        """
        return [
            {key.decode(): val for key, val in metadata}
            for metadata in self.meta.rowgroup_metadata()
        ]

    cpdef dict columnchunk_metadata(self):
        """
        Returns a map of leaf column names to lists of `total_uncompressed_size`
        metadata from all column chunks in the file footer.

        Returns
        -------
        dict[str, list[int]]
            Map of leaf column names to lists of `total_uncompressed_size` metadata
            from all their column chunks.
        """
        return {
            col_name.decode(): uncompressed_sizes
            for col_name, uncompressed_sizes in self.meta.columnchunk_metadata()
        }


cdef class SortingColumn:
    """Sort metadata for a row group column."""

    def __init__(self):
        raise ValueError("SortingColumn cannot be constructed directly")

    @staticmethod
    cdef SortingColumn from_cpp(cpp_SortingColumn sorting_column):
        cdef SortingColumn result = SortingColumn.__new__(SortingColumn)
        result.c_obj = sorting_column
        return result

    @property
    def column_idx(self):
        """Column index (within the row group)."""
        return self.c_obj.column_idx

    @property
    def descending(self):
        """Whether this column is sorted in descending order."""
        return self.c_obj.descending

    @property
    def nulls_first(self):
        """Whether null values are ordered before non-null values."""
        return self.c_obj.nulls_first


cdef class ColumnChunk:
    """Metadata for a row group's column chunk."""

    def __init__(self):
        raise ValueError("ColumnChunk cannot be constructed directly")

    @staticmethod
    cdef ColumnChunk from_cpp(cpp_ColumnChunk column_chunk):
        cdef ColumnChunk result = ColumnChunk.__new__(ColumnChunk)
        result.c_obj = column_chunk
        return result

    @property
    def file_path(self):
        """Relative file path for this column chunk."""
        return self.c_obj.file_path.decode("utf-8")

    @property
    def file_offset(self):
        """Deprecated byte offset to column metadata."""
        return self.c_obj.file_offset

    @property
    def offset_index_offset(self):
        """File offset of the chunk's OffsetIndex."""
        return self.c_obj.offset_index_offset

    @property
    def offset_index_length(self):
        """Size of the chunk's OffsetIndex, in bytes."""
        return self.c_obj.offset_index_length

    @property
    def column_index_offset(self):
        """File offset of the chunk's ColumnIndex."""
        return self.c_obj.column_index_offset

    @property
    def column_index_length(self):
        """Size of the chunk's ColumnIndex, in bytes."""
        return self.c_obj.column_index_length

    @property
    def schema_idx(self):
        """Derived index in the flattened schema."""
        return self.c_obj.schema_idx

    @property
    def meta_data(self):
        """Column metadata for this chunk."""
        return ColumnChunkMetaData.from_cpp(self.c_obj.meta_data)


cdef class ColumnChunkMetaData:
    """Metadata payload for a column chunk."""

    def __init__(self):
        raise ValueError("ColumnChunkMetaData cannot be constructed directly")

    @staticmethod
    cdef ColumnChunkMetaData from_cpp(cpp_ColumnChunkMetaData meta_data):
        cdef ColumnChunkMetaData result = ColumnChunkMetaData.__new__(
            ColumnChunkMetaData
        )
        result.c_obj = meta_data
        return result

    @property
    def path_in_schema(self):
        """Column path components in the flattened schema."""
        cdef string path
        return [path.decode("utf-8") for path in self.c_obj.path_in_schema]

    @property
    def num_values(self):
        """Number of values in this chunk."""
        return self.c_obj.num_values

    @property
    def total_uncompressed_size(self):
        """Total uncompressed page bytes for this chunk."""
        return self.c_obj.total_uncompressed_size

    @property
    def total_compressed_size(self):
        """Total compressed page bytes for this chunk."""
        return self.c_obj.total_compressed_size


cdef class RowGroup:
    """Parquet row group metadata."""

    def __init__(self):
        raise ValueError("RowGroup cannot be constructed directly")

    @staticmethod
    cdef RowGroup from_cpp(cpp_RowGroup row_group):
        cdef RowGroup result = RowGroup.__new__(RowGroup)
        result.c_obj = row_group
        return result

    @property
    def columns(self):
        """Column chunk metadata for each column in this row group."""
        cdef cpp_ColumnChunk column_chunk
        return [
            ColumnChunk.from_cpp(column_chunk) for column_chunk in self.c_obj.columns
        ]

    @property
    def total_byte_size(self):
        """Total uncompressed byte size in this row group."""
        return self.c_obj.total_byte_size

    @property
    def num_rows(self):
        """Number of rows in this row group."""
        return self.c_obj.num_rows

    @property
    def sorting_columns(self):
        """Optional row sort order metadata."""
        cdef cpp_SortingColumn sorting_column
        if not self.c_obj.sorting_columns.has_value():
            return None
        return [
            SortingColumn.from_cpp(sorting_column)
            for sorting_column in self.c_obj.sorting_columns.value()
        ]

    @property
    def file_offset(self):
        """Optional byte offset to first page in this row group."""
        if not self.c_obj.file_offset.has_value():
            return None
        return self.c_obj.file_offset.value()

    @property
    def total_compressed_size(self):
        """Optional total compressed bytes for this row group."""
        if not self.c_obj.total_compressed_size.has_value():
            return None
        return self.c_obj.total_compressed_size.value()

    @property
    def ordinal(self):
        """Optional row group ordinal within the file."""
        if not self.c_obj.ordinal.has_value():
            return None
        return self.c_obj.ordinal.value()


cdef class FileMetaData:
    """Parquet file footer metadata.

    For details, see :cpp:class:`cudf::io::parquet::FileMetaData`

    See Also
    --------
    read_parquet_footers
        Read one ``FileMetaData`` per source directly from
        :class:`pylibcudf.io.types.SourceInfo`.
    """

    def __init__(self):
        raise ValueError("FileMetaData cannot be constructed directly")

    @staticmethod
    cdef FileMetaData from_cpp(cpp_FileMetaData metadata):
        cdef FileMetaData result = FileMetaData.__new__(FileMetaData)
        result.c_obj = metadata
        return result

    @property
    def version(self):
        """Get the file format version."""
        return self.c_obj.version

    @property
    def num_rows(self):
        """Get the total number of rows."""
        return self.c_obj.num_rows

    @property
    def created_by(self):
        """Get the application that created the file."""
        return self.c_obj.created_by.decode("utf-8")

    @property
    def row_groups(self):
        """Get row group metadata in this file."""
        cdef cpp_RowGroup row_group
        return [RowGroup.from_cpp(row_group) for row_group in self.c_obj.row_groups]

    @property
    def row_group_num_rows(self):
        """
        Get row counts for each row group in this file.

        Returns
        -------
        list
            A list with the row count per row group in this file.

        Notes
        -----
        Equivalent to, but faster than, checking each row groups' num_rows:

        .. code-block:: python

           >>> [rg.num_rows for rg in file_metadata.row_groups]
        """
        cdef Py_ssize_t i
        cdef Py_ssize_t n = self.c_obj.row_groups.size()
        return [self.c_obj.row_groups[i].num_rows for i in range(n)]

    @property
    def columnchunk_metadata(self):
        """
        Get a map of dotted column paths to lists of
        `total_uncompressed_size` values from every column chunk in
        this file.

        Returns
        -------
        dict[str, list[int]]
            Map of dotted column paths (``".".join(path_in_schema)``)
            to lists of `total_uncompressed_size` metadata from all
            their column chunks.

        Notes
        -----
        Equivalent to, but faster than, walking each row group's columns:

        .. code-block:: python

           >>> result: dict[str, list[int]] = {}
           >>> for rg in file_metadata.row_groups:
           ...     for col in rg.columns:
           ...         name = ".".join(col.meta_data.path_in_schema)
           ...         result.setdefault(name, []).append(
           ...             col.meta_data.total_uncompressed_size
           ...         )
        """
        cdef Py_ssize_t i, j, k, n_path, n_col
        cdef Py_ssize_t n_rg = self.c_obj.row_groups.size()
        cdef dict result = {}
        cdef str name
        cdef list path_parts
        for i in range(n_rg):
            n_col = self.c_obj.row_groups[i].columns.size()
            for j in range(n_col):
                n_path = self.c_obj.row_groups[i].columns[j].meta_data.path_in_schema.size()
                path_parts = [
                    self.c_obj.row_groups[i].columns[j].meta_data.path_in_schema[k].decode("utf-8")
                    for k in range(n_path)
                ]
                name = ".".join(path_parts)
                result.setdefault(name, []).append(
                    self.c_obj.row_groups[i].columns[j].meta_data.total_uncompressed_size
                )
        return result

    @classmethod
    def from_bytes(cls, const uint8_t[::1] footer_bytes):
        """Build ``FileMetaData`` from parquet footer bytes.

        Parameters
        ----------
        footer_bytes : Buffer
            A contiguous bytes-like object containing parquet footer bytes.
            The bytes are forwarded as-is to
            :cpp:class:`cudf::io::parquet::experimental::hybrid_scan_reader`
            without Python-side preprocessing. This method does not strip the
            parquet footer suffix (4-byte footer length + ``PAR1`` magic), so
            callers should generally pass only the footer region bytes.

        Returns
        -------
        FileMetaData
            Parsed parquet file footer metadata.
        """
        cdef parquet_reader_options options = parquet_reader_options()
        cdef unique_ptr[cpp_hybrid_scan_reader] reader
        cdef cpp_FileMetaData metadata
        cdef const uint8_t* footer_ptr = <const uint8_t*>0

        if len(footer_bytes) > 0:
            footer_ptr = &footer_bytes[0]

        with nogil:
            reader = make_unique[cpp_hybrid_scan_reader](
                host_span[const_uint8_t](footer_ptr, len(footer_bytes)),
                options,
            )
            metadata = reader.get()[0].parquet_metadata()

        return FileMetaData.from_cpp(metadata)


cpdef ParquetMetadata read_parquet_metadata(SourceInfo src_info):
    """
    Reads metadata of parquet dataset.

    Parameters
    ----------
    src_info : SourceInfo
        Dataset source.

    Returns
    -------
    ParquetMetadata
        Parquet_metadata with parquet schema, number of rows,
        number of row groups and key-value metadata.

    See Also
    --------
    read_parquet_footers
        To read the pre-materialized file footer metadata used
        in :func:`pylibcudf.io.parquet.read_parquet`.
    """
    cdef cpp_parquet_metadata.parquet_metadata c_result

    with nogil:
        c_result = cpp_parquet_metadata.read_parquet_metadata(src_info.c_obj)

    return ParquetMetadata.from_metadata(c_result)


cpdef list read_parquet_footers(SourceInfo src_info):
    """
    Read parquet file footers as ``FileMetaData`` objects.

    Parameters
    ----------
    src_info : SourceInfo
        Dataset source.

    Returns
    -------
    list[FileMetaData]
        One footer metadata object per input source.
    """
    cdef vector[unique_ptr[datasource]] sources
    cdef vector[cpp_FileMetaData] c_result
    cdef cpp_FileMetaData metadata
    with nogil:
        sources = make_datasources(src_info.c_obj)
        c_result = cpp_parquet_metadata.read_parquet_footers(
            host_span[const_unique_ptr_datasource](
                <const_unique_ptr_datasource*>sources.data(),
                sources.size(),
            )
        )

    return [FileMetaData.from_cpp(metadata) for metadata in c_result]
