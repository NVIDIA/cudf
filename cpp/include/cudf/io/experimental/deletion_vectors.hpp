/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/roaring_bitmap.hpp>

#include <queue>

/**
 * @file
 * @brief APIs for applying Delta Lake style deletion vectors when reading Parquet data.
 */

namespace CUDF_EXPORT cudf {
namespace io::parquet::experimental {

/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief Struct used to specify information about deletion vectors and the index column to the
 * experimental parquet reader
 */
struct deletion_vector_info {
  // Following vectors specify the data spans of input deletion vectors and the number of rows
  // spanned by each deletion vector in order. Deletion vectors are applied in order of their
  // appearance in the vectors. These vectors if empty will result in no table filtration.

  /// Host spans of 64-bit roaring bitmaps serialized in `portable` format
  std::vector<cudf::host_span<cuda::std::byte const>> serialized_roaring_bitmaps;
  /// Number of rows spanned by each deletion vector
  std::vector<size_type> deletion_vector_row_counts;

  // Following vectors customize the row index column prepended to the read table from the Parquet
  // source(s). These vectors if empty will result in an index column that is a sequence from 0 to
  // the total number of rows in the table.

  /// Row index offset for each row group to be read from the Parquet source(s)
  std::vector<size_t> row_group_offsets;
  /// Number of rows in each row group to be read from the Parquet source(s)
  std::vector<size_type> row_group_num_rows;

  /// Whether the roaring bitmaps represent retention vectors
  bool are_retention_vectors = false;
};

/**
 * @brief The chunked parquet reader class to read a Parquet source iteratively in a series of
 * tables, chunk by chunk. Each chunk is prepended with a row index column built using the specified
 * row group offsets and row counts. The resultant table chunk is filtered using the supplied
 * serialized roaring64 bitmap deletion vector and returned
 *
 * This class is designed to address the reading issue when reading very large Parquet source such
 * that the row count exceeds the cudf column size limit or if there are device memory constraints.
 * By reading the source content by chunks using this class, each chunk is guaranteed to have its
 * sizes stay within the given limit. Note that the given memory limits do not account for the
 * device memory needed to deserialize and construct the roaring64 bitmap deletion vector that stays
 * alive throughout the the lifetime of the reader.
 */
class chunked_parquet_reader {
 public:
  /**
   * @brief Constructor for the chunked reader
   *
   * Requires the same arguments as the `cudf::io::parquet::experimental::read_parquet()`, and an
   * additional parameter to specify the size byte limit of the output table chunk produced.
   *
   * @param chunk_read_limit Byte limit on the returned table chunk size, `0` if there is no limit
   * @param options Parquet reader options
   * @param deletion_vector_info Information about the deletion vectors and the index column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    parquet_reader_options const& options,
    deletion_vector_info const& deletion_vector_info,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Constructor for the chunked reader
   *
   * Requires the same arguments as `cudf::io::parquet::experimental::read_parquet()`, with
   * additional parameters to specify the size byte limit of the output table chunk produced, and a
   * byte limit on the amount of temporary memory to use when reading. The `pass_read_limit` affects
   * how many row groups we can read at a time by limiting the amount of memory dedicated to
   * decompression space. The `pass_read_limit` is a hint, not an absolute limit - if a single row
   * group cannot fit within the limit given, it will still be loaded. Also note that the
   * `pass_read_limit` does not include the memory to deserialize and construct the roaring64 bitmap
   * deletion vectors that stay alive throughout the the lifetime of the reader.
   *
   * @param chunk_read_limit Byte limit on the returned table chunk size, `0` if there is no limit
   * @param pass_read_limit Byte limit on the amount of memory used for decompressing and decoding
   * data, `0` if there is no limit
   * @param options Parquet reader options
   * @param deletion_vector_info Information about the deletion vectors and the index column
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  chunked_parquet_reader(
    std::size_t chunk_read_limit,
    std::size_t pass_read_limit,
    parquet_reader_options const& options,
    deletion_vector_info const& deletion_vector_info,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Destructor, destroying the internal reader instance and the roaring bitmap deletion
   * vector
   */
  ~chunked_parquet_reader();

  /**
   * @brief Check if there is any data in the given source that has not yet been read
   *
   * @return Boolean value indicating if there is any data left to be read
   */
  [[nodiscard]] bool has_next() const;

  /**
   * @brief Read a chunk of table from the Parquet source, prepend an index column to it, and
   * filters the resultant table chunk using the 64-bit roaring bitmap deletion vector, if provided
   *
   * The sequence of returned tables, if concatenated by their order, guarantees to form a complete
   * dataset as reading the entire given source at once.
   *
   * An empty table will be returned if the given source is empty, or all the data in the source has
   * been read and returned by the previous calls.
   *
   * @return An output `cudf::table` along with its metadata
   */
  [[nodiscard]] table_with_metadata read_chunk();

 private:
  std::unique_ptr<cudf::io::chunked_parquet_reader> _reader;
  std::queue<size_t> _row_group_row_offsets;
  std::queue<size_type> _row_group_row_counts;
  std::queue<cudf::roaring_bitmap> _deletion_vectors;
  std::queue<size_type> _deletion_vector_row_counts;
  size_t _start_row;
  bool _is_unspecified_row_group_data;
  bool _are_retentions;
  rmm::cuda_stream_view _stream;
  rmm::device_async_resource_ref _mr;
  rmm::device_async_resource_ref _table_mr;
};

/**
 * @brief Reads a table from parquet source, prepends an index column to it, deserializes the
 * specified 64-bit roaring bitmap deletion vectors and applies them to the read table
 *
 * Reads a table from a parquet source, builds a row index column to the table using the specified
 * row group offsets and row counts and prepends it to the table, deserializes the specified
 * 64-bit roaring bitmap deletion vectors and applies them to the read table using the specified
 * deletion vector row counts. If the row group offsets and row counts are empty, the index column
 * is simply a sequence of UINT64 from 0 to the total number of rows in the table. If the serialized
 * roaring64 bitmap span is empty, the read table (prepended with the index column) is returned as
 * is.
 *
 * @ingroup io_readers
 *
 * @param options Parquet reader options
 * @param deletion_vector_info Information about the deletion vectors and the index column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate device memory of the returned table
 *
 * @return Read table with a prepended index column filtered using the deletion vector, along with
 * its metadata
 */
table_with_metadata read_parquet(
  parquet_reader_options const& options,
  deletion_vector_info const& deletion_vector_info,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Computes the number of rows deleted by the serialized 64-bit roaring bitmap deletion
 * vectors
 *
 * @param deletion_vector_info Information about the deletion vectors and the index column
 * @param max_chunk_rows Maximum number of rows to process at a time
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Number of rows deleted by the specified 64-bit roaring bitmap deletion vectors
 */
[[nodiscard]] size_t compute_num_deleted_rows(
  deletion_vector_info const& deletion_vector_info,
  cudf::size_type max_chunk_rows = std::numeric_limits<size_type>::max(),
  rmm::cuda_stream_view stream   = cudf::get_default_stream());

/** @} */  // end of group

}  // namespace io::parquet::experimental
}  // namespace CUDF_EXPORT cudf
