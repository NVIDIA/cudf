/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "multi_host_buffer_source.hpp"

#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>

#include <vector>

extern "C" {

/**
 * @brief Convert serialized bitmaps from jlong array to vector of byte spans
 *
 * The input jlong array contains pairs of (address, size) for each serialized bitmap.
 * This function converts that into a vector of cudf::host_span<cuda::std::byte const>.
 */
std::vector<cudf::host_span<cuda::std::byte const>> to_vector_of_spans(
  cudf::jni::native_jlongArray const& n_serialized_roaring64)
{
  if (n_serialized_roaring64.size() % 2 != 0) {
    throw std::logic_error("n_serialized_roaring64 length not a multiple of 2");
  }
  auto const num_bitmaps = n_serialized_roaring64.size() / 2;

  std::vector<cudf::host_span<cuda::std::byte const>> serialized_bitmaps;
  serialized_bitmaps.reserve(num_bitmaps);
  // transform n_serialized_roaring64 to vector of byte spans
  for (int i = 0; i < n_serialized_roaring64.size(); i += 2) {
    auto bitmap_addr = n_serialized_roaring64[i];
    auto bitmap_size = n_serialized_roaring64[i + 1];
    serialized_bitmaps.emplace_back(reinterpret_cast<cuda::std::byte const*>(bitmap_addr),
                                    static_cast<std::size_t>(bitmap_size));
  }
  return serialized_bitmaps;
}

cudf::io::parquet_reader_options make_parquet_reader_options(JNIEnv* env,
                                                             jobjectArray const& filter_col_names,
                                                             jbooleanArray const& col_binary_read,
                                                             jobjectArray const& row_groups,
                                                             cudf::io::source_info&& source,
                                                             jint unit)
{
  cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
  cudf::jni::native_jobjectArray<jintArray> n_row_groups(env, row_groups);

  // TODO: This variable is unused now, but we still don't know what to do with it yet.
  // As such, it needs to stay here for a little more time before we decide to use it again,
  // or remove it completely.
  cudf::jni::native_jbooleanArray n_col_binary_read(env, col_binary_read);
  (void)n_col_binary_read;

  auto builder = cudf::io::parquet_reader_options::builder(source);
  if (n_filter_col_names.size() > 0) {
    builder = builder.column_names(n_filter_col_names.as_cpp_vector());
  }
  if (n_row_groups.size() > 0) {
    auto row_groups_vec = std::vector<std::vector<cudf::size_type>>{};
    for (int i = 0; i < n_row_groups.size(); i++) {
      cudf::jni::native_jintArray n_row_group(env, n_row_groups.get(i));
      row_groups_vec.emplace_back(n_row_group.to_vector());
    }
    builder = builder.row_groups(row_groups_vec);
  }

  return builder.convert_strings_to_categories(false)
    .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
    // Ignore any missing projected column(s) by default
    .ignore_missing_columns(true)
    .build();
}

std::unique_ptr<cudf::io::parquet::experimental::deletion_vector_info> make_deletion_vector_info(
  JNIEnv* env,
  jlongArray const& serialized_roaring64,
  jintArray const& deletion_vector_row_counts,
  jlongArray const& row_group_offsets,
  jintArray const& row_group_num_rows,
  jboolean are_retention_vectors)
{
  cudf::jni::native_jlongArray n_serialized_roaring64(env, serialized_roaring64);
  std::vector<cudf::host_span<cuda::std::byte const>> serialized_bitmaps =
    to_vector_of_spans(n_serialized_roaring64);

  cudf::jni::native_jintArray n_deletion_vector_row_counts(env, deletion_vector_row_counts);
  cudf::jni::native_jlongArray n_row_group_offsets(env, row_group_offsets);
  cudf::jni::native_jintArray n_row_group_num_rows(env, row_group_num_rows);

  auto dv_info = std::make_unique<cudf::io::parquet::experimental::deletion_vector_info>();
  dv_info->serialized_roaring_bitmaps = std::move(serialized_bitmaps);
  dv_info->deletion_vector_row_counts = n_deletion_vector_row_counts.to_vector();
  dv_info->row_group_num_rows         = n_row_group_num_rows.to_vector();
  dv_info->row_group_offsets.reserve(n_row_group_offsets.size());
  dv_info->are_retention_vectors = are_retention_vectors;
  std::transform(n_row_group_offsets.begin(),
                 n_row_group_offsets.end(),
                 std::back_inserter(dv_info->row_group_offsets),
                 [](jlong v) { return static_cast<size_t>(v); });
  return dv_info;
}

/**
 * @brief Read a Parquet file with deletion vector support
 *
 * This JNI function wraps cudf::io::parquet::experimental::read_parquet to read a Parquet file,
 * prepend an index column, and apply a deletion vector filter.
 *
 * @param env JNI environment
 * @param filter_col_names Column names to filter
 * @param col_binary_read Boolean array indicating binary read for string columns
 * @param input_file_paths Input file paths (if reading from files)
 * @param addrs_and_sizes Address and size pairs for buffer reading (if reading from buffer)
 * @param row_groups Row group indices to read
 * @param unit Timestamp unit
 * @param serialized_roaring64 Serialized 64-bit roaring bitmaps from deletion vectors
 * @param deletion_vector_row_counts Number of rows read from data files associated to each deletion
 * vector
 * @param row_group_offsets Row offsets for each row group.
 *                            When reading multiple files, the offsets are stored in order of files,
 *                            i.e., all row groups from file 1 followed by all row groups from file
 * 2, etc.
 * @param row_group_num_rows Number of rows in each row group.
 *                             When reading multiple files, the row counts are stored in order of
 * files, i.e., all row groups from file 1 followed by all row groups from file 2, etc.
 * @return Handle to the columns of the resulting table (as jlongArray)
 */
JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_DeletionVector_readParquet(JNIEnv* env,
                                               jclass,
                                               jobjectArray filter_col_names,
                                               jbooleanArray col_binary_read,
                                               jobjectArray input_file_paths,
                                               jlongArray addrs_and_sizes,
                                               jobjectArray row_groups,
                                               jint unit,
                                               jlongArray serialized_roaring64,
                                               jintArray deletion_vector_row_counts,
                                               jlongArray row_group_offsets,
                                               jintArray row_group_num_rows,
                                               jboolean are_retention_vectors)
{
  bool read_buffer = true;
  if (addrs_and_sizes == nullptr) {
    JNI_NULL_CHECK(env, input_file_paths, "input file or buffer must be supplied", nullptr);
    read_buffer = false;
  } else if (input_file_paths != nullptr) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "cannot pass in both a buffer and an input_file_paths",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstringArray filenames(env, input_file_paths);
    if (!read_buffer && filenames.size() == 0) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "input_file_paths can't be empty", nullptr);
    }

    std::unique_ptr<cudf::jni::multi_host_buffer_source> multi_buffer_source;
    if (read_buffer) {
      cudf::jni::native_jlongArray n_addrs_sizes(env, addrs_and_sizes);
      multi_buffer_source.reset(new cudf::jni::multi_host_buffer_source(n_addrs_sizes));
    }
    cudf::io::source_info source = read_buffer ? cudf::io::source_info(multi_buffer_source.get())
                                               : cudf::io::source_info(filenames.as_cpp_vector());

    cudf::io::parquet_reader_options opts = make_parquet_reader_options(
      env, filter_col_names, col_binary_read, row_groups, std::move(source), unit);

    auto dv_info = make_deletion_vector_info(env,
                                             serialized_roaring64,
                                             deletion_vector_row_counts,
                                             row_group_offsets,
                                             row_group_num_rows,
                                             are_retention_vectors);

    auto tbl = cudf::io::parquet::experimental::read_parquet(opts, *dv_info).tbl;
    return cudf::jni::convert_table_for_return(env, tbl);
  }
  JNI_CATCH(env, nullptr);
}

/**
 * @brief Create a chunked Parquet reader with multiple deletion vectors and pass read limit
 *
 * This JNI function creates a chunked_parquet_reader with both chunk and pass read limits
 * and multiple deletion vectors.
 *
 * @param env JNI environment
 * @param chunk_read_limit Byte limit on returned table chunk size, 0 if no limit.
 *                         Actual size of the returned table can be smaller than this limit.
 * @param pass_read_limit Byte limit hint on decompression memory, 0 if no limit. See cudf docs
 *                         for details.
 * @param filter_col_names Column names to filter
 * @param col_binary_read Boolean array indicating binary read for string columns
 * @param input_file_paths Input file paths (if reading from files)
 * @param addrs_sizes Address and size pairs for buffer reading (if reading from buffer)
 * @param row_groups Row group indices to read
 * @param unit Timestamp unit
 * @param serialized_roaring_bitmaps Array of serialized 64-bit roaring bitmaps from deletion
 * vectors
 * @param deletion_vector_row_counts Number of rows read from data files associated to each deletion
 * vector
 * @param row_group_offsets Row offsets for each row group.
 *                            When reading multiple files, the offsets are stored in order of files,
 *                            i.e., all row groups from file 1 followed by all row groups from file
 * 2, etc.
 * @param row_group_num_rows Number of rows in each row group.
 *                             When reading multiple files, the row counts are stored in order of
 * files, i.e., all row groups from file 1 followed by all row groups from file 2, etc.
 * @return jlongArray of size 3:
 *         [0] Handle to the chunked_parquet_reader
 *         [1] Handle to the multi_host_buffer_source (nullptr if reading from file)
 *         [2] Handle to the deletion_vector_info
 */
JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_DeletionVector_createParquetChunkedReader(JNIEnv* env,
                                                              jclass,
                                                              jlong chunk_read_limit,
                                                              jlong pass_read_limit,
                                                              jobjectArray filter_col_names,
                                                              jbooleanArray col_binary_read,
                                                              jobjectArray input_file_paths,
                                                              jlongArray addrs_sizes,
                                                              jobjectArray row_groups,
                                                              jint unit,
                                                              jlongArray serialized_roaring64,
                                                              jintArray deletion_vector_row_counts,
                                                              jlongArray row_group_offsets,
                                                              jintArray row_group_num_rows,
                                                              jboolean are_retention_vectors)
{
  bool read_buffer = true;
  if (addrs_sizes == nullptr) {
    JNI_NULL_CHECK(env, input_file_paths, "Input file or buffer must be supplied", nullptr);
    read_buffer = false;
  } else if (input_file_paths != nullptr) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "Cannot pass in both buffers and an input_file_paths",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstringArray filenames(env, input_file_paths);
    if (!read_buffer && filenames.size() == 0) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "input_file_paths cannot be empty", nullptr);
    }

    std::unique_ptr<cudf::jni::multi_host_buffer_source> multi_buffer_source;
    if (read_buffer) {
      cudf::jni::native_jlongArray n_addrs_sizes(env, addrs_sizes);
      multi_buffer_source.reset(new cudf::jni::multi_host_buffer_source(n_addrs_sizes));
    }
    cudf::io::source_info source = read_buffer ? cudf::io::source_info(multi_buffer_source.get())
                                               : cudf::io::source_info(filenames.as_cpp_vector());

    cudf::io::parquet_reader_options opts = make_parquet_reader_options(
      env, filter_col_names, col_binary_read, row_groups, std::move(source), unit);

    auto dv_info = make_deletion_vector_info(env,
                                             serialized_roaring64,
                                             deletion_vector_row_counts,
                                             row_group_offsets,
                                             row_group_num_rows,
                                             are_retention_vectors);

    // Create the chunked reader with pass read limit and multiple deletion vectors
    auto reader = new cudf::io::parquet::experimental::chunked_parquet_reader(
      static_cast<std::size_t>(chunk_read_limit),
      static_cast<std::size_t>(pass_read_limit),
      opts,
      *dv_info);

    auto reader_handle = reinterpret_cast<jlong>(reader);
    cudf::jni::native_jlongArray result(env, 3);
    result[0] = reader_handle;
    result[1] = cudf::jni::release_as_jlong(multi_buffer_source);
    result[2] = cudf::jni::release_as_jlong(dv_info);
    return result.get_jArray();
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Check if the chunked reader has more data to read
 *
 * @param env JNI environment
 * @param j_reader_handle Handle to the chunked_parquet_reader
 * @return true if there is more data to read, false otherwise
 */
JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_DeletionVector_parquetChunkedReaderHasNext(
  JNIEnv* env, jclass, jlong j_reader_handle)
{
  JNI_NULL_CHECK(env, j_reader_handle, "reader handle is null", false);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader =
      reinterpret_cast<cudf::io::parquet::experimental::chunked_parquet_reader* const>(
        j_reader_handle);
    return reader->has_next();
  }
  JNI_CATCH(env, false);
}

/**
 * @brief Read the next chunk from the chunked reader
 *
 * @param env JNI environment
 * @param j_reader_handle Handle to the chunked_parquet_reader
 * @return Handle to the resulting table (as jlongArray)
 */
JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_DeletionVector_parquetChunkedReaderReadChunk(
  JNIEnv* env, jclass, jlong j_reader_handle)
{
  JNI_NULL_CHECK(env, j_reader_handle, "reader handle is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader =
      reinterpret_cast<cudf::io::parquet::experimental::chunked_parquet_reader* const>(
        j_reader_handle);
    auto chunk = reader->read_chunk();
    return chunk.tbl ? cudf::jni::convert_table_for_return(env, chunk.tbl) : nullptr;
  }
  JNI_CATCH(env, nullptr);
}

/**
 * @brief Close and destroy the chunked reader
 *
 * @param env JNI environment
 * @param j_reader_handle Handle to the chunked_parquet_reader
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_DeletionVector_closeParquetChunkedReader(
  JNIEnv* env, jclass, jlong j_reader_handle)
{
  JNI_NULL_CHECK(env, j_reader_handle, "reader handle is null", );

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::io::parquet::experimental::chunked_parquet_reader*>(
      j_reader_handle);
  }
  JNI_CATCH(env, );
}

/**
 * @brief Destroy the multi_host_buffer_source
 *
 * @param env JNI environment
 * @param handle Handle to the multi_host_buffer_source
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_DeletionVector_destroyMultiHostBufferSource(JNIEnv* env,
                                                                                       jclass,
                                                                                       jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );

  JNI_TRY { delete reinterpret_cast<cudf::jni::multi_host_buffer_source*>(handle); }
  JNI_CATCH(env, );
}

/**
 * @brief Destroy the deletion_vector_info
 *
 * @param env JNI environment
 * @param handle Handle to the deletion_vector_info
 */
JNIEXPORT void JNICALL Java_ai_rapids_cudf_DeletionVector_destroyDeletionVectorParam(JNIEnv* env,
                                                                                     jclass,
                                                                                     jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );

  JNI_TRY
  {
    delete reinterpret_cast<cudf::io::parquet::experimental::deletion_vector_info*>(handle);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
