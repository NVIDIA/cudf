/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "multi_host_buffer_source.hpp"

#include <cudf/column/column.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/parquet.hpp>

#include <memory>
#include <optional>

// This file is for the code related to chunked reader (Parquet, ORC, etc.).

extern "C" {

//
// Chunked Parquet reader JNI
//

// This function should take all the parameters that `Table.readParquet` takes,
// plus one more parameter `long chunkSizeByteLimit`.
JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_ParquetChunkedReader_create(JNIEnv* env,
                                                jclass,
                                                jlong chunk_read_limit,
                                                jlong pass_read_limit,
                                                jobjectArray filter_col_names,
                                                jbooleanArray j_col_binary_read,
                                                jstring inp_file_path,
                                                jlongArray addrs_sizes,
                                                jint unit)
{
  JNI_NULL_CHECK(env, j_col_binary_read, "Null col_binary_read", nullptr);
  bool read_buffer = true;
  if (addrs_sizes == nullptr) {
    JNI_NULL_CHECK(env, inp_file_path, "Input file or buffer must be supplied", nullptr);
    read_buffer = false;
  } else if (inp_file_path != nullptr) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "Cannot pass in both buffers and an inp_file_path",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inp_file_path);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inp_file_path cannot be empty", nullptr);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    // TODO: This variable is unused now, but we still don't know what to do with it yet.
    // As such, it needs to stay here for a little more time before we decide to use it again,
    // or remove it completely.
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);
    (void)n_col_binary_read;

    cudf::jni::native_jlongArray n_addrs_sizes(env, addrs_sizes);
    std::unique_ptr<cudf::io::datasource> multi_buffer_source;
    cudf::io::source_info source;
    if (read_buffer) {
      multi_buffer_source.reset(new cudf::jni::multi_host_buffer_source(n_addrs_sizes));
      source = cudf::io::source_info(multi_buffer_source.get());
    } else {
      source = cudf::io::source_info(filename);
    }

    auto opts_builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      opts_builder = opts_builder.column_names(n_filter_col_names.as_cpp_vector());
    }
    auto const read_opts = opts_builder.convert_strings_to_categories(false)
                             .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
                             // Ignore any missing projected column(s) by default
                             .ignore_missing_columns(true)
                             .build();
    n_addrs_sizes.cancel();
    n_col_binary_read.cancel();
    auto reader_handle = reinterpret_cast<jlong>(
      new cudf::io::chunked_parquet_reader(static_cast<std::size_t>(chunk_read_limit),
                                           static_cast<std::size_t>(pass_read_limit),
                                           read_opts));
    cudf::jni::native_jlongArray result(env, 2);
    result[0] = reader_handle;
    result[1] = cudf::jni::release_as_jlong(multi_buffer_source);
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_ParquetChunkedReader_createWithDataSource(JNIEnv* env,
                                                              jclass,
                                                              jlong chunk_read_limit,
                                                              jobjectArray filter_col_names,
                                                              jbooleanArray j_col_binary_read,
                                                              jint unit,
                                                              jlong ds_handle)
{
  JNI_NULL_CHECK(env, j_col_binary_read, "Null col_binary_read", 0);
  JNI_NULL_CHECK(env, ds_handle, "Null DataSouurce", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    // TODO: This variable is unused now, but we still don't know what to do with it yet.
    // As such, it needs to stay here for a little more time before we decide to use it again,
    // or remove it completely.
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);
    (void)n_col_binary_read;

    auto ds = reinterpret_cast<cudf::io::datasource*>(ds_handle);
    cudf::io::source_info source{ds};

    auto opts_builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      opts_builder = opts_builder.column_names(n_filter_col_names.as_cpp_vector());
    }
    auto const read_opts = opts_builder.convert_strings_to_categories(false)
                             .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
                             // Ignore any missing projected column(s) by default
                             .ignore_missing_columns(true)
                             .build();

    return reinterpret_cast<jlong>(
      new cudf::io::chunked_parquet_reader(static_cast<std::size_t>(chunk_read_limit), read_opts));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_hasNext(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", false);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_parquet_reader* const>(handle);
    return reader_ptr->has_next();
  }
  JNI_CATCH(env, false);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_readChunk(JNIEnv* env,
                                                                                jclass,
                                                                                jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_parquet_reader* const>(handle);
    auto chunk            = reader_ptr->read_chunk();
    return chunk.tbl ? cudf::jni::convert_table_for_return(env, chunk.tbl) : nullptr;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_close(JNIEnv* env,
                                                                      jclass,
                                                                      jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::io::chunked_parquet_reader*>(handle);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ParquetChunkedReader_destroyMultiHostBufferSource(
  JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );

  JNI_TRY { delete reinterpret_cast<cudf::jni::multi_host_buffer_source*>(handle); }
  JNI_CATCH(env, );
}

//
// Chunked ORC reader JNI
//

namespace {
jlong create_chunked_orc_reader(JNIEnv* env,
                                jlong chunk_read_limit,
                                jlong pass_read_limit,
                                std::optional<jlong> output_granularity,
                                jobjectArray filter_col_names,
                                jlong buffer,
                                jlong buffer_length,
                                jboolean using_numpy_Types,
                                jint unit,
                                jobjectArray dec128_col_names,
                                jboolean ignoreTimezoneInStripeFooter)
{
  JNI_NULL_CHECK(env, buffer, "buffer is null", 0);
  if (buffer_length <= 0) {
    JNI_THROW_NEW(
      env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "An empty buffer is not supported", 0);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
    cudf::jni::native_jstringArray n_dec128_col_names(env, dec128_col_names);
    auto const source = cudf::io::source_info(cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buffer), static_cast<std::size_t>(buffer_length)));
    auto opts_builder = cudf::io::orc_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      opts_builder = opts_builder.columns(n_filter_col_names.as_cpp_vector());
    }
    auto const read_opts = opts_builder.use_index(false)
                             .use_np_dtypes(static_cast<bool>(using_numpy_Types))
                             .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
                             .decimal128_columns(n_dec128_col_names.as_cpp_vector())
                             .ignore_timezone_in_stripe_footer(ignoreTimezoneInStripeFooter)
                             .build();

    if (output_granularity) {
      return reinterpret_cast<jlong>(
        new cudf::io::chunked_orc_reader(static_cast<std::size_t>(chunk_read_limit),
                                         static_cast<std::size_t>(pass_read_limit),
                                         static_cast<std::size_t>(output_granularity.value()),
                                         read_opts));
    }
    return reinterpret_cast<jlong>(
      new cudf::io::chunked_orc_reader(static_cast<std::size_t>(chunk_read_limit),
                                       static_cast<std::size_t>(pass_read_limit),
                                       read_opts));
  }
  JNI_CATCH(env, 0);
}
}  // namespace

// This function should take all the parameters that `Table.readORC` takes,
// plus two more parameters: `chunk_read_limit` and `pass_read_limit`.
JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_ORCChunkedReader_createReader(JNIEnv* env,
                                                  jclass,
                                                  jlong chunk_read_limit,
                                                  jlong pass_read_limit,
                                                  jobjectArray filter_col_names,
                                                  jlong buffer,
                                                  jlong buffer_length,
                                                  jboolean using_numpy_Types,
                                                  jint unit,
                                                  jobjectArray dec128_col_names,
                                                  jboolean ignoreTimezoneInStripeFooter)
{
  return create_chunked_orc_reader(env,
                                   chunk_read_limit,
                                   pass_read_limit,
                                   std::nullopt,
                                   filter_col_names,
                                   buffer,
                                   buffer_length,
                                   using_numpy_Types,
                                   unit,
                                   dec128_col_names,
                                   ignoreTimezoneInStripeFooter);
}

// This function should take all the parameters that `Table.readORC` takes,
// plus three more parameters: `chunk_read_limit`, `pass_read_limit`, `output_granularity`.
JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ORCChunkedReader_createReaderWithOutputGranularity(
  JNIEnv* env,
  jclass,
  jlong chunk_read_limit,
  jlong pass_read_limit,
  jlong output_granularity,
  jobjectArray filter_col_names,
  jlong buffer,
  jlong buffer_length,
  jboolean using_numpy_Types,
  jint unit,
  jobjectArray dec128_col_names,
  jboolean ignoreTimezoneInStripeFooter)
{
  return create_chunked_orc_reader(env,
                                   chunk_read_limit,
                                   pass_read_limit,
                                   output_granularity,
                                   filter_col_names,
                                   buffer,
                                   buffer_length,
                                   using_numpy_Types,
                                   unit,
                                   dec128_col_names,
                                   ignoreTimezoneInStripeFooter);
}

JNIEXPORT jboolean JNICALL Java_ai_rapids_cudf_ORCChunkedReader_hasNext(JNIEnv* env,
                                                                        jclass,
                                                                        jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", false);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_orc_reader* const>(handle);
    return reader_ptr->has_next();
  }
  JNI_CATCH(env, false);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ORCChunkedReader_readChunk(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader_ptr = reinterpret_cast<cudf::io::chunked_orc_reader* const>(handle);
    auto chunk            = reader_ptr->read_chunk();
    return chunk.tbl ? cudf::jni::convert_table_for_return(env, chunk.tbl) : nullptr;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ORCChunkedReader_close(JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::io::chunked_orc_reader*>(handle);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
