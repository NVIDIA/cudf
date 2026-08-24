/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "csv_chunked_writer.hpp"
#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_cccl_any_resource.hpp"
#include "jni_compiled_expr.hpp"
#include "jni_utils.hpp"
#include "jni_writer_data_sink.hpp"
#include "multi_host_buffer_source.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/interop.hpp>
#include <cudf/io/avro.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/conditional_join.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/mixed_join.hpp>
#include <cudf/lists/explode.hpp>
#include <cudf/merge.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/replace.hpp>
#include <cudf/reshape.hpp>
#include <cudf/rolling.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/resource_ref.hpp>

#include <arrow/api.h>
#include <arrow/c/bridge.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include <algorithm>

namespace cudf {
namespace jni {

/**
 * @brief The base class for table writer.
 *
 * By storing a pointer to this base class instead of pointer to specific writer class, we can
 * retrieve common data like `sink` and `stats` for any derived writer class without the need of
 * casting or knowing its type.
 */
struct jni_table_writer_handle_base {
  explicit jni_table_writer_handle_base(
    std::unique_ptr<jni_writer_data_sink>&& sink_,
    std::shared_ptr<cudf::io::writer_compression_statistics>&& stats_)
    : sink{std::move(sink_)}, stats{std::move(stats_)}
  {
  }

  std::unique_ptr<jni_writer_data_sink> sink;
  std::shared_ptr<cudf::io::writer_compression_statistics> stats;
};

template <typename Writer>
struct jni_table_writer_handle final : public jni_table_writer_handle_base {
  explicit jni_table_writer_handle(std::unique_ptr<Writer>&& writer_)
    : jni_table_writer_handle_base(nullptr, nullptr), writer{std::move(writer_)}
  {
  }
  explicit jni_table_writer_handle(
    std::unique_ptr<Writer>&& writer_,
    std::unique_ptr<jni_writer_data_sink>&& sink_,
    std::shared_ptr<cudf::io::writer_compression_statistics>&& stats_)
    : jni_table_writer_handle_base(std::move(sink_), std::move(stats_)), writer{std::move(writer_)}
  {
  }

  std::unique_ptr<Writer> writer;
};

typedef jni_table_writer_handle<cudf::io::chunked_parquet_writer> native_parquet_writer_handle;
typedef jni_table_writer_handle<cudf::io::orc_chunked_writer> native_orc_writer_handle;

class native_arrow_ipc_writer_handle final {
 public:
  explicit native_arrow_ipc_writer_handle(std::vector<std::string> const& col_names,
                                          std::string const& file_name)
    : initialized(false), column_names(col_names), file_name(file_name)
  {
  }

  explicit native_arrow_ipc_writer_handle(std::vector<std::string> const& col_names,
                                          std::shared_ptr<arrow::io::OutputStream> const& sink)
    : initialized(false), column_names(col_names), file_name(""), sink(sink)
  {
  }

 private:
  bool initialized;
  std::vector<std::string> column_names;
  std::vector<cudf::column_metadata> columns_meta;
  std::string file_name;
  std::shared_ptr<arrow::io::OutputStream> sink;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;

 public:
  void write(std::shared_ptr<arrow::Table>& arrow_tab, int64_t max_chunk)
  {
    if (!initialized) {
      if (!sink) {
        auto tmp_sink = arrow::io::FileOutputStream::Open(file_name);
        if (!tmp_sink.ok()) { throw std::runtime_error(tmp_sink.status().message()); }
        sink = *tmp_sink;
      }

      // There is an option to have a file writer too, with metadata
      auto tmp_writer = arrow::ipc::MakeStreamWriter(sink, arrow_tab->schema());
      if (!tmp_writer.ok()) { throw std::runtime_error(tmp_writer.status().message()); }
      writer      = *tmp_writer;
      initialized = true;
    }
    if (arrow_tab->num_rows() == 0) {
      // Arrow C++ IPC writer will not write an empty batch in the case of an
      // empty table, so need to write an empty batch explicitly.
      // For more please see https://issues.apache.org/jira/browse/ARROW-17912.
      auto empty_batch = arrow::RecordBatch::MakeEmpty(arrow_tab->schema());
      auto status      = writer->WriteRecordBatch(*(*empty_batch));
      if (!status.ok()) {
        throw std::runtime_error("writer failed to write batch with the following error: " +
                                 status.ToString());
      }
    } else {
      auto status = writer->WriteTable(*arrow_tab, max_chunk);
      if (!status.ok()) {
        throw std::runtime_error("writer failed to write table with the following error: " +
                                 status.ToString());
      };
    }
  }

  void close()
  {
    if (initialized) {
      {
        auto status = writer->Close();
        if (!status.ok()) {
          throw std::runtime_error("Closing writer failed with the following error: " +
                                   status.ToString());
        }
      }
      {
        auto status = sink->Close();
        if (!status.ok()) {
          throw std::runtime_error("Closing sink failed with the following error: " +
                                   status.ToString());
        }
      }
    }
    initialized = false;
  }

  std::vector<cudf::column_metadata> get_column_metadata(cudf::table_view const& tview)
  {
    if (!column_names.empty() && columns_meta.empty()) {
      // Rebuild the structure of column meta according to table schema.
      // All the tables written by this writer should share the same schema,
      // so build column metadata only once.
      columns_meta.reserve(tview.num_columns());
      size_t idx = 0;
      for (auto itr = tview.begin(); itr < tview.end(); ++itr) {
        // It should consume the column names only when a column is
        //   - type of struct, or
        //   - not a child.
        columns_meta.push_back(build_one_column_meta(*itr, idx));
      }
      if (idx < column_names.size()) {
        throw cudf::jni::jni_exception("Too many column names are provided.");
      }
    }
    return columns_meta;
  }

 private:
  cudf::column_metadata build_one_column_meta(cudf::column_view const& cview,
                                              size_t& idx,
                                              bool const consume_name = true)
  {
    auto col_meta = cudf::column_metadata{};
    if (consume_name) { col_meta.name = get_column_name(idx++); }
    // Process children
    if (cview.type().id() == cudf::type_id::LIST) {
      // list type:
      //   - requires a stub metadata for offset column(index: 0).
      //   - does not require a name for the child column(index 1).
      col_meta.children_meta = {{}, build_one_column_meta(cview.child(1), idx, false)};
    } else if (cview.type().id() == cudf::type_id::STRUCT) {
      // struct type always consumes the column names.
      col_meta.children_meta.reserve(cview.num_children());
      for (auto itr = cview.child_begin(); itr < cview.child_end(); ++itr) {
        col_meta.children_meta.push_back(build_one_column_meta(*itr, idx));
      }
    } else if (cview.type().id() == cudf::type_id::DICTIONARY32) {
      // not supported yet in JNI, nested type?
      throw cudf::jni::jni_exception("Unsupported type 'DICTIONARY32'");
    }
    return col_meta;
  }

  std::string& get_column_name(size_t const idx)
  {
    if (idx < 0 || idx >= column_names.size()) {
      throw cudf::jni::jni_exception("Missing names for columns or nested struct columns");
    }
    return column_names[idx];
  }
};

class jni_arrow_output_stream final : public arrow::io::OutputStream {
 public:
  explicit jni_arrow_output_stream(JNIEnv* env, jobject callback, jobject host_memory_allocator)
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }

    jclass cls = env->GetObjectClass(callback);
    if (cls == nullptr) { throw cudf::jni::jni_exception("class not found"); }

    handle_buffer_method =
      env->GetMethodID(cls, "handleBuffer", "(Lai/rapids/cudf/HostMemoryBuffer;J)V");
    if (handle_buffer_method == nullptr) { throw cudf::jni::jni_exception("handleBuffer method"); }
    this->callback              = add_global_ref(env, callback);
    this->host_memory_allocator = add_global_ref(env, host_memory_allocator);
  }

  virtual ~jni_arrow_output_stream()
  {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      callback              = del_global_ref(env, callback);
      current_buffer        = del_global_ref(env, current_buffer);
      host_memory_allocator = del_global_ref(env, host_memory_allocator);
    }
    callback              = nullptr;
    current_buffer        = nullptr;
    host_memory_allocator = nullptr;
  }

  arrow::Status Write(std::shared_ptr<arrow::Buffer> const& data) override
  {
    return Write(data->data(), data->size());
  }

  arrow::Status Write(void const* data, int64_t nbytes) override
  {
    JNIEnv* env           = cudf::jni::get_jni_env(jvm);
    int64_t left_to_copy  = nbytes;
    char const* copy_from = static_cast<char const*>(data);
    while (left_to_copy > 0) {
      long buffer_amount_available = current_buffer_len - current_buffer_written;
      if (buffer_amount_available <= 0) {
        // should never be < 0, but just to be safe
        rotate_buffer(env);
        buffer_amount_available = current_buffer_len - current_buffer_written;
      }
      long amount_to_copy =
        left_to_copy < buffer_amount_available ? left_to_copy : buffer_amount_available;
      char* copy_to = current_buffer_data + current_buffer_written;

      std::memcpy(copy_to, copy_from, amount_to_copy);
      copy_from = copy_from + amount_to_copy;
      current_buffer_written += amount_to_copy;
      total_written += amount_to_copy;
      left_to_copy -= amount_to_copy;
    }
    return arrow::Status::OK();
  }

  arrow::Status Flush() override
  {
    if (current_buffer_written > 0) {
      JNIEnv* env = cudf::jni::get_jni_env(jvm);
      handle_buffer(env, current_buffer, current_buffer_written);
      current_buffer         = del_global_ref(env, current_buffer);
      current_buffer_len     = 0;
      current_buffer_data    = nullptr;
      current_buffer_written = 0;
    }
    return arrow::Status::OK();
  }

  arrow::Status Close() override
  {
    auto ret  = Flush();
    is_closed = true;
    return ret;
  }

  arrow::Status Abort() override
  {
    is_closed = true;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override { return total_written; }

  bool closed() const override { return is_closed; }

 private:
  void rotate_buffer(JNIEnv* env)
  {
    if (current_buffer != nullptr) { handle_buffer(env, current_buffer, current_buffer_written); }
    current_buffer         = del_global_ref(env, current_buffer);
    jobject tmp_buffer     = allocate_host_buffer(env, alloc_size, true, host_memory_allocator);
    current_buffer         = add_global_ref(env, tmp_buffer);
    current_buffer_len     = get_host_buffer_length(env, current_buffer);
    current_buffer_data    = reinterpret_cast<char*>(get_host_buffer_address(env, current_buffer));
    current_buffer_written = 0;
  }

  void handle_buffer(JNIEnv* env, jobject buffer, jlong len)
  {
    env->CallVoidMethod(callback, handle_buffer_method, buffer, len);
    if (env->ExceptionCheck()) { throw std::runtime_error("handleBuffer threw an exception"); }
  }

  JavaVM* jvm;
  jobject callback;
  jmethodID handle_buffer_method;
  jobject current_buffer      = nullptr;
  char* current_buffer_data   = nullptr;
  long current_buffer_len     = 0;
  long current_buffer_written = 0;
  int64_t total_written       = 0;
  long alloc_size             = MINIMUM_WRITE_BUFFER_SIZE;
  bool is_closed              = false;
  jobject host_memory_allocator;
};

class jni_arrow_input_stream final : public arrow::io::InputStream {
 public:
  explicit jni_arrow_input_stream(JNIEnv* env, jobject callback)
    : mm(arrow::default_cpu_memory_manager())
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }

    jclass cls = env->GetObjectClass(callback);
    if (cls == nullptr) { throw cudf::jni::jni_exception("class not found"); }

    read_into_method = env->GetMethodID(cls, "readInto", "(JJ)J");
    if (read_into_method == nullptr) { throw cudf::jni::jni_exception("readInto method"); }

    this->callback = add_global_ref(env, callback);
  }

  virtual ~jni_arrow_input_stream()
  {
    // This should normally be called by a JVM thread. If the JVM environment is missing then this
    // is likely being triggered by the C++ runtime during shutdown. In that case the JVM may
    // already be destroyed and this thread should not try to attach to get an environment.
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
      callback = del_global_ref(env, callback);
    }
    callback = nullptr;
  }

  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override
  {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    jlong ret   = read_into(env, ptr_as_jlong(out), nbytes);
    total_read += ret;
    return ret;
  }

  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override
  {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    arrow::Result<std::shared_ptr<arrow::ResizableBuffer>> tmp_buffer =
      arrow::AllocateResizableBuffer(nbytes);
    if (!tmp_buffer.ok()) { return tmp_buffer; }
    jlong amount_read  = read_into(env, ptr_as_jlong((*tmp_buffer)->data()), nbytes);
    arrow::Status stat = (*tmp_buffer)->Resize(amount_read);
    if (!stat.ok()) { return stat; }
    return tmp_buffer;
  }

  arrow::Status Close() override
  {
    is_closed = true;
    return arrow::Status::OK();
  }

  arrow::Status Abort() override
  {
    is_closed = true;
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override { return total_read; }

  bool closed() const override { return is_closed; }

 private:
  jlong read_into(JNIEnv* env, jlong addr, jlong len)
  {
    jlong ret = env->CallLongMethod(callback, read_into_method, addr, len);
    if (env->ExceptionCheck()) { throw std::runtime_error("readInto threw an exception"); }
    return ret;
  }

  JavaVM* jvm;
  jobject callback;
  jmethodID read_into_method;
  int64_t total_read = 0;
  bool is_closed     = false;
  std::vector<uint8_t> tmp_buffer;
  std::shared_ptr<arrow::MemoryManager> mm;
};

class native_arrow_ipc_reader_handle final {
 public:
  explicit native_arrow_ipc_reader_handle(std::string const& file_name)
  {
    auto tmp_source = arrow::io::ReadableFile::Open(file_name);
    if (!tmp_source.ok()) { throw std::runtime_error(tmp_source.status().message()); }
    source          = *tmp_source;
    auto tmp_reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!tmp_reader.ok()) { throw std::runtime_error(tmp_reader.status().message()); }
    reader = *tmp_reader;
  }

  explicit native_arrow_ipc_reader_handle(std::shared_ptr<arrow::io::InputStream> source)
    : source(source)
  {
    auto tmp_reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!tmp_reader.ok()) { throw std::runtime_error(tmp_reader.status().message()); }
    reader = *tmp_reader;
  }

  std::shared_ptr<arrow::Table> next(int32_t row_target)
  {
    int64_t total_rows = 0;
    bool done          = false;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    while (!done) {
      arrow::Result<std::shared_ptr<arrow::RecordBatch>> batch = reader->Next();
      if (!batch.ok()) { throw std::runtime_error(batch.status().message()); }
      if (!*batch) {
        done = true;
      } else {
        batches.push_back(*batch);
        total_rows += (*batch)->num_rows();
        done = (total_rows >= row_target);
      }
    }
    if (batches.empty()) {
      // EOF
      return std::unique_ptr<arrow::Table>();
    }
    arrow::Result<std::shared_ptr<arrow::Table>> tmp =
      arrow::Table::FromRecordBatches(reader->schema(), batches);
    if (!tmp.ok()) { throw std::runtime_error(tmp.status().message()); }
    return *tmp;
  }

  std::shared_ptr<arrow::io::InputStream> source;
  std::shared_ptr<arrow::ipc::RecordBatchReader> reader;

  void close()
  {
    auto status = source->Close();
    if (!status.ok()) {
      throw std::runtime_error("Closing source failed with the following error: " +
                               status.ToString());
    }
  }
};

jlongArray convert_table_for_return(JNIEnv* env,
                                    std::unique_ptr<cudf::table>&& table_result,
                                    std::vector<std::unique_ptr<cudf::column>>&& extra_columns)
{
  std::vector<std::unique_ptr<cudf::column>> ret = table_result->release();
  int table_cols                                 = ret.size();
  int num_columns                                = table_cols + extra_columns.size();
  cudf::jni::native_jlongArray outcol_handles(env, num_columns);
  std::transform(ret.begin(), ret.end(), outcol_handles.begin(), [](auto& col) {
    return release_as_jlong(col);
  });
  std::transform(
    extra_columns.begin(), extra_columns.end(), outcol_handles.begin() + table_cols, [](auto& col) {
      return release_as_jlong(col);
    });
  return outcol_handles.get_jArray();
}

jlongArray convert_table_for_return(JNIEnv* env,
                                    std::unique_ptr<cudf::table>& table_result,
                                    std::vector<std::unique_ptr<cudf::column>>&& extra_columns)
{
  return convert_table_for_return(env, std::move(table_result), std::move(extra_columns));
}

jlongArray convert_table_for_return(JNIEnv* env,
                                    std::unique_ptr<cudf::table>& first_table,
                                    std::unique_ptr<cudf::table>& second_table)
{
  return convert_table_for_return(env, first_table, second_table->release());
}

// Convert the JNI boolean array of key column sort order to a vector of cudf::order
// for groupby.
std::vector<cudf::order> resolve_column_order(JNIEnv* env,
                                              jbooleanArray jkeys_sort_desc,
                                              int key_size)
{
  cudf::jni::native_jbooleanArray keys_sort_desc(env, jkeys_sort_desc);
  auto keys_sort_num = keys_sort_desc.size();
  // The number of column order should be 0 or equal to the number of key.
  if (keys_sort_num != 0 && keys_sort_num != key_size) {
    throw cudf::jni::jni_exception("key-column and key-sort-order size mismatch.");
  }

  std::vector<cudf::order> column_order(keys_sort_num);
  if (keys_sort_num > 0) {
    std::transform(
      keys_sort_desc.data(),
      keys_sort_desc.data() + keys_sort_num,
      column_order.begin(),
      [](jboolean is_desc) { return is_desc ? cudf::order::DESCENDING : cudf::order::ASCENDING; });
  }
  return column_order;
}

// Convert the JNI boolean array of key column null order to a vector of cudf::null_order
// for groupby.
std::vector<cudf::null_order> resolve_null_precedence(JNIEnv* env,
                                                      jbooleanArray jkeys_null_first,
                                                      int key_size)
{
  cudf::jni::native_jbooleanArray keys_null_first(env, jkeys_null_first);
  auto null_order_num = keys_null_first.size();
  // The number of null order should be 0 or equal to the number of key.
  if (null_order_num != 0 && null_order_num != key_size) {
    throw cudf::jni::jni_exception("key-column and key-null-order size mismatch.");
  }

  std::vector<cudf::null_order> null_precedence(null_order_num);
  if (null_order_num > 0) {
    std::transform(keys_null_first.data(),
                   keys_null_first.data() + null_order_num,
                   null_precedence.begin(),
                   [](jboolean null_before) {
                     return null_before ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
                   });
  }
  return null_precedence;
}

namespace {

int set_column_metadata(cudf::io::column_in_metadata& column_metadata,
                        std::vector<std::string>& col_names,
                        cudf::jni::native_jbooleanArray& nullability,
                        cudf::jni::native_jbooleanArray& is_int96,
                        cudf::jni::native_jintArray& precisions,
                        cudf::jni::native_jbooleanArray& is_map,
                        cudf::jni::native_jbooleanArray& hasParquetFieldIds,
                        cudf::jni::native_jintArray& parquetFieldIds,
                        cudf::jni::native_jintArray& children,
                        int num_children,
                        int read_index,
                        cudf::jni::native_jbooleanArray& is_binary)
{
  int write_index = 0;
  for (int i = 0; i < num_children; i++, write_index++) {
    cudf::io::column_in_metadata child;
    child.set_name(col_names[read_index]).set_nullability(nullability[read_index]);
    if (precisions[read_index] > -1) { child.set_decimal_precision(precisions[read_index]); }
    if (!is_int96.is_null()) { child.set_int96_timestamps(is_int96[read_index]); }
    if (!is_binary.is_null()) { child.set_output_as_binary(is_binary[read_index]); }
    if (is_map[read_index]) { child.set_list_column_as_map(); }
    if (!parquetFieldIds.is_null() && hasParquetFieldIds[read_index]) {
      child.set_parquet_field_id(parquetFieldIds[read_index]);
    }
    column_metadata.add_child(child);
    int childs_children = children[read_index++];
    if (childs_children > 0) {
      read_index = set_column_metadata(column_metadata.child(write_index),
                                       col_names,
                                       nullability,
                                       is_int96,
                                       precisions,
                                       is_map,
                                       hasParquetFieldIds,
                                       parquetFieldIds,
                                       children,
                                       childs_children,
                                       read_index,
                                       is_binary);
    }
  }
  return read_index;
}

void createTableMetaData(JNIEnv* env,
                         jint num_children,
                         jobjectArray& j_col_names,
                         jintArray& j_children,
                         jbooleanArray& j_col_nullability,
                         jbooleanArray& j_is_int96,
                         jintArray& j_precisions,
                         jbooleanArray& j_is_map,
                         cudf::io::table_input_metadata& metadata,
                         jbooleanArray& j_hasParquetFieldIds,
                         jintArray& j_parquetFieldIds,
                         jbooleanArray& j_is_binary)
{
  cudf::jni::auto_set_device(env);
  cudf::jni::native_jstringArray col_names(env, j_col_names);
  cudf::jni::native_jbooleanArray col_nullability(env, j_col_nullability);
  cudf::jni::native_jbooleanArray is_int96(env, j_is_int96);
  cudf::jni::native_jintArray precisions(env, j_precisions);
  cudf::jni::native_jbooleanArray hasParquetFieldIds(env, j_hasParquetFieldIds);
  cudf::jni::native_jintArray parquetFieldIds(env, j_parquetFieldIds);
  cudf::jni::native_jintArray children(env, j_children);
  cudf::jni::native_jbooleanArray is_map(env, j_is_map);
  cudf::jni::native_jbooleanArray is_binary(env, j_is_binary);

  auto cpp_names = col_names.as_cpp_vector();

  int top_level_children = num_children;

  metadata.column_metadata.resize(top_level_children);
  int read_index = 0;  // the read_index, which will be used to read the arrays
  for (int i = read_index, write_index = 0; i < top_level_children; i++, write_index++) {
    metadata.column_metadata[write_index]
      .set_name(cpp_names[read_index])
      .set_nullability(col_nullability[read_index]);
    if (precisions[read_index] > -1) {
      metadata.column_metadata[write_index].set_decimal_precision(precisions[read_index]);
    }
    if (!is_int96.is_null()) {
      metadata.column_metadata[write_index].set_int96_timestamps(is_int96[read_index]);
    }
    if (!is_binary.is_null()) {
      metadata.column_metadata[write_index].set_output_as_binary(is_binary[read_index]);
    }
    if (is_map[read_index]) { metadata.column_metadata[write_index].set_list_column_as_map(); }
    if (!parquetFieldIds.is_null() && hasParquetFieldIds[read_index]) {
      metadata.column_metadata[write_index].set_parquet_field_id(parquetFieldIds[read_index]);
    }
    int childs_children = children[read_index++];
    if (childs_children > 0) {
      read_index = set_column_metadata(metadata.column_metadata[write_index],
                                       cpp_names,
                                       col_nullability,
                                       is_int96,
                                       precisions,
                                       is_map,
                                       hasParquetFieldIds,
                                       parquetFieldIds,
                                       children,
                                       childs_children,
                                       read_index,
                                       is_binary);
    }
  }
}

// Check that window parameters are valid.
bool valid_window_parameters(native_jintArray const& values,
                             native_jpointerArray<cudf::aggregation> const& ops,
                             native_jintArray const& min_periods,
                             native_jintArray const& preceding,
                             native_jintArray const& following)
{
  return values.size() == ops.size() && values.size() == min_periods.size() &&
         values.size() == preceding.size() && values.size() == following.size();
}

// Check that window parameters are valid.
bool valid_window_parameters(native_jintArray const& values,
                             native_jpointerArray<cudf::aggregation> const& ops,
                             native_jintArray const& min_periods,
                             native_jpointerArray<cudf::scalar> const& preceding,
                             native_jpointerArray<cudf::scalar> const& following)
{
  return values.size() == ops.size() && values.size() == min_periods.size() &&
         values.size() == preceding.size() && values.size() == following.size();
}

// Convert a cudf gather map pair into the form that Java expects
// The resulting Java long array contains the following at each index:
//   0: Size of each gather map in bytes
//   1: Device address of the gather map for the left table
//   2: Host address of the rmm::device_buffer instance that owns the left gather map data
//   3: Device address of the gather map for the right table
//   4: Host address of the rmm::device_buffer instance that owns the right gather map data
jlongArray gather_maps_to_java(
  JNIEnv* env,
  std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
            std::unique_ptr<rmm::device_uvector<cudf::size_type>>> maps)
{
  // release the underlying device buffer to Java
  auto left_map_buffer  = std::make_unique<rmm::device_buffer>(maps.first->release());
  auto right_map_buffer = std::make_unique<rmm::device_buffer>(maps.second->release());
  cudf::jni::native_jlongArray result(env, 5);
  result[0] = static_cast<jlong>(left_map_buffer->size());
  result[1] = ptr_as_jlong(left_map_buffer->data());
  result[2] = release_as_jlong(left_map_buffer);
  result[3] = ptr_as_jlong(right_map_buffer->data());
  result[4] = release_as_jlong(right_map_buffer);
  return result.get_jArray();
}

// Convert a cudf gather map into the form that Java expects
// The resulting Java long array contains the following at each index:
//   0: Size of the gather map in bytes
//   1: Device address of the gather map
//   2: Host address of the rmm::device_buffer instance that owns the gather map data
jlongArray gather_map_to_java(JNIEnv* env,
                              std::unique_ptr<rmm::device_uvector<cudf::size_type>> map)
{
  // release the underlying device buffer to Java
  cudf::jni::native_jlongArray result(env, 3);
  result[0]              = static_cast<jlong>(map->size() * sizeof(cudf::size_type));
  auto gather_map_buffer = std::make_unique<rmm::device_buffer>(map->release());
  result[1]              = ptr_as_jlong(gather_map_buffer->data());
  result[2]              = release_as_jlong(gather_map_buffer);
  return result.get_jArray();
}

// Generate gather maps needed to manifest the result of an equi-join between two tables.
template <typename T>
jlongArray join_gather_maps(
  JNIEnv* env, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal, T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_keys, "right_table is null", NULL);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto nulleq = compare_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_maps_to_java(env, join_func(*left_keys, *right_keys, nulleq));
  }
  JNI_CATCH(env, NULL);
}

// Generate gather maps needed to manifest the result of an equi-join between a left table and
// a hash table built from the join's right table.
template <typename T>
jlongArray hash_join_gather_maps(JNIEnv* env,
                                 jlong j_left_keys,
                                 jlong j_right_hash_join,
                                 T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left table is null", NULL);
  JNI_NULL_CHECK(env, j_right_hash_join, "hash join is null", NULL);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_keys = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto hash_join = reinterpret_cast<cudf::hash_join const*>(j_right_hash_join);
    return gather_maps_to_java(env, join_func(*left_keys, *hash_join));
  }
  JNI_CATCH(env, NULL);
}

// Generate gather maps needed to manifest the result of a conditional join between two tables.
template <typename T>
jlongArray cond_join_gather_maps(
  JNIEnv* env, jlong j_left_table, jlong j_right_table, jlong j_condition, T join_func)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, j_condition, "condition is null", NULL);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    return gather_maps_to_java(
      env, join_func(*left_table, *right_table, condition->get_top_expression()));
  }
  JNI_CATCH(env, NULL);
}

// Generate a gather map needed to manifest the result of a semi/anti join between two tables.
template <typename T>
jlongArray join_gather_single_map(
  JNIEnv* env, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal, T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_keys, "right_table is null", NULL);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto nulleq = compare_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_map_to_java(env, join_func(*left_keys, *right_keys, nulleq));
  }
  JNI_CATCH(env, NULL);
}

// Generate a gather map needed to manifest the result of a conditional semi/anti join
// between two tables.
template <typename T>
jlongArray cond_join_gather_single_map(
  JNIEnv* env, jlong j_left_table, jlong j_right_table, jlong j_condition, T join_func)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", NULL);
  JNI_NULL_CHECK(env, j_condition, "condition is null", NULL);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr*>(j_condition);
    return gather_map_to_java(
      env, join_func(*left_table, *right_table, condition->get_top_expression()));
  }
  JNI_CATCH(env, NULL);
}

template <typename T>
jlongArray mixed_join_size(JNIEnv* env,
                           jlong j_left_keys,
                           jlong j_right_keys,
                           jlong j_left_condition,
                           jlong j_right_condition,
                           jlong j_condition,
                           jboolean j_nulls_equal,
                           T join_size_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", 0);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", 0);
  JNI_NULL_CHECK(env, j_left_condition, "left condition table is null", 0);
  JNI_NULL_CHECK(env, j_right_condition, "right condition table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const left_keys       = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys      = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto const left_condition  = reinterpret_cast<cudf::table_view const*>(j_left_condition);
    auto const right_condition = reinterpret_cast<cudf::table_view const*>(j_right_condition);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto [join_size, matches_per_row] = join_size_func(*left_keys,
                                                       *right_keys,
                                                       *left_condition,
                                                       *right_condition,
                                                       condition->get_top_expression(),
                                                       nulls_equal);
    if (matches_per_row->size() > std::numeric_limits<cudf::size_type>::max()) {
      throw std::runtime_error("Too many values in device buffer to convert into a column");
    }
    auto col_size = static_cast<size_type>(matches_per_row->size());
    auto col_data = matches_per_row->release();
    cudf::jni::native_jlongArray result(env, 2);
    result[0] = static_cast<jlong>(join_size);
    result[1] = ptr_as_jlong(new cudf::column{cudf::data_type{cudf::type_id::INT32},
                                              col_size,
                                              std::move(col_data),
                                              rmm::device_buffer{},
                                              0});
    return result.get_jArray();
  }
  JNI_CATCH(env, NULL);
}

std::pair<std::size_t, cudf::device_span<cudf::size_type const>> get_mixed_size_info(
  JNIEnv* env, jlong j_output_row_count, jlong j_matches_view)
{
  auto const row_count = static_cast<std::size_t>(j_output_row_count);
  auto const matches   = reinterpret_cast<cudf::column_view const*>(j_matches_view);
  return std::make_pair(row_count,
                        cudf::device_span<cudf::size_type const>(
                          matches->template data<cudf::size_type>(), matches->size()));
}

template <typename T>
jlongArray mixed_join_gather_maps(JNIEnv* env,
                                  jlong j_left_keys,
                                  jlong j_right_keys,
                                  jlong j_left_condition,
                                  jlong j_right_condition,
                                  jlong j_condition,
                                  jboolean j_nulls_equal,
                                  T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", 0);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", 0);
  JNI_NULL_CHECK(env, j_left_condition, "left condition table is null", 0);
  JNI_NULL_CHECK(env, j_right_condition, "right condition table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const left_keys       = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys      = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto const left_condition  = reinterpret_cast<cudf::table_view const*>(j_left_condition);
    auto const right_condition = reinterpret_cast<cudf::table_view const*>(j_right_condition);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_maps_to_java(env,
                               join_func(*left_keys,
                                         *right_keys,
                                         *left_condition,
                                         *right_condition,
                                         condition->get_top_expression(),
                                         nulls_equal));
  }
  JNI_CATCH(env, NULL);
}

template <typename T>
jlongArray mixed_join_gather_single_map(JNIEnv* env,
                                        jlong j_left_keys,
                                        jlong j_right_keys,
                                        jlong j_left_condition,
                                        jlong j_right_condition,
                                        jlong j_condition,
                                        jboolean j_nulls_equal,
                                        T join_func)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", 0);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", 0);
  JNI_NULL_CHECK(env, j_left_condition, "left condition table is null", 0);
  JNI_NULL_CHECK(env, j_right_condition, "right condition table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const left_keys       = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys      = reinterpret_cast<cudf::table_view const*>(j_right_keys);
    auto const left_condition  = reinterpret_cast<cudf::table_view const*>(j_left_condition);
    auto const right_condition = reinterpret_cast<cudf::table_view const*>(j_right_condition);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    return gather_map_to_java(env,
                              join_func(*left_keys,
                                        *right_keys,
                                        *left_condition,
                                        *right_condition,
                                        condition->get_top_expression(),
                                        nulls_equal));
  }
  JNI_CATCH(env, NULL);
}

cudf::column_view remove_validity_from_col(cudf::column_view column_view)
{
  if (!cudf::is_compound(column_view.type())) {
    if (column_view.nullable() && column_view.null_count() == 0) {
      // null_mask is allocated but no nulls present therefore we create a new column_view without
      // the null_mask to avoid things blowing up in reading the parquet file
      return cudf::column_view(column_view.type(),
                               column_view.size(),
                               column_view.head(),
                               nullptr,
                               0,
                               column_view.offset());
    } else {
      return cudf::column_view(column_view);
    }
  } else {
    std::vector<cudf::column_view> children;
    children.reserve(column_view.num_children());
    for (auto it = column_view.child_begin(); it != column_view.child_end(); it++) {
      children.push_back(remove_validity_from_col(*it));
    }
    if (!column_view.nullable() || column_view.null_count() != 0) {
      return cudf::column_view(column_view.type(),
                               column_view.size(),
                               column_view.head(),
                               column_view.null_mask(),
                               column_view.null_count(),
                               column_view.offset(),
                               children);
    } else {
      return cudf::column_view(column_view.type(),
                               column_view.size(),
                               column_view.head(),
                               nullptr,
                               0,
                               column_view.offset(),
                               children);
    }
  }
}

cudf::table_view remove_validity_if_needed(cudf::table_view* input_table_view)
{
  std::vector<cudf::column_view> views;
  views.reserve(input_table_view->num_columns());
  for (auto it = input_table_view->begin(); it != input_table_view->end(); it++) {
    views.push_back(remove_validity_from_col(*it));
  }

  return cudf::table_view(views);
}

cudf::io::schema_element read_schema_element(int& index,
                                             cudf::jni::native_jintArray const& children,
                                             cudf::jni::native_jstringArray const& names,
                                             cudf::jni::native_jintArray const& types,
                                             cudf::jni::native_jintArray const& scales)
{
  auto d_type = cudf::data_type{static_cast<cudf::type_id>(types[index]), scales[index]};
  if (d_type.id() == cudf::type_id::STRUCT || d_type.id() == cudf::type_id::LIST) {
    std::map<std::string, cudf::io::schema_element> child_elems;
    int num_children = children[index];
    std::vector<std::string> child_names(num_children);
    // go to the next entry, so recursion can parse it.
    index++;
    for (int i = 0; i < num_children; i++) {
      std::string name = names.get(index);
      child_elems.insert(
        std::pair{name, cudf::jni::read_schema_element(index, children, names, types, scales)});
      child_names[i] = std::move(name);
    }
    return cudf::io::schema_element{d_type, std::move(child_elems), {std::move(child_names)}};
  } else {
    if (children[index] != 0) {
      throw std::invalid_argument("found children for a type that should have none");
    }
    // go to the next entry before returning...
    index++;
    return cudf::io::schema_element{d_type, {}, std::nullopt};
  }
}

void append_flattened_child_counts(cudf::io::column_name_info const& info, std::vector<int>& counts)
{
  counts.push_back(info.children.size());
  for (cudf::io::column_name_info const& child : info.children) {
    append_flattened_child_counts(child, counts);
  }
}

void append_flattened_child_names(cudf::io::column_name_info const& info,
                                  std::vector<std::string>& names)
{
  names.push_back(info.name);
  for (cudf::io::column_name_info const& child : info.children) {
    append_flattened_child_names(child, names);
  }
}

// Recursively make schema and its children nullable
void set_nullable(ArrowSchema* schema)
{
  schema->flags |= ARROW_FLAG_NULLABLE;
  for (int i = 0; i < schema->n_children; ++i) {
    set_nullable(schema->children[i]);
  }
}

}  // namespace

}  // namespace jni
}  // namespace cudf

using cudf::jni::convert_table_for_return;
using cudf::jni::ptr_as_jlong;
using cudf::jni::release_as_jlong;

extern "C" {

// This is a method purely added for testing remove_validity_if_needed method
JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_removeNullMasksIfNeeded(JNIEnv* env,
                                                                               jclass,
                                                                               jlong j_table_view)
{
  JNI_NULL_CHECK(env, j_table_view, "table view handle is null", 0);
  JNI_TRY
  {
    cudf::table_view* tview = reinterpret_cast<cudf::table_view*>(j_table_view);
    cudf::table_view result = cudf::jni::remove_validity_if_needed(tview);
    cudf::table m_tbl(result);
    std::vector<std::unique_ptr<cudf::column>> cols = m_tbl.release();
    auto results = cudf::jni::native_jlongArray(env, cols.size());
    std::transform(
      cols.begin(), cols.end(), results.begin(), [](auto& col) { return release_as_jlong(col); });
    return results.get_jArray();
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_createCudfTableView(JNIEnv* env,
                                                                      jclass,
                                                                      jlongArray j_cudf_columns)
{
  JNI_NULL_CHECK(env, j_cudf_columns, "columns are null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_columns(env, j_cudf_columns);

    std::vector<cudf::column_view> column_views = n_cudf_columns.get_dereferenced();
    return ptr_as_jlong(new cudf::table_view(column_views));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_deleteCudfTable(JNIEnv* env,
                                                                 jclass,
                                                                 jlong j_cudf_table_view)
{
  JNI_NULL_CHECK(env, j_cudf_table_view, "table view handle is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::table_view*>(j_cudf_table_view);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_columnViewsFromPacked(JNIEnv* env,
                                                                             jclass,
                                                                             jobject buffer_obj,
                                                                             jlong j_data_address)
{
  // The GPU data address can be null when the table is empty, so it is not null-checked here.
  JNI_NULL_CHECK(env, buffer_obj, "metadata is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    void const* metadata_address = env->GetDirectBufferAddress(buffer_obj);
    JNI_NULL_CHECK(env, metadata_address, "metadata buffer address is null", nullptr);
    cudf::table_view table = cudf::unpack(static_cast<uint8_t const*>(metadata_address),
                                          reinterpret_cast<uint8_t const*>(j_data_address));
    cudf::jni::native_jlongArray views(env, table.num_columns());
    for (int i = 0; i < table.num_columns(); i++) {
      // TODO Exception handling is not ideal, if no exceptions are thrown ownership of the new cv
      // is passed to Java. If an exception is thrown we need to free it, but this needs to be
      // coordinated with the Java side because one column may have changed ownership while
      // another may not have. We don't want to double free the view so for now we just let it
      // leak because it should be a small amount of host memory.
      //
      // In the ideal case we would keep the view where it is at, and pass in a pointer to it
      // That pointer would then be copied when Java takes ownership of it, but that adds an
      // extra JNI call that I would like to avoid for performance reasons.
      views[i] = ptr_as_jlong(new cudf::column_view(table.column(i)));
    }
    views.commit();

    return views.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_sortOrder(JNIEnv* env,
                                                            jclass,
                                                            jlong j_input_table,
                                                            jlongArray j_sort_keys_columns,
                                                            jbooleanArray j_is_descending,
                                                            jbooleanArray j_are_nulls_smallest)
{
  // input validations & verifications
  JNI_NULL_CHECK(env, j_input_table, "input table is null", 0);
  JNI_NULL_CHECK(env, j_sort_keys_columns, "sort keys columns is null", 0);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", 0);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null order array is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_sort_keys_columns(env,
                                                                           j_sort_keys_columns);
    jsize num_columns = n_sort_keys_columns.size();
    cudf::jni::native_jbooleanArray const n_is_descending(env, j_is_descending);
    jsize num_columns_is_desc = n_is_descending.size();

    JNI_ARG_CHECK(
      env, num_columns_is_desc == num_columns, "columns and is_descending lengths don't match", 0);

    cudf::jni::native_jbooleanArray const n_are_nulls_smallest(env, j_are_nulls_smallest);
    jsize num_columns_null_smallest = n_are_nulls_smallest.size();

    JNI_ARG_CHECK(env,
                  num_columns_null_smallest == num_columns,
                  "columns and is_descending lengths don't match",
                  0);

    std::vector<cudf::order> order =
      n_is_descending.transform_if_else(cudf::order::DESCENDING, cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_order =
      n_are_nulls_smallest.transform_if_else(cudf::null_order::BEFORE, cudf::null_order::AFTER);

    std::vector<cudf::column_view> sort_keys = n_sort_keys_columns.get_dereferenced();
    return release_as_jlong(cudf::sorted_order(cudf::table_view{sort_keys}, order, null_order));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_orderBy(JNIEnv* env,
                                                               jclass,
                                                               jlong j_input_table,
                                                               jlongArray j_sort_keys_columns,
                                                               jbooleanArray j_is_descending,
                                                               jbooleanArray j_are_nulls_smallest)
{
  // input validations & verifications
  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_sort_keys_columns, "sort keys columns is null", NULL);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", NULL);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null order array is null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_sort_keys_columns(env,
                                                                           j_sort_keys_columns);
    jsize num_columns = n_sort_keys_columns.size();
    cudf::jni::native_jbooleanArray const n_is_descending(env, j_is_descending);
    jsize num_columns_is_desc = n_is_descending.size();

    JNI_ARG_CHECK(
      env, num_columns_is_desc == num_columns, "columns and is_descending lengths don't match", 0);

    cudf::jni::native_jbooleanArray const n_are_nulls_smallest(env, j_are_nulls_smallest);
    jsize num_columns_null_smallest = n_are_nulls_smallest.size();

    JNI_ARG_CHECK(env,
                  num_columns_null_smallest == num_columns,
                  "columns and areNullsSmallest lengths don't match",
                  0);

    std::vector<cudf::order> order =
      n_is_descending.transform_if_else(cudf::order::DESCENDING, cudf::order::ASCENDING);

    std::vector<cudf::null_order> null_order =
      n_are_nulls_smallest.transform_if_else(cudf::null_order::BEFORE, cudf::null_order::AFTER);

    std::vector<cudf::column_view> sort_keys = n_sort_keys_columns.get_dereferenced();
    auto sorted_col = cudf::sorted_order(cudf::table_view{sort_keys}, order, null_order);

    auto const input_table = reinterpret_cast<cudf::table_view const*>(j_input_table);
    return convert_table_for_return(env, cudf::gather(*input_table, sorted_col->view()));
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_merge(JNIEnv* env,
                                                             jclass,
                                                             jlongArray j_table_handles,
                                                             jintArray j_sort_key_indexes,
                                                             jbooleanArray j_is_descending,
                                                             jbooleanArray j_are_nulls_smallest)
{
  // input validations & verifications
  JNI_NULL_CHECK(env, j_table_handles, "input tables are null", NULL);
  JNI_NULL_CHECK(env, j_sort_key_indexes, "key indexes is null", NULL);
  JNI_NULL_CHECK(env, j_is_descending, "sort order array is null", NULL);
  JNI_NULL_CHECK(env, j_are_nulls_smallest, "null order array is null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::table_view> n_table_handles(env, j_table_handles);

    cudf::jni::native_jintArray const n_sort_key_indexes(env, j_sort_key_indexes);
    jsize num_columns = n_sort_key_indexes.size();
    cudf::jni::native_jbooleanArray const n_is_descending(env, j_is_descending);
    jsize num_columns_is_desc = n_is_descending.size();

    JNI_ARG_CHECK(env,
                  num_columns_is_desc == num_columns,
                  "columns and is_descending lengths don't match",
                  NULL);

    cudf::jni::native_jbooleanArray const n_are_nulls_smallest(env, j_are_nulls_smallest);
    jsize num_columns_null_smallest = n_are_nulls_smallest.size();

    JNI_ARG_CHECK(env,
                  num_columns_null_smallest == num_columns,
                  "columns and areNullsSmallest lengths don't match",
                  NULL);

    std::vector<int> indexes = n_sort_key_indexes.to_vector<int>();
    std::vector<cudf::order> order =
      n_is_descending.transform_if_else(cudf::order::DESCENDING, cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_order =
      n_are_nulls_smallest.transform_if_else(cudf::null_order::BEFORE, cudf::null_order::AFTER);
    std::vector<cudf::table_view> tables = n_table_handles.get_dereferenced();

    return convert_table_for_return(env, cudf::merge(tables, indexes, order, null_order));
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_readCSVFromDataSource(JNIEnv* env,
                                                jclass,
                                                jobjectArray col_names,
                                                jintArray j_types,
                                                jintArray j_scales,
                                                jobjectArray filter_col_names,
                                                jint header_row,
                                                jbyte delim,
                                                jint j_quote_style,
                                                jbyte quote,
                                                jbyte comment,
                                                jobjectArray null_values,
                                                jobjectArray true_values,
                                                jobjectArray false_values,
                                                jlong ds_handle)
{
  JNI_NULL_CHECK(env, null_values, "null_values must be supplied, even if it is empty", NULL);
  JNI_NULL_CHECK(env, ds_handle, "no data source handle given", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray n_col_names(env, col_names);
    cudf::jni::native_jintArray n_types(env, j_types);
    cudf::jni::native_jintArray n_scales(env, j_scales);
    if (n_types.is_null() != n_scales.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match null", NULL);
    }
    std::vector<cudf::data_type> data_types;
    if (!n_types.is_null()) {
      if (n_types.size() != n_scales.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match size", NULL);
      }
      data_types.reserve(n_types.size());
      std::transform(n_types.begin(),
                     n_types.end(),
                     n_scales.begin(),
                     std::back_inserter(data_types),
                     [](auto type, auto scale) {
                       return cudf::data_type{static_cast<cudf::type_id>(type), scale};
                     });
    }

    cudf::jni::native_jstringArray n_null_values(env, null_values);
    cudf::jni::native_jstringArray n_true_values(env, true_values);
    cudf::jni::native_jstringArray n_false_values(env, false_values);
    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    auto ds = reinterpret_cast<cudf::io::datasource*>(ds_handle);
    cudf::io::source_info source{ds};

    auto const quote_style = static_cast<cudf::io::quote_style>(j_quote_style);

    cudf::io::csv_reader_options opts = cudf::io::csv_reader_options::builder(source)
                                          .delimiter(delim)
                                          .header(header_row)
                                          .names(n_col_names.as_cpp_vector())
                                          .dtypes(data_types)
                                          .use_cols_names(n_filter_col_names.as_cpp_vector())
                                          .true_values(n_true_values.as_cpp_vector())
                                          .false_values(n_false_values.as_cpp_vector())
                                          .na_values(n_null_values.as_cpp_vector())
                                          .keep_default_na(false)
                                          .na_filter(n_null_values.size() > 0)
                                          .quoting(quote_style)
                                          .quotechar(quote)
                                          .comment(comment)
                                          .build();

    return convert_table_for_return(env, cudf::io::read_csv(opts).tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readCSV(JNIEnv* env,
                                                               jclass,
                                                               jobjectArray col_names,
                                                               jintArray j_types,
                                                               jintArray j_scales,
                                                               jobjectArray filter_col_names,
                                                               jstring inputfilepath,
                                                               jlong buffer,
                                                               jlong buffer_length,
                                                               jint header_row,
                                                               jbyte delim,
                                                               jint j_quote_style,
                                                               jbyte quote,
                                                               jbyte comment,
                                                               jobjectArray null_values,
                                                               jobjectArray true_values,
                                                               jobjectArray false_values)
{
  JNI_NULL_CHECK(env, null_values, "null_values must be supplied, even if it is empty", NULL);

  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "cannot pass in both a buffer and an inputfilepath",
                  NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(
      env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "An empty buffer is not supported", NULL);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray n_col_names(env, col_names);
    cudf::jni::native_jintArray n_types(env, j_types);
    cudf::jni::native_jintArray n_scales(env, j_scales);
    if (n_types.is_null() != n_scales.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match null", NULL);
    }
    std::vector<cudf::data_type> data_types;
    if (!n_types.is_null()) {
      if (n_types.size() != n_scales.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match size", NULL);
      }
      data_types.reserve(n_types.size());
      std::transform(n_types.begin(),
                     n_types.end(),
                     n_scales.begin(),
                     std::back_inserter(data_types),
                     [](auto type, auto scale) {
                       return cudf::data_type{static_cast<cudf::type_id>(type), scale};
                     });
    }

    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inputfilepath can't be empty", NULL);
    }

    cudf::jni::native_jstringArray n_null_values(env, null_values);
    cudf::jni::native_jstringArray n_true_values(env, true_values);
    cudf::jni::native_jstringArray n_false_values(env, false_values);
    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    auto source =
      read_buffer
        ? cudf::io::source_info{cudf::host_span<std::byte const>(
            reinterpret_cast<std::byte const*>(buffer), static_cast<std::size_t>(buffer_length))}
        : cudf::io::source_info{filename};
    auto const quote_style = static_cast<cudf::io::quote_style>(j_quote_style);

    cudf::io::csv_reader_options opts = cudf::io::csv_reader_options::builder(source)
                                          .delimiter(delim)
                                          .header(header_row)
                                          .names(n_col_names.as_cpp_vector())
                                          .dtypes(data_types)
                                          .use_cols_names(n_filter_col_names.as_cpp_vector())
                                          .true_values(n_true_values.as_cpp_vector())
                                          .false_values(n_false_values.as_cpp_vector())
                                          .na_values(n_null_values.as_cpp_vector())
                                          .keep_default_na(false)
                                          .na_filter(n_null_values.size() > 0)
                                          .quoting(quote_style)
                                          .quotechar(quote)
                                          .comment(comment)
                                          .build();

    return convert_table_for_return(env, cudf::io::read_csv(opts).tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeCSVToFile(JNIEnv* env,
                                                                jclass,
                                                                jlong j_table_handle,
                                                                jobjectArray j_column_names,
                                                                jboolean include_header,
                                                                jstring j_row_delimiter,
                                                                jbyte j_field_delimiter,
                                                                jstring j_null_value,
                                                                jstring j_true_value,
                                                                jstring j_false_value,
                                                                jint j_quote_style,
                                                                jstring j_output_path)
{
  JNI_NULL_CHECK(env, j_table_handle, "table handle cannot be null.", );
  JNI_NULL_CHECK(env, j_column_names, "column name array cannot be null", );
  JNI_NULL_CHECK(env, j_row_delimiter, "row delimiter cannot be null", );
  JNI_NULL_CHECK(env, j_field_delimiter, "field delimiter cannot be null", );
  JNI_NULL_CHECK(env, j_null_value, "null representation string cannot be itself null", );
  JNI_NULL_CHECK(env, j_true_value, "representation string for `true` cannot be null", );
  JNI_NULL_CHECK(env, j_false_value, "representation string for `false` cannot be null", );
  JNI_NULL_CHECK(env, j_output_path, "output path cannot be null", );

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const output_path = cudf::jni::native_jstring{env, j_output_path};

    auto const table          = reinterpret_cast<cudf::table_view*>(j_table_handle);
    auto const n_column_names = cudf::jni::native_jstringArray{env, j_column_names};
    auto const column_names   = n_column_names.as_cpp_vector();

    auto const line_terminator = cudf::jni::native_jstring{env, j_row_delimiter};
    auto const na_rep          = cudf::jni::native_jstring{env, j_null_value};
    auto const true_value      = cudf::jni::native_jstring{env, j_true_value};
    auto const false_value     = cudf::jni::native_jstring{env, j_false_value};
    auto const quote_style     = static_cast<cudf::io::quote_style>(j_quote_style);

    auto options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{output_path}, *table)
                     .names(column_names)
                     .include_header(static_cast<bool>(include_header))
                     .line_terminator(line_terminator)
                     .inter_column_delimiter(j_field_delimiter)
                     .na_rep(na_rep)
                     .true_value(true_value)
                     .false_value(false_value)
                     .quoting(quote_style);

    cudf::io::write_csv(options.build());
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_Table_startWriteCSVToBuffer(JNIEnv* env,
                                                jclass,
                                                jobjectArray j_column_names,
                                                jboolean include_header,
                                                jstring j_row_delimiter,
                                                jbyte j_field_delimiter,
                                                jstring j_null_value,
                                                jstring j_true_value,
                                                jstring j_false_value,
                                                jint j_quote_style,
                                                jobject j_buffer,
                                                jobject host_memory_allocator)
{
  JNI_NULL_CHECK(env, j_column_names, "column name array cannot be null", 0);
  JNI_NULL_CHECK(env, j_row_delimiter, "row delimiter cannot be null", 0);
  JNI_NULL_CHECK(env, j_field_delimiter, "field delimiter cannot be null", 0);
  JNI_NULL_CHECK(env, j_null_value, "null representation string cannot be itself null", 0);
  JNI_NULL_CHECK(env, j_buffer, "output buffer cannot be null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto data_sink =
      std::make_unique<cudf::jni::jni_writer_data_sink>(env, j_buffer, host_memory_allocator);

    auto const n_column_names = cudf::jni::native_jstringArray{env, j_column_names};
    auto const column_names   = n_column_names.as_cpp_vector();

    auto const line_terminator = cudf::jni::native_jstring{env, j_row_delimiter};
    auto const na_rep          = cudf::jni::native_jstring{env, j_null_value};
    auto const true_value      = cudf::jni::native_jstring{env, j_true_value};
    auto const false_value     = cudf::jni::native_jstring{env, j_false_value};
    auto const quote_style     = static_cast<cudf::io::quote_style>(j_quote_style);

    auto options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{data_sink.get()},
                                                         cudf::table_view{})
                     .names(column_names)
                     .include_header(static_cast<bool>(include_header))
                     .line_terminator(line_terminator)
                     .inter_column_delimiter(j_field_delimiter)
                     .na_rep(na_rep)
                     .true_value(true_value)
                     .false_value(false_value)
                     .quoting(quote_style)
                     .build();

    return ptr_as_jlong(new cudf::jni::io::csv_chunked_writer{options, data_sink});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeCSVChunkToBuffer(JNIEnv* env,
                                                                       jclass,
                                                                       jlong j_writer_handle,
                                                                       jlong j_table_handle)
{
  JNI_NULL_CHECK(env, j_writer_handle, "writer handle cannot be null.", );
  JNI_NULL_CHECK(env, j_table_handle, "table handle cannot be null.", );

  auto const table = reinterpret_cast<cudf::table_view*>(j_table_handle);
  auto writer      = reinterpret_cast<cudf::jni::io::csv_chunked_writer*>(j_writer_handle);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    writer->write(*table);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_endWriteCSVToBuffer(JNIEnv* env,
                                                                     jclass,
                                                                     jlong j_writer_handle)
{
  JNI_NULL_CHECK(env, j_writer_handle, "writer handle cannot be null.", );

  using cudf::jni::io::csv_chunked_writer;
  auto writer =
    std::unique_ptr<csv_chunked_writer>{reinterpret_cast<csv_chunked_writer*>(j_writer_handle)};

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    writer->close();
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_Table_readAndInferJSONFromDataSource(JNIEnv* env,
                                                         jclass,
                                                         jboolean day_first,
                                                         jboolean lines,
                                                         jboolean recover_with_null,
                                                         jboolean normalize_single_quotes,
                                                         jboolean normalize_whitespace,
                                                         jboolean mixed_types_as_string,
                                                         jboolean keep_quotes,
                                                         jboolean strict_validation,
                                                         jboolean allow_leading_zeros,
                                                         jboolean allow_nonnumeric_numbers,
                                                         jboolean allow_unquoted_control,
                                                         jboolean experimental,
                                                         jbyte line_delimiter,
                                                         jlong ds_handle)
{
  JNI_NULL_CHECK(env, ds_handle, "no data source handle given", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto ds = reinterpret_cast<cudf::io::datasource*>(ds_handle);
    cudf::io::source_info source{ds};

    auto const recovery_mode = recover_with_null ? cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL
                                                 : cudf::io::json_recovery_mode_t::FAIL;
    cudf::io::json_reader_options_builder opts =
      cudf::io::json_reader_options::builder(source)
        .dayfirst(static_cast<bool>(day_first))
        .lines(static_cast<bool>(lines))
        .recovery_mode(recovery_mode)
        .normalize_single_quotes(static_cast<bool>(normalize_single_quotes))
        .normalize_whitespace(static_cast<bool>(normalize_whitespace))
        .mixed_types_as_string(mixed_types_as_string)
        .delimiter(static_cast<char>(line_delimiter))
        .strict_validation(strict_validation)
        .experimental(experimental)
        .keep_quotes(keep_quotes)
        .prune_columns(false);
    if (strict_validation) {
      opts.numeric_leading_zeros(allow_leading_zeros)
        .nonnumeric_numbers(allow_nonnumeric_numbers)
        .unquoted_control_chars(allow_unquoted_control);
    }
    auto result =
      std::make_unique<cudf::io::table_with_metadata>(cudf::io::read_json(opts.build()));

    return reinterpret_cast<jlong>(result.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_Table_readAndInferJSON(JNIEnv* env,
                                           jclass,
                                           jlong buffer,
                                           jlong buffer_length,
                                           jboolean day_first,
                                           jboolean lines,
                                           jboolean recover_with_null,
                                           jboolean normalize_single_quotes,
                                           jboolean normalize_whitespace,
                                           jboolean mixed_types_as_string,
                                           jboolean keep_quotes,
                                           jboolean strict_validation,
                                           jboolean allow_leading_zeros,
                                           jboolean allow_nonnumeric_numbers,
                                           jboolean allow_unquoted_control,
                                           jboolean experimental,
                                           jbyte line_delimiter)
{
  JNI_NULL_CHECK(env, buffer, "buffer cannot be null", 0);
  if (buffer_length <= 0) {
    JNI_THROW_NEW(
      env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "An empty buffer is not supported", 0);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto source = cudf::io::source_info{cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buffer), static_cast<std::size_t>(buffer_length))};

    auto const recovery_mode = recover_with_null ? cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL
                                                 : cudf::io::json_recovery_mode_t::FAIL;
    cudf::io::json_reader_options_builder opts =
      cudf::io::json_reader_options::builder(source)
        .dayfirst(static_cast<bool>(day_first))
        .lines(static_cast<bool>(lines))
        .recovery_mode(recovery_mode)
        .normalize_single_quotes(static_cast<bool>(normalize_single_quotes))
        .normalize_whitespace(static_cast<bool>(normalize_whitespace))
        .strict_validation(strict_validation)
        .mixed_types_as_string(mixed_types_as_string)
        .prune_columns(false)
        .experimental(experimental)
        .delimiter(static_cast<char>(line_delimiter))
        .keep_quotes(keep_quotes);
    if (strict_validation) {
      opts.numeric_leading_zeros(allow_leading_zeros)
        .nonnumeric_numbers(allow_nonnumeric_numbers)
        .unquoted_control_chars(allow_unquoted_control);
    }

    auto result =
      std::make_unique<cudf::io::table_with_metadata>(cudf::io::read_json(opts.build()));

    return reinterpret_cast<jlong>(result.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_TableWithMeta_close(JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::io::table_with_metadata*>(handle);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jintArray JNICALL Java_ai_rapids_cudf_TableWithMeta_getFlattenedChildCounts(JNIEnv* env,
                                                                                      jclass,
                                                                                      jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto ptr = reinterpret_cast<cudf::io::table_with_metadata*>(handle);
    std::vector<int> counts;
    counts.push_back(ptr->metadata.schema_info.size());
    for (cudf::io::column_name_info const& child : ptr->metadata.schema_info) {
      cudf::jni::append_flattened_child_counts(child, counts);
    }

    auto length = counts.size();
    cudf::jni::native_jintArray ret(env, length);
    for (size_t i = 0; i < length; i++) {
      ret[i] = counts[i];
    }
    ret.commit();
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jobjectArray JNICALL
Java_ai_rapids_cudf_TableWithMeta_getFlattenedColumnNames(JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto ptr = reinterpret_cast<cudf::io::table_with_metadata*>(handle);
    std::vector<std::string> names;
    names.push_back("ROOT");
    for (cudf::io::column_name_info const& child : ptr->metadata.schema_info) {
      cudf::jni::append_flattened_child_names(child, names);
    }

    auto length = names.size();
    auto ret    = static_cast<jobjectArray>(
      env->NewObjectArray(length, env->FindClass("java/lang/String"), nullptr));
    for (size_t i = 0; i < length; i++) {
      env->SetObjectArrayElement(ret, i, env->NewStringUTF(names[i].c_str()));
    }

    return ret;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_TableWithMeta_releaseTable(JNIEnv* env,
                                                                            jclass,
                                                                            jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto ptr = reinterpret_cast<cudf::io::table_with_metadata*>(handle);
    if (ptr->tbl) {
      return convert_table_for_return(env, ptr->tbl);
    } else {
      return nullptr;
    }
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_Table_readJSONFromDataSource(JNIEnv* env,
                                                 jclass,
                                                 jintArray j_num_children,
                                                 jobjectArray col_names,
                                                 jintArray j_types,
                                                 jintArray j_scales,
                                                 jboolean day_first,
                                                 jboolean lines,
                                                 jboolean recover_with_null,
                                                 jboolean normalize_single_quotes,
                                                 jboolean normalize_whitespace,
                                                 jboolean mixed_types_as_string,
                                                 jboolean keep_quotes,
                                                 jboolean strict_validation,
                                                 jboolean allow_leading_zeros,
                                                 jboolean allow_nonnumeric_numbers,
                                                 jboolean allow_unquoted_control,
                                                 jboolean experimental,
                                                 jbyte line_delimiter,
                                                 jlong ds_handle)
{
  JNI_NULL_CHECK(env, ds_handle, "no data source handle given", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray n_col_names(env, col_names);
    cudf::jni::native_jintArray n_types(env, j_types);
    cudf::jni::native_jintArray n_scales(env, j_scales);
    cudf::jni::native_jintArray n_children(env, j_num_children);
    if (n_types.is_null() != n_scales.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match null", 0);
    }
    if (n_types.is_null() != n_col_names.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and names must match null", 0);
    }
    if (n_types.is_null() != n_children.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and num children must match null", 0);
    }

    auto ds = reinterpret_cast<cudf::io::datasource*>(ds_handle);
    cudf::io::source_info source{ds};

    cudf::io::json_recovery_mode_t recovery_mode =
      recover_with_null ? cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL
                        : cudf::io::json_recovery_mode_t::FAIL;

    cudf::io::json_reader_options_builder opts =
      cudf::io::json_reader_options::builder(source)
        .dayfirst(static_cast<bool>(day_first))
        .lines(static_cast<bool>(lines))
        .recovery_mode(recovery_mode)
        .normalize_single_quotes(static_cast<bool>(normalize_single_quotes))
        .normalize_whitespace(static_cast<bool>(normalize_whitespace))
        .mixed_types_as_string(mixed_types_as_string)
        .delimiter(static_cast<char>(line_delimiter))
        .strict_validation(strict_validation)
        .keep_quotes(keep_quotes)
        .experimental(experimental);
    if (strict_validation) {
      opts.numeric_leading_zeros(allow_leading_zeros)
        .nonnumeric_numbers(allow_nonnumeric_numbers)
        .unquoted_control_chars(allow_unquoted_control);
    }

    if (!n_types.is_null()) {
      if (n_types.size() != n_scales.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match size", 0);
      }
      if (n_col_names.size() != n_types.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and column names must match size", 0);
      }
      if (n_children.size() != n_types.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and num children must match size", 0);
      }

      std::map<std::string, cudf::io::schema_element> data_types;
      std::vector<std::string> name_order;
      int at = 0;
      while (at < n_types.size()) {
        std::string const name = n_col_names.get(at);
        data_types.insert(std::pair{
          name, cudf::jni::read_schema_element(at, n_children, n_col_names, n_types, n_scales)});
        name_order.push_back(name);
      }
      auto const prune_columns = data_types.size() != 0;
      cudf::io::schema_element structs{
        cudf::data_type{cudf::type_id::STRUCT}, std::move(data_types), {std::move(name_order)}};
      opts.prune_columns(prune_columns).dtypes(structs);

    } else {
      // should infer the types
    }

    auto result =
      std::make_unique<cudf::io::table_with_metadata>(cudf::io::read_json(opts.build()));

    return reinterpret_cast<jlong>(result.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_readJSON(JNIEnv* env,
                                                           jclass,
                                                           jintArray j_num_children,
                                                           jobjectArray col_names,
                                                           jintArray j_types,
                                                           jintArray j_scales,
                                                           jstring inputfilepath,
                                                           jlong buffer,
                                                           jlong buffer_length,
                                                           jboolean day_first,
                                                           jboolean lines,
                                                           jboolean recover_with_null,
                                                           jboolean normalize_single_quotes,
                                                           jboolean normalize_whitespace,
                                                           jboolean mixed_types_as_string,
                                                           jboolean keep_quotes,
                                                           jboolean strict_validation,
                                                           jboolean allow_leading_zeros,
                                                           jboolean allow_nonnumeric_numbers,
                                                           jboolean allow_unquoted_control,
                                                           jboolean experimental,
                                                           jbyte line_delimiter)
{
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", 0);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "cannot pass in both a buffer and an inputfilepath",
                  0);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(
      env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "An empty buffer is not supported", 0);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray n_col_names(env, col_names);
    cudf::jni::native_jintArray n_types(env, j_types);
    cudf::jni::native_jintArray n_scales(env, j_scales);
    cudf::jni::native_jintArray n_children(env, j_num_children);
    if (n_types.is_null() != n_scales.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match null", 0);
    }
    if (n_types.is_null() != n_col_names.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and names must match null", 0);
    }
    if (n_types.is_null() != n_children.is_null()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and num children must match null", 0);
    }

    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inputfilepath can't be empty", 0);
    }

    auto source =
      read_buffer
        ? cudf::io::source_info{cudf::host_span<std::byte const>(
            reinterpret_cast<std::byte const*>(buffer), static_cast<std::size_t>(buffer_length))}
        : cudf::io::source_info{filename};

    cudf::io::json_recovery_mode_t recovery_mode =
      recover_with_null ? cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL
                        : cudf::io::json_recovery_mode_t::FAIL;

    cudf::io::json_reader_options_builder opts =
      cudf::io::json_reader_options::builder(source)
        .dayfirst(static_cast<bool>(day_first))
        .lines(static_cast<bool>(lines))
        .recovery_mode(recovery_mode)
        .normalize_single_quotes(static_cast<bool>(normalize_single_quotes))
        .normalize_whitespace(static_cast<bool>(normalize_whitespace))
        .mixed_types_as_string(mixed_types_as_string)
        .delimiter(static_cast<char>(line_delimiter))
        .strict_validation(strict_validation)
        .keep_quotes(keep_quotes)
        .experimental(experimental);
    if (strict_validation) {
      opts.numeric_leading_zeros(allow_leading_zeros)
        .nonnumeric_numbers(allow_nonnumeric_numbers)
        .unquoted_control_chars(allow_unquoted_control);
    }

    if (!n_types.is_null()) {
      if (n_types.size() != n_scales.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and scales must match size", 0);
      }
      if (n_col_names.size() != n_types.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and column names must match size", 0);
      }
      if (n_children.size() != n_types.size()) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "types and num children must match size", 0);
      }

      std::map<std::string, cudf::io::schema_element> data_types;
      std::vector<std::string> name_order;
      name_order.reserve(n_types.size());
      int at = 0;
      while (at < n_types.size()) {
        std::string name = n_col_names.get(at);
        data_types.insert(std::pair{
          name, cudf::jni::read_schema_element(at, n_children, n_col_names, n_types, n_scales)});
        name_order.emplace_back(std::move(name));
      }
      auto const prune_columns = data_types.size() != 0;
      cudf::io::schema_element structs{
        cudf::data_type{cudf::type_id::STRUCT}, std::move(data_types), {std::move(name_order)}};
      opts.prune_columns(prune_columns).dtypes(structs);
    } else {
      // should infer the types
    }

    auto result =
      std::make_unique<cudf::io::table_with_metadata>(cudf::io::read_json(opts.build()));

    return reinterpret_cast<jlong>(result.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_readParquetFromDataSource(JNIEnv* env,
                                                    jclass,
                                                    jobjectArray filter_col_names,
                                                    jbooleanArray j_col_binary_read,
                                                    jint unit,
                                                    jlong ds_handle)
{
  JNI_NULL_CHECK(env, ds_handle, "no data source handle given", 0);
  JNI_NULL_CHECK(env, j_col_binary_read, "null col_binary_read", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);

    auto ds = reinterpret_cast<cudf::io::datasource*>(ds_handle);
    cudf::io::source_info source{ds};

    auto builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      builder = builder.column_names(n_filter_col_names.as_cpp_vector());
    }

    cudf::io::parquet_reader_options opts =
      builder.convert_strings_to_categories(false)
        .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
        // Ignore any missing projected column(s) by default
        .ignore_missing_columns(true)
        .build();
    return convert_table_for_return(env, cudf::io::read_parquet(opts).tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readParquet(JNIEnv* env,
                                                                   jclass,
                                                                   jobjectArray filter_col_names,
                                                                   jbooleanArray j_col_binary_read,
                                                                   jstring inputfilepath,
                                                                   jlongArray addrs_and_sizes,
                                                                   jint unit)
{
  JNI_NULL_CHECK(env, j_col_binary_read, "null col_binary_read", 0);
  bool read_buffer = true;
  if (addrs_and_sizes == nullptr) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "cannot pass in both a buffer and an inputfilepath",
                  NULL);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inputfilepath can't be empty", NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);
    cudf::jni::native_jlongArray n_addrs_sizes(env, addrs_and_sizes);
    std::unique_ptr<cudf::io::datasource> multi_buffer_source;
    cudf::io::source_info source;
    if (read_buffer) {
      multi_buffer_source.reset(new cudf::jni::multi_host_buffer_source(n_addrs_sizes));
      source = cudf::io::source_info(multi_buffer_source.get());
    } else {
      source = cudf::io::source_info(filename);
    }

    auto builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      builder = builder.column_names(n_filter_col_names.as_cpp_vector());
    }

    cudf::io::parquet_reader_options opts =
      builder.convert_strings_to_categories(false)
        .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
        // Ignore any missing projected column(s) by default
        .ignore_missing_columns(true)
        .build();
    auto tbl = cudf::io::read_parquet(opts).tbl;
    n_col_binary_read.cancel();
    n_addrs_sizes.cancel();
    return convert_table_for_return(env, tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readAvroFromDataSource(
  JNIEnv* env, jclass, jobjectArray filter_col_names, jlong ds_handle)
{
  JNI_NULL_CHECK(env, ds_handle, "no data source handle given", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    auto ds = reinterpret_cast<cudf::io::datasource*>(ds_handle);
    cudf::io::source_info source{ds};

    cudf::io::avro_reader_options opts = cudf::io::avro_reader_options::builder(source)
                                           .columns(n_filter_col_names.as_cpp_vector())
                                           .build();
    return convert_table_for_return(env, cudf::io::read_avro(opts).tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_readAvro(JNIEnv* env,
                                                                jclass,
                                                                jobjectArray filter_col_names,
                                                                jstring inputfilepath,
                                                                jlong buffer,
                                                                jlong buffer_length)
{
  bool const read_buffer = (buffer != 0);
  if (!read_buffer) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "cannot pass in both a buffer and an inputfilepath",
                  NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(
      env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "An empty buffer is not supported", NULL);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inputfilepath can't be empty", NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    auto source =
      read_buffer
        ? cudf::io::source_info(cudf::host_span<std::byte const>(
            reinterpret_cast<std::byte const*>(buffer), static_cast<std::size_t>(buffer_length)))
        : cudf::io::source_info(filename);

    cudf::io::avro_reader_options opts = cudf::io::avro_reader_options::builder(source)
                                           .columns(n_filter_col_names.as_cpp_vector())
                                           .build();
    return convert_table_for_return(env, cudf::io::read_avro(opts).tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT long JNICALL
Java_ai_rapids_cudf_Table_writeParquetBufferBegin(JNIEnv* env,
                                                  jclass,
                                                  jobjectArray j_col_names,
                                                  jint j_num_children,
                                                  jintArray j_children,
                                                  jbooleanArray j_col_nullability,
                                                  jobjectArray j_metadata_keys,
                                                  jobjectArray j_metadata_values,
                                                  jint j_compression,
                                                  jint j_row_group_size_rows,
                                                  jlong j_row_group_size_bytes,
                                                  jlong j_max_dictionary_size,
                                                  jint j_dictionary_policy,
                                                  jint j_stats_freq,
                                                  jbooleanArray j_isInt96,
                                                  jintArray j_precisions,
                                                  jbooleanArray j_is_map,
                                                  jbooleanArray j_is_binary,
                                                  jbooleanArray j_hasParquetFieldIds,
                                                  jintArray j_parquetFieldIds,
                                                  jobject consumer,
                                                  jobject host_memory_allocator)
{
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, consumer, "null consumer", 0);
  JNI_TRY
  {
    std::unique_ptr<cudf::jni::jni_writer_data_sink> data_sink(
      new cudf::jni::jni_writer_data_sink(env, consumer, host_memory_allocator));

    using namespace cudf::io;
    using namespace cudf::jni;
    sink_info sink{data_sink.get()};
    table_input_metadata metadata;
    createTableMetaData(env,
                        j_num_children,
                        j_col_names,
                        j_children,
                        j_col_nullability,
                        j_isInt96,
                        j_precisions,
                        j_is_map,
                        metadata,
                        j_hasParquetFieldIds,
                        j_parquetFieldIds,
                        j_is_binary);

    auto meta_keys   = cudf::jni::native_jstringArray{env, j_metadata_keys}.as_cpp_vector();
    auto meta_values = cudf::jni::native_jstringArray{env, j_metadata_values}.as_cpp_vector();

    std::map<std::string, std::string> kv_metadata;
    std::transform(meta_keys.begin(),
                   meta_keys.end(),
                   meta_values.begin(),
                   std::inserter(kv_metadata, kv_metadata.end()),
                   [](auto const& key, auto const& value) {
                     // The metadata value will be ignored if it is empty.
                     // We modify it into a space character to workaround such issue.
                     return std::make_pair(key, value.empty() ? std::string(" ") : value);
                   });

    auto stats = std::make_shared<cudf::io::writer_compression_statistics>();
    chunked_parquet_writer_options opts =
      chunked_parquet_writer_options::builder(sink)
        .metadata(std::move(metadata))
        .compression(static_cast<compression_type>(j_compression))
        .row_group_size_rows(j_row_group_size_rows)
        .row_group_size_bytes(j_row_group_size_bytes)
        .max_dictionary_size(j_max_dictionary_size)
        .dictionary_policy(static_cast<cudf::io::dictionary_policy>(j_dictionary_policy))
        .stats_level(static_cast<statistics_freq>(j_stats_freq))
        .key_value_metadata({kv_metadata})
        .compression_statistics(stats)
        .build();
    auto writer_ptr = std::make_unique<cudf::io::chunked_parquet_writer>(opts);
    cudf::jni::native_parquet_writer_handle* ret = new cudf::jni::native_parquet_writer_handle(
      std::move(writer_ptr), std::move(data_sink), std::move(stats));
    return ptr_as_jlong(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT long JNICALL
Java_ai_rapids_cudf_Table_writeParquetFileBegin(JNIEnv* env,
                                                jclass,
                                                jobjectArray j_col_names,
                                                jint j_num_children,
                                                jintArray j_children,
                                                jbooleanArray j_col_nullability,
                                                jobjectArray j_metadata_keys,
                                                jobjectArray j_metadata_values,
                                                jint j_compression,
                                                jint j_row_group_size_rows,
                                                jlong j_row_group_size_bytes,
                                                jlong j_max_dictionary_size,
                                                jint j_dictionary_policy,
                                                jint j_stats_freq,
                                                jbooleanArray j_isInt96,
                                                jintArray j_precisions,
                                                jbooleanArray j_is_map,
                                                jbooleanArray j_is_binary,
                                                jbooleanArray j_hasParquetFieldIds,
                                                jintArray j_parquetFieldIds,
                                                jstring j_output_path)
{
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, j_output_path, "null output path", 0);
  JNI_TRY
  {
    cudf::jni::native_jstring output_path(env, j_output_path);

    using namespace cudf::io;
    using namespace cudf::jni;
    table_input_metadata metadata;
    createTableMetaData(env,
                        j_num_children,
                        j_col_names,
                        j_children,
                        j_col_nullability,
                        j_isInt96,
                        j_precisions,
                        j_is_map,
                        metadata,
                        j_hasParquetFieldIds,
                        j_parquetFieldIds,
                        j_is_binary);

    auto meta_keys   = cudf::jni::native_jstringArray{env, j_metadata_keys}.as_cpp_vector();
    auto meta_values = cudf::jni::native_jstringArray{env, j_metadata_values}.as_cpp_vector();

    std::map<std::string, std::string> kv_metadata;
    std::transform(meta_keys.begin(),
                   meta_keys.end(),
                   meta_values.begin(),
                   std::inserter(kv_metadata, kv_metadata.end()),
                   [](auto const& key, auto const& value) {
                     // The metadata value will be ignored if it is empty.
                     // We modify it into a space character to workaround such issue.
                     return std::make_pair(key, value.empty() ? std::string(" ") : value);
                   });

    sink_info sink{output_path};
    auto stats = std::make_shared<cudf::io::writer_compression_statistics>();
    chunked_parquet_writer_options opts =
      chunked_parquet_writer_options::builder(sink)
        .metadata(std::move(metadata))
        .compression(static_cast<compression_type>(j_compression))
        .row_group_size_rows(j_row_group_size_rows)
        .row_group_size_bytes(j_row_group_size_bytes)
        .max_dictionary_size(j_max_dictionary_size)
        .dictionary_policy(static_cast<cudf::io::dictionary_policy>(j_dictionary_policy))
        .stats_level(static_cast<statistics_freq>(j_stats_freq))
        .key_value_metadata({kv_metadata})
        .compression_statistics(stats)
        .build();

    auto writer_ptr = std::make_unique<cudf::io::chunked_parquet_writer>(opts);
    cudf::jni::native_parquet_writer_handle* ret =
      new cudf::jni::native_parquet_writer_handle(std::move(writer_ptr), nullptr, std::move(stats));
    return ptr_as_jlong(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeParquetChunk(
  JNIEnv* env, jclass, jlong j_state, jlong j_table, jlong mem_size)
{
  JNI_NULL_CHECK(env, j_table, "null table", );
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::table_view* tview_with_empty_nullmask = reinterpret_cast<cudf::table_view*>(j_table);
  cudf::table_view tview = cudf::jni::remove_validity_if_needed(tview_with_empty_nullmask);
  cudf::jni::native_parquet_writer_handle* state =
    reinterpret_cast<cudf::jni::native_parquet_writer_handle*>(j_state);

  if (state->sink) {
    long alloc_size = std::max(cudf::jni::MINIMUM_WRITE_BUFFER_SIZE, mem_size / 2);
    state->sink->set_alloc_size(alloc_size);
  }
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    state->writer->write(tview);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeParquetEnd(JNIEnv* env, jclass, jlong j_state)
{
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::jni::native_parquet_writer_handle* state =
    reinterpret_cast<cudf::jni::native_parquet_writer_handle*>(j_state);
  std::unique_ptr<cudf::jni::native_parquet_writer_handle> make_sure_we_delete(state);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    state->writer->close();
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_readORCFromDataSource(JNIEnv* env,
                                                jclass,
                                                jobjectArray filter_col_names,
                                                jboolean usingNumPyTypes,
                                                jint unit,
                                                jobjectArray dec128_col_names,
                                                jboolean ignore_timezone_in_stripe_footer,
                                                jlong ds_handle)
{
  JNI_NULL_CHECK(env, ds_handle, "no data source handle given", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    cudf::jni::native_jstringArray n_dec128_col_names(env, dec128_col_names);

    auto ds = reinterpret_cast<cudf::io::datasource*>(ds_handle);
    cudf::io::source_info source{ds};

    auto builder = cudf::io::orc_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      builder = builder.columns(n_filter_col_names.as_cpp_vector());
    }

    cudf::io::orc_reader_options opts =
      builder.use_index(false)
        .use_np_dtypes(static_cast<bool>(usingNumPyTypes))
        .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
        .decimal128_columns(n_dec128_col_names.as_cpp_vector())
        .ignore_timezone_in_stripe_footer(ignore_timezone_in_stripe_footer)
        .build();
    return convert_table_for_return(env, cudf::io::read_orc(opts).tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_readORC(JNIEnv* env,
                                  jclass,
                                  jobjectArray filter_col_names,
                                  jstring inputfilepath,
                                  jlong buffer,
                                  jlong buffer_length,
                                  jboolean usingNumPyTypes,
                                  jint unit,
                                  jobjectArray dec128_col_names,
                                  jboolean ignore_timezone_in_stripe_footer)
{
  bool read_buffer = true;
  if (buffer == 0) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "cannot pass in both a buffer and an inputfilepath",
                  NULL);
  } else if (buffer_length <= 0) {
    JNI_THROW_NEW(
      env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "An empty buffer is not supported", NULL);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inputfilepath can't be empty", NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    cudf::jni::native_jstringArray n_dec128_col_names(env, dec128_col_names);

    auto source = read_buffer ? cudf::io::source_info(cudf::host_span<std::byte const>(
                                  reinterpret_cast<std::byte const*>(buffer), buffer_length))
                              : cudf::io::source_info(filename);

    auto builder = cudf::io::orc_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      builder = builder.columns(n_filter_col_names.as_cpp_vector());
    }

    cudf::io::orc_reader_options opts =
      builder.use_index(false)
        .use_np_dtypes(static_cast<bool>(usingNumPyTypes))
        .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
        .decimal128_columns(n_dec128_col_names.as_cpp_vector())
        .ignore_timezone_in_stripe_footer(ignore_timezone_in_stripe_footer)
        .build();
    return convert_table_for_return(env, cudf::io::read_orc(opts).tbl);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT long JNICALL
Java_ai_rapids_cudf_Table_writeORCBufferBegin(JNIEnv* env,
                                              jclass,
                                              jobjectArray j_col_names,
                                              jint j_num_children,
                                              jintArray j_children,
                                              jbooleanArray j_col_nullability,
                                              jobjectArray j_metadata_keys,
                                              jobjectArray j_metadata_values,
                                              jint j_compression,
                                              jintArray j_precisions,
                                              jbooleanArray j_is_map,
                                              jint j_stripe_size_rows,
                                              jobject consumer,
                                              jobject host_memory_allocator)
{
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, consumer, "null consumer", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using namespace cudf::io;
    using namespace cudf::jni;
    table_input_metadata metadata;
    // ORC has no `j_is_int96`, but `createTableMetaData` needs a lvalue.
    jbooleanArray j_is_int96 = NULL;
    // ORC has no `j_parquetFieldIds`, but `createTableMetaData` needs a lvalue.
    jbooleanArray j_hasParquetFieldIds = NULL;
    jintArray j_parquetFieldIds        = NULL;
    // temp stub
    jbooleanArray j_is_binary = NULL;

    createTableMetaData(env,
                        j_num_children,
                        j_col_names,
                        j_children,
                        j_col_nullability,
                        j_is_int96,
                        j_precisions,
                        j_is_map,
                        metadata,
                        j_hasParquetFieldIds,
                        j_parquetFieldIds,
                        j_is_binary);

    auto meta_keys   = cudf::jni::native_jstringArray{env, j_metadata_keys}.as_cpp_vector();
    auto meta_values = cudf::jni::native_jstringArray{env, j_metadata_values}.as_cpp_vector();

    std::map<std::string, std::string> kv_metadata;
    std::transform(meta_keys.begin(),
                   meta_keys.end(),
                   meta_values.begin(),
                   std::inserter(kv_metadata, kv_metadata.end()),
                   [](std::string const& k, std::string const& v) { return std::make_pair(k, v); });

    std::unique_ptr<cudf::jni::jni_writer_data_sink> data_sink(
      new cudf::jni::jni_writer_data_sink(env, consumer, host_memory_allocator));
    sink_info sink{data_sink.get()};

    auto stats                      = std::make_shared<cudf::io::writer_compression_statistics>();
    chunked_orc_writer_options opts = chunked_orc_writer_options::builder(sink)
                                        .metadata(std::move(metadata))
                                        .compression(static_cast<compression_type>(j_compression))
                                        .enable_statistics(ORC_STATISTICS_ROW_GROUP)
                                        .key_value_metadata(kv_metadata)
                                        .compression_statistics(stats)
                                        .stripe_size_rows(j_stripe_size_rows)
                                        .build();
    auto writer_ptr                          = std::make_unique<cudf::io::orc_chunked_writer>(opts);
    cudf::jni::native_orc_writer_handle* ret = new cudf::jni::native_orc_writer_handle(
      std::move(writer_ptr), std::move(data_sink), std::move(stats));
    return ptr_as_jlong(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeORCFileBegin(JNIEnv* env,
                                                                   jclass,
                                                                   jobjectArray j_col_names,
                                                                   jint j_num_children,
                                                                   jintArray j_children,
                                                                   jbooleanArray j_col_nullability,
                                                                   jobjectArray j_metadata_keys,
                                                                   jobjectArray j_metadata_values,
                                                                   jint j_compression,
                                                                   jintArray j_precisions,
                                                                   jbooleanArray j_is_map,
                                                                   jint j_stripe_size_rows,
                                                                   jstring j_output_path)
{
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_col_nullability, "null nullability", 0);
  JNI_NULL_CHECK(env, j_metadata_keys, "null metadata keys", 0);
  JNI_NULL_CHECK(env, j_metadata_values, "null metadata values", 0);
  JNI_NULL_CHECK(env, j_output_path, "null output path", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    using namespace cudf::io;
    using namespace cudf::jni;
    cudf::jni::native_jstring output_path(env, j_output_path);
    table_input_metadata metadata;
    // ORC has no `j_is_int96`, but `createTableMetaData` needs a lvalue.
    jbooleanArray j_is_int96 = NULL;
    // ORC has no `j_parquetFieldIds`, but `createTableMetaData` needs a lvalue.
    jbooleanArray j_hasParquetFieldIds = NULL;
    jintArray j_parquetFieldIds        = NULL;
    // temp stub
    jbooleanArray j_is_binary = NULL;
    createTableMetaData(env,
                        j_num_children,
                        j_col_names,
                        j_children,
                        j_col_nullability,
                        j_is_int96,
                        j_precisions,
                        j_is_map,
                        metadata,
                        j_hasParquetFieldIds,
                        j_parquetFieldIds,
                        j_is_binary);

    auto meta_keys   = cudf::jni::native_jstringArray{env, j_metadata_keys}.as_cpp_vector();
    auto meta_values = cudf::jni::native_jstringArray{env, j_metadata_values}.as_cpp_vector();

    std::map<std::string, std::string> kv_metadata;
    std::transform(meta_keys.begin(),
                   meta_keys.end(),
                   meta_values.begin(),
                   std::inserter(kv_metadata, kv_metadata.end()),
                   [](std::string const& k, std::string const& v) { return std::make_pair(k, v); });

    sink_info sink{output_path};
    auto stats                      = std::make_shared<cudf::io::writer_compression_statistics>();
    chunked_orc_writer_options opts = chunked_orc_writer_options::builder(sink)
                                        .metadata(std::move(metadata))
                                        .compression(static_cast<compression_type>(j_compression))
                                        .enable_statistics(ORC_STATISTICS_ROW_GROUP)
                                        .key_value_metadata(kv_metadata)
                                        .compression_statistics(stats)
                                        .stripe_size_rows(j_stripe_size_rows)
                                        .build();
    auto writer_ptr = std::make_unique<cudf::io::orc_chunked_writer>(opts);
    cudf::jni::native_orc_writer_handle* ret =
      new cudf::jni::native_orc_writer_handle(std::move(writer_ptr), nullptr, std::move(stats));
    return ptr_as_jlong(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeORCChunk(
  JNIEnv* env, jclass, jlong j_state, jlong j_table, jlong mem_size)
{
  JNI_NULL_CHECK(env, j_table, "null table", );
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::table_view* tview_orig = reinterpret_cast<cudf::table_view*>(j_table);
  cudf::table_view tview       = cudf::jni::remove_validity_if_needed(tview_orig);
  cudf::jni::native_orc_writer_handle* state =
    reinterpret_cast<cudf::jni::native_orc_writer_handle*>(j_state);

  if (state->sink) {
    long alloc_size = std::max(cudf::jni::MINIMUM_WRITE_BUFFER_SIZE, mem_size / 2);
    state->sink->set_alloc_size(alloc_size);
  }
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    state->writer->write(tview);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeORCEnd(JNIEnv* env, jclass, jlong j_state)
{
  JNI_NULL_CHECK(env, j_state, "null state", );

  using namespace cudf::io;
  cudf::jni::native_orc_writer_handle* state =
    reinterpret_cast<cudf::jni::native_orc_writer_handle*>(j_state);
  std::unique_ptr<cudf::jni::native_orc_writer_handle> make_sure_we_delete(state);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    state->writer->close();
  }
  JNI_CATCH(env, );
}

JNIEXPORT jdoubleArray JNICALL Java_ai_rapids_cudf_TableWriter_getWriteStatistics(JNIEnv* env,
                                                                                  jclass,
                                                                                  jlong j_state)
{
  JNI_NULL_CHECK(env, j_state, "null state", nullptr);

  using namespace cudf::io;
  auto const state = reinterpret_cast<cudf::jni::jni_table_writer_handle_base const*>(j_state);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    if (!state->stats) { return nullptr; }

    auto const& stats = *state->stats;
    auto output       = cudf::jni::native_jdoubleArray(env, 4);
    output[0]         = static_cast<jdouble>(stats.num_compressed_bytes());
    output[1]         = static_cast<jdouble>(stats.num_failed_bytes());
    output[2]         = static_cast<jdouble>(stats.num_skipped_bytes());
    output[3]         = static_cast<jdouble>(stats.compression_ratio());

    return output.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCBufferBegin(
  JNIEnv* env, jclass, jobjectArray j_col_names, jobject consumer, jobject host_memory_allocator)
{
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, consumer, "null consumer", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray col_names(env, j_col_names);

    std::shared_ptr<cudf::jni::jni_arrow_output_stream> data_sink(
      new cudf::jni::jni_arrow_output_stream(env, consumer, host_memory_allocator));

    cudf::jni::native_arrow_ipc_writer_handle* ret =
      new cudf::jni::native_arrow_ipc_writer_handle(col_names.as_cpp_vector(), data_sink);
    return ptr_as_jlong(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCFileBegin(JNIEnv* env,
                                                                        jclass,
                                                                        jobjectArray j_col_names,
                                                                        jstring j_output_path)
{
  JNI_NULL_CHECK(env, j_col_names, "null columns", 0);
  JNI_NULL_CHECK(env, j_output_path, "null output path", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray col_names(env, j_col_names);
    cudf::jni::native_jstring output_path(env, j_output_path);

    cudf::jni::native_arrow_ipc_writer_handle* ret =
      new cudf::jni::native_arrow_ipc_writer_handle(col_names.as_cpp_vector(), output_path);
    return ptr_as_jlong(ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_convertCudfToArrowTable(JNIEnv* env,
                                                                          jclass,
                                                                          jlong j_state,
                                                                          jlong j_table)
{
  JNI_NULL_CHECK(env, j_table, "null table", 0);
  JNI_NULL_CHECK(env, j_state, "null state", 0);

  cudf::table_view* tview = reinterpret_cast<cudf::table_view*>(j_table);
  cudf::jni::native_arrow_ipc_writer_handle* state =
    reinterpret_cast<cudf::jni::native_arrow_ipc_writer_handle*>(j_state);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    // The semantics of this function are confusing:
    // The return value is a pointer to a heap-allocated shared_ptr<arrow::Table>.
    // i.e. the shared_ptr<> is on the heap.
    // The pointer to the shared_ptr<> is returned as a jlong.
    using result_t = std::shared_ptr<arrow::Table>;

    auto got_arrow_schema = cudf::to_arrow_schema(*tview, state->get_column_metadata(*tview));
    cudf::jni::set_nullable(got_arrow_schema.get());
    auto got_arrow_array = cudf::to_arrow_host(*tview);
    auto batch =
      arrow::ImportRecordBatch(&got_arrow_array->array, got_arrow_schema.get()).ValueOrDie();
    auto result = arrow::Table::FromRecordBatches({batch}).ValueOrDie();

    return ptr_as_jlong(new result_t{result});
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCArrowChunk(
  JNIEnv* env, jclass, jlong j_state, jlong arrow_table_handle, jlong max_chunk)
{
  JNI_NULL_CHECK(env, arrow_table_handle, "null arrow table", );
  JNI_NULL_CHECK(env, j_state, "null state", );

  std::shared_ptr<arrow::Table>* handle =
    reinterpret_cast<std::shared_ptr<arrow::Table>*>(arrow_table_handle);
  cudf::jni::native_arrow_ipc_writer_handle* state =
    reinterpret_cast<cudf::jni::native_arrow_ipc_writer_handle*>(j_state);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    state->write(*handle, max_chunk);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_writeArrowIPCEnd(JNIEnv* env,
                                                                  jclass,
                                                                  jlong j_state)
{
  JNI_NULL_CHECK(env, j_state, "null state", );

  cudf::jni::native_arrow_ipc_writer_handle* state =
    reinterpret_cast<cudf::jni::native_arrow_ipc_writer_handle*>(j_state);
  std::unique_ptr<cudf::jni::native_arrow_ipc_writer_handle> make_sure_we_delete(state);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    state->close();
  }
  JNI_CATCH(env, );
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_readArrowIPCFileBegin(JNIEnv* env,
                                                                       jclass,
                                                                       jstring j_input_path)
{
  JNI_NULL_CHECK(env, j_input_path, "null input path", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring input_path(env, j_input_path);
    return ptr_as_jlong(new cudf::jni::native_arrow_ipc_reader_handle(input_path));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT long JNICALL Java_ai_rapids_cudf_Table_readArrowIPCBufferBegin(JNIEnv* env,
                                                                         jclass,
                                                                         jobject provider)
{
  JNI_NULL_CHECK(env, provider, "null provider", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::shared_ptr<cudf::jni::jni_arrow_input_stream> data_source(
      new cudf::jni::jni_arrow_input_stream(env, provider));
    return ptr_as_jlong(new cudf::jni::native_arrow_ipc_reader_handle(data_source));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_readArrowIPCChunkToArrowTable(JNIEnv* env,
                                                                                jclass,
                                                                                jlong j_state,
                                                                                jint row_target)
{
  JNI_NULL_CHECK(env, j_state, "null state", 0);

  cudf::jni::native_arrow_ipc_reader_handle* state =
    reinterpret_cast<cudf::jni::native_arrow_ipc_reader_handle*>(j_state);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    // This is a little odd because we have to return a pointer
    // and arrow wants to deal with shared pointers for everything.
    auto result = state->next(row_target);
    return result ? ptr_as_jlong(new std::shared_ptr<arrow::Table>{result}) : 0;
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_closeArrowTable(JNIEnv* env,
                                                                 jclass,
                                                                 jlong arrow_table_handle)
{
  std::shared_ptr<arrow::Table>* handle =
    reinterpret_cast<std::shared_ptr<arrow::Table>*>(arrow_table_handle);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete handle;
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_convertArrowTableToCudf(JNIEnv* env, jclass, jlong arrow_table_handle)
{
  JNI_NULL_CHECK(env, arrow_table_handle, "null arrow handle", 0);

  std::shared_ptr<arrow::Table>* handle =
    reinterpret_cast<std::shared_ptr<arrow::Table>*>(arrow_table_handle);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    ArrowSchema sch;
    if (!arrow::ExportSchema(*handle->get()->schema(), &sch).ok()) {
      JNI_THROW_NEW(env, cudf::jni::RUNTIME_EXCEPTION_CLASS, "Unable to produce an ArrowSchema", 0)
    }
    auto batch = handle->get()->CombineChunksToBatch().ValueOrDie();
    ArrowArray arr;
    if (!arrow::ExportRecordBatch(*batch, &arr).ok()) {
      JNI_THROW_NEW(env, cudf::jni::RUNTIME_EXCEPTION_CLASS, "Unable to produce an ArrowArray", 0)
    }
    auto ret = cudf::from_arrow(&sch, &arr);
    arr.release(&arr);
    sch.release(&sch);

    return convert_table_for_return(env, ret);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_Table_readArrowIPCEnd(JNIEnv* env, jclass, jlong j_state)
{
  JNI_NULL_CHECK(env, j_state, "null state", );

  cudf::jni::native_arrow_ipc_reader_handle* state =
    reinterpret_cast<cudf::jni::native_arrow_ipc_reader_handle*>(j_state);
  std::unique_ptr<cudf::jni::native_arrow_ipc_reader_handle> make_sure_we_delete(state);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    state->close();
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  return cudf::jni::join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    compare_nulls_equal,
    [](cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
      return cudf::left_join(left, right, nulleq);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftDistinctJoinGatherMap(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  double const load_factor = 0.5;
  return cudf::jni::join_gather_single_map(
    env,
    j_left_keys,
    j_right_keys,
    compare_nulls_equal,
    [load_factor](
      cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
      cudf::distinct_hash_join hash(right, nulleq, load_factor);
      return hash.left_join(left);
    });
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_leftJoinRowCount(JNIEnv* env,
                                                                   jclass,
                                                                   jlong j_left_table,
                                                                   jlong j_right_hash_join)
{
  JNI_NULL_CHECK(env, j_left_table, "left table is null", 0);
  JNI_NULL_CHECK(env, j_right_hash_join, "right hash join is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto hash_join  = reinterpret_cast<cudf::hash_join const*>(j_right_hash_join);
    auto row_count  = hash_join->left_join_size(*left_table);
    return static_cast<jlong>(row_count);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftHashJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_hash_join)
{
  return cudf::jni::hash_join_gather_maps(
    env,
    j_left_table,
    j_right_hash_join,
    [](cudf::table_view const& left, cudf::hash_join const& hash) { return hash.left_join(left); });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftHashJoinGatherMapsWithCount(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_hash_join, jlong j_output_row_count)
{
  auto output_row_count = static_cast<std::size_t>(j_output_row_count);
  return cudf::jni::hash_join_gather_maps(
    env,
    j_left_table,
    j_right_hash_join,
    [output_row_count](cudf::table_view const& left, cudf::hash_join const& hash) {
      return hash.left_join(left, output_row_count);
    });
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_conditionalLeftJoinRowCount(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", 0);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto row_count =
      cudf::conditional_left_join_size(*left_table, *right_table, condition->get_top_expression());
    return static_cast<jlong>(row_count);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_conditionalLeftJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  return cudf::jni::cond_join_gather_maps(env,
                                          j_left_table,
                                          j_right_table,
                                          j_condition,
                                          [](cudf::table_view const& left,
                                             cudf::table_view const& right,
                                             cudf::ast::expression const& cond_expr) {
                                            return cudf::conditional_left_join(
                                              left, right, cond_expr);
                                          });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_conditionalLeftJoinGatherMapsWithCount(JNIEnv* env,
                                                                 jclass,
                                                                 jlong j_left_table,
                                                                 jlong j_right_table,
                                                                 jlong j_condition,
                                                                 jlong j_row_count)
{
  auto row_count = static_cast<std::size_t>(j_row_count);
  return cudf::jni::cond_join_gather_maps(env,
                                          j_left_table,
                                          j_right_table,
                                          j_condition,
                                          [row_count](cudf::table_view const& left,
                                                      cudf::table_view const& right,
                                                      cudf::ast::expression const& cond_expr) {
                                            return cudf::conditional_left_join(
                                              left, right, cond_expr, row_count);
                                          });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_mixedLeftJoinSize(JNIEnv* env,
                                                                         jclass,
                                                                         jlong j_left_keys,
                                                                         jlong j_right_keys,
                                                                         jlong j_left_condition,
                                                                         jlong j_right_condition,
                                                                         jlong j_condition,
                                                                         jboolean j_nulls_equal)
{
  return cudf::jni::mixed_join_size(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [](cudf::table_view const& left_keys,
       cudf::table_view const& right_keys,
       cudf::table_view const& left_condition,
       cudf::table_view const& right_condition,
       cudf::ast::expression const& condition,
       cudf::null_equality nulls_equal) {
      return cudf::mixed_left_join_size(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal);
    });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_mixedLeftJoinGatherMaps(JNIEnv* env,
                                                  jclass,
                                                  jlong j_left_keys,
                                                  jlong j_right_keys,
                                                  jlong j_left_condition,
                                                  jlong j_right_condition,
                                                  jlong j_condition,
                                                  jboolean j_nulls_equal)
{
  return cudf::jni::mixed_join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [](cudf::table_view const& left_keys,
       cudf::table_view const& right_keys,
       cudf::table_view const& left_condition,
       cudf::table_view const& right_condition,
       cudf::ast::expression const& condition,
       cudf::null_equality nulls_equal) {
      return cudf::mixed_left_join(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal);
    });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_mixedLeftJoinGatherMapsWithSize(JNIEnv* env,
                                                          jclass,
                                                          jlong j_left_keys,
                                                          jlong j_right_keys,
                                                          jlong j_left_condition,
                                                          jlong j_right_condition,
                                                          jlong j_condition,
                                                          jboolean j_nulls_equal,
                                                          jlong j_output_row_count,
                                                          jlong j_matches_view)
{
  auto size_info = cudf::jni::get_mixed_size_info(env, j_output_row_count, j_matches_view);
  return cudf::jni::mixed_join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [&size_info](cudf::table_view const& left_keys,
                 cudf::table_view const& right_keys,
                 cudf::table_view const& left_condition,
                 cudf::table_view const& right_condition,
                 cudf::ast::expression const& condition,
                 cudf::null_equality nulls_equal) {
      return cudf::mixed_left_join(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal, size_info);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  return cudf::jni::join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    compare_nulls_equal,
    [](cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
      return cudf::inner_join(left, right, nulleq);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerDistinctJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  double const load_factor = 0.5;
  return cudf::jni::join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    compare_nulls_equal,
    [load_factor](
      cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
      cudf::distinct_hash_join hash(right, nulleq, load_factor);
      return hash.inner_join(left);
    });
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_innerJoinRowCount(JNIEnv* env,
                                                                    jclass,
                                                                    jlong j_left_table,
                                                                    jlong j_right_hash_join)
{
  JNI_NULL_CHECK(env, j_left_table, "left table is null", 0);
  JNI_NULL_CHECK(env, j_right_hash_join, "right hash join is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto hash_join  = reinterpret_cast<cudf::hash_join const*>(j_right_hash_join);
    auto row_count  = hash_join->inner_join_size(*left_table);
    return static_cast<jlong>(row_count);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerHashJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_hash_join)
{
  return cudf::jni::hash_join_gather_maps(
    env,
    j_left_table,
    j_right_hash_join,
    [](cudf::table_view const& left, cudf::hash_join const& hash) {
      return hash.inner_join(left);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_innerHashJoinGatherMapsWithCount(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_hash_join, jlong j_output_row_count)
{
  auto output_row_count = static_cast<std::size_t>(j_output_row_count);
  return cudf::jni::hash_join_gather_maps(
    env,
    j_left_table,
    j_right_hash_join,
    [output_row_count](cudf::table_view const& left, cudf::hash_join const& hash) {
      return hash.inner_join(left, output_row_count);
    });
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_conditionalInnerJoinRowCount(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", 0);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto row_count =
      cudf::conditional_inner_join_size(*left_table, *right_table, condition->get_top_expression());
    return static_cast<jlong>(row_count);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_conditionalInnerJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  return cudf::jni::cond_join_gather_maps(env,
                                          j_left_table,
                                          j_right_table,
                                          j_condition,
                                          [](cudf::table_view const& left,
                                             cudf::table_view const& right,
                                             cudf::ast::expression const& cond_expr) {
                                            return cudf::conditional_inner_join(
                                              left, right, cond_expr);
                                          });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_conditionalInnerJoinGatherMapsWithCount(JNIEnv* env,
                                                                  jclass,
                                                                  jlong j_left_table,
                                                                  jlong j_right_table,
                                                                  jlong j_condition,
                                                                  jlong j_row_count)
{
  auto row_count = static_cast<std::size_t>(j_row_count);
  return cudf::jni::cond_join_gather_maps(env,
                                          j_left_table,
                                          j_right_table,
                                          j_condition,
                                          [row_count](cudf::table_view const& left,
                                                      cudf::table_view const& right,
                                                      cudf::ast::expression const& cond_expr) {
                                            return cudf::conditional_inner_join(
                                              left, right, cond_expr, row_count);
                                          });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_mixedInnerJoinSize(JNIEnv* env,
                                                                          jclass,
                                                                          jlong j_left_keys,
                                                                          jlong j_right_keys,
                                                                          jlong j_left_condition,
                                                                          jlong j_right_condition,
                                                                          jlong j_condition,
                                                                          jboolean j_nulls_equal)
{
  return cudf::jni::mixed_join_size(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [](cudf::table_view const& left_keys,
       cudf::table_view const& right_keys,
       cudf::table_view const& left_condition,
       cudf::table_view const& right_condition,
       cudf::ast::expression const& condition,
       cudf::null_equality nulls_equal) {
      return cudf::mixed_inner_join_size(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal);
    });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_mixedInnerJoinGatherMaps(JNIEnv* env,
                                                   jclass,
                                                   jlong j_left_keys,
                                                   jlong j_right_keys,
                                                   jlong j_left_condition,
                                                   jlong j_right_condition,
                                                   jlong j_condition,
                                                   jboolean j_nulls_equal)
{
  return cudf::jni::mixed_join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [](cudf::table_view const& left_keys,
       cudf::table_view const& right_keys,
       cudf::table_view const& left_condition,
       cudf::table_view const& right_condition,
       cudf::ast::expression const& condition,
       cudf::null_equality nulls_equal) {
      return cudf::mixed_inner_join(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal);
    });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_mixedInnerJoinGatherMapsWithSize(JNIEnv* env,
                                                           jclass,
                                                           jlong j_left_keys,
                                                           jlong j_right_keys,
                                                           jlong j_left_condition,
                                                           jlong j_right_condition,
                                                           jlong j_condition,
                                                           jboolean j_nulls_equal,
                                                           jlong j_output_row_count,
                                                           jlong j_matches_view)
{
  auto size_info = cudf::jni::get_mixed_size_info(env, j_output_row_count, j_matches_view);
  return cudf::jni::mixed_join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [&size_info](cudf::table_view const& left_keys,
                 cudf::table_view const& right_keys,
                 cudf::table_view const& left_condition,
                 cudf::table_view const& right_condition,
                 cudf::ast::expression const& condition,
                 cudf::null_equality nulls_equal) {
      return cudf::mixed_inner_join(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal, size_info);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_fullJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  return cudf::jni::join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    compare_nulls_equal,
    [](cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
      return cudf::full_join(left, right, nulleq);
    });
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_fullJoinRowCount(JNIEnv* env,
                                                                   jclass,
                                                                   jlong j_left_table,
                                                                   jlong j_right_hash_join)
{
  JNI_NULL_CHECK(env, j_left_table, "left table is null", 0);
  JNI_NULL_CHECK(env, j_right_hash_join, "right hash join is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto hash_join  = reinterpret_cast<cudf::hash_join const*>(j_right_hash_join);
    auto row_count  = hash_join->full_join_size(*left_table);
    return static_cast<jlong>(row_count);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_fullHashJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_hash_join)
{
  return cudf::jni::hash_join_gather_maps(
    env,
    j_left_table,
    j_right_hash_join,
    [](cudf::table_view const& left, cudf::hash_join const& hash) { return hash.full_join(left); });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_fullHashJoinGatherMapsWithCount(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_hash_join, jlong j_output_row_count)
{
  auto output_row_count = static_cast<std::size_t>(j_output_row_count);
  return cudf::jni::hash_join_gather_maps(
    env,
    j_left_table,
    j_right_hash_join,
    [output_row_count](cudf::table_view const& left, cudf::hash_join const& hash) {
      return hash.full_join(left, output_row_count);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_conditionalFullJoinGatherMaps(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  return cudf::jni::cond_join_gather_maps(env,
                                          j_left_table,
                                          j_right_table,
                                          j_condition,
                                          [](cudf::table_view const& left,
                                             cudf::table_view const& right,
                                             cudf::ast::expression const& cond_expr) {
                                            return cudf::conditional_full_join(
                                              left, right, cond_expr);
                                          });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_mixedFullJoinGatherMaps(JNIEnv* env,
                                                  jclass,
                                                  jlong j_left_keys,
                                                  jlong j_right_keys,
                                                  jlong j_left_condition,
                                                  jlong j_right_condition,
                                                  jlong j_condition,
                                                  jboolean j_nulls_equal)
{
  return cudf::jni::mixed_join_gather_maps(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [](cudf::table_view const& left_keys,
       cudf::table_view const& right_keys,
       cudf::table_view const& left_condition,
       cudf::table_view const& right_condition,
       cudf::ast::expression const& condition,
       cudf::null_equality nulls_equal) {
      return cudf::mixed_full_join(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftSemiJoinGatherMap(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  double constexpr load_factor = 0.5;
  return cudf::jni::join_gather_single_map(
    env,
    j_left_keys,
    j_right_keys,
    compare_nulls_equal,
    [load_factor](
      cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
      cudf::filtered_join obj(right, nulleq, load_factor, cudf::get_default_stream());
      return obj.semi_join(left);
    });
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_conditionalLeftSemiJoinRowCount(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", 0);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto row_count   = cudf::conditional_left_semi_join_size(
      *left_table, *right_table, condition->get_top_expression());
    return static_cast<jlong>(row_count);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_conditionalLeftSemiJoinGatherMap(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  return cudf::jni::cond_join_gather_single_map(env,
                                                j_left_table,
                                                j_right_table,
                                                j_condition,
                                                [](cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   cudf::ast::expression const& cond_expr) {
                                                  return cudf::conditional_left_semi_join(
                                                    left, right, cond_expr);
                                                });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_conditionalLeftSemiJoinGatherMapWithCount(JNIEnv* env,
                                                                    jclass,
                                                                    jlong j_left_table,
                                                                    jlong j_right_table,
                                                                    jlong j_condition,
                                                                    jlong j_row_count)
{
  auto row_count = static_cast<std::size_t>(j_row_count);
  return cudf::jni::cond_join_gather_single_map(
    env,
    j_left_table,
    j_right_table,
    j_condition,
    [row_count](cudf::table_view const& left,
                cudf::table_view const& right,
                cudf::ast::expression const& cond_expr) {
      return cudf::conditional_left_semi_join(left, right, cond_expr, row_count);
    });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_mixedLeftSemiJoinGatherMap(JNIEnv* env,
                                                     jclass,
                                                     jlong j_left_keys,
                                                     jlong j_right_keys,
                                                     jlong j_left_condition,
                                                     jlong j_right_condition,
                                                     jlong j_condition,
                                                     jboolean j_nulls_equal)
{
  return cudf::jni::mixed_join_gather_single_map(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [](cudf::table_view const& left_keys,
       cudf::table_view const& right_keys,
       cudf::table_view const& left_condition,
       cudf::table_view const& right_condition,
       cudf::ast::expression const& condition,
       cudf::null_equality nulls_equal) {
      return cudf::mixed_left_semi_join(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_leftAntiJoinGatherMap(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean compare_nulls_equal)
{
  double constexpr load_factor = 0.5;
  return cudf::jni::join_gather_single_map(
    env,
    j_left_keys,
    j_right_keys,
    compare_nulls_equal,
    [load_factor](
      cudf::table_view const& left, cudf::table_view const& right, cudf::null_equality nulleq) {
      cudf::filtered_join obj(right, nulleq, load_factor, cudf::get_default_stream());
      return obj.anti_join(left);
    });
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_conditionalLeftAntiJoinRowCount(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  JNI_NULL_CHECK(env, j_left_table, "left_table is null", 0);
  JNI_NULL_CHECK(env, j_right_table, "right_table is null", 0);
  JNI_NULL_CHECK(env, j_condition, "condition is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto condition   = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);
    auto row_count   = cudf::conditional_left_anti_join_size(
      *left_table, *right_table, condition->get_top_expression());
    return static_cast<jlong>(row_count);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_conditionalLeftAntiJoinGatherMap(
  JNIEnv* env, jclass, jlong j_left_table, jlong j_right_table, jlong j_condition)
{
  return cudf::jni::cond_join_gather_single_map(env,
                                                j_left_table,
                                                j_right_table,
                                                j_condition,
                                                [](cudf::table_view const& left,
                                                   cudf::table_view const& right,
                                                   cudf::ast::expression const& cond_expr) {
                                                  return cudf::conditional_left_anti_join(
                                                    left, right, cond_expr);
                                                });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_conditionalLeftAntiJoinGatherMapWithCount(JNIEnv* env,
                                                                    jclass,
                                                                    jlong j_left_table,
                                                                    jlong j_right_table,
                                                                    jlong j_condition,
                                                                    jlong j_row_count)
{
  auto row_count = static_cast<std::size_t>(j_row_count);
  return cudf::jni::cond_join_gather_single_map(
    env,
    j_left_table,
    j_right_table,
    j_condition,
    [row_count](cudf::table_view const& left,
                cudf::table_view const& right,
                cudf::ast::expression const& cond_expr) {
      return cudf::conditional_left_anti_join(left, right, cond_expr, row_count);
    });
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_mixedLeftAntiJoinGatherMap(JNIEnv* env,
                                                     jclass,
                                                     jlong j_left_keys,
                                                     jlong j_right_keys,
                                                     jlong j_left_condition,
                                                     jlong j_right_condition,
                                                     jlong j_condition,
                                                     jboolean j_nulls_equal)
{
  return cudf::jni::mixed_join_gather_single_map(
    env,
    j_left_keys,
    j_right_keys,
    j_left_condition,
    j_right_condition,
    j_condition,
    j_nulls_equal,
    [](cudf::table_view const& left_keys,
       cudf::table_view const& right_keys,
       cudf::table_view const& left_condition,
       cudf::table_view const& right_condition,
       cudf::ast::expression const& condition,
       cudf::null_equality nulls_equal) {
      return cudf::mixed_left_anti_join(
        left_keys, right_keys, left_condition, right_condition, condition, nulls_equal);
    });
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_crossJoin(JNIEnv* env,
                                                                 jclass,
                                                                 jlong left_table,
                                                                 jlong right_table)
{
  JNI_NULL_CHECK(env, left_table, "left_table is null", NULL);
  JNI_NULL_CHECK(env, right_table, "right_table is null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const left  = reinterpret_cast<cudf::table_view const*>(left_table);
    auto const right = reinterpret_cast<cudf::table_view const*>(right_table);
    return convert_table_for_return(env, cudf::cross_join(*left, *right));
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_interleaveColumns(JNIEnv* env,
                                                                    jclass,
                                                                    jlongArray j_cudf_table_view)
{
  JNI_NULL_CHECK(env, j_cudf_table_view, "table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view* table_view = reinterpret_cast<cudf::table_view*>(j_cudf_table_view);
    return release_as_jlong(cudf::interleave_columns(*table_view));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_concatenate(JNIEnv* env,
                                                                   jclass,
                                                                   jlongArray table_handles)
{
  JNI_NULL_CHECK(env, table_handles, "input tables are null", NULL);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::table_view> tables(env, table_handles);
    std::vector<cudf::table_view> const to_concat = tables.get_dereferenced();
    return convert_table_for_return(env, cudf::concatenate(to_concat));
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_partition(JNIEnv* env,
                                                                 jclass,
                                                                 jlong input_table,
                                                                 jlong partition_column,
                                                                 jint number_of_partitions,
                                                                 jintArray output_offsets)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, partition_column, "partition_column is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const n_input_table = reinterpret_cast<cudf::table_view const*>(input_table);
    auto const n_part_column = reinterpret_cast<cudf::column_view const*>(partition_column);

    auto [partitioned_table, partition_offsets] =
      cudf::partition(*n_input_table, *n_part_column, number_of_partitions);

    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);
    std::copy(partition_offsets.begin(), partition_offsets.end(), n_output_offsets.begin());

    return convert_table_for_return(env, partitioned_table);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_hashPartition(JNIEnv* env,
                                                                     jclass,
                                                                     jlong input_table,
                                                                     jintArray columns_to_hash,
                                                                     jint hash_function,
                                                                     jint number_of_partitions,
                                                                     jint seed,
                                                                     jintArray output_offsets)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, columns_to_hash, "columns_to_hash is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, number_of_partitions > 0, "number_of_partitions is zero", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const hash_func     = static_cast<cudf::hash_id>(hash_function);
    auto const hash_seed     = static_cast<uint32_t>(seed);
    auto const n_input_table = reinterpret_cast<cudf::table_view const*>(input_table);
    cudf::jni::native_jintArray n_columns_to_hash(env, columns_to_hash);
    JNI_ARG_CHECK(env, n_columns_to_hash.size() > 0, "columns_to_hash is zero", NULL);

    std::vector<cudf::size_type> columns_to_hash_vec(n_columns_to_hash.begin(),
                                                     n_columns_to_hash.end());

    auto [partitioned_table, partition_offsets] = cudf::hash_partition(
      *n_input_table, columns_to_hash_vec, number_of_partitions, hash_func, hash_seed);

    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);
    std::copy(partition_offsets.begin(), partition_offsets.end(), n_output_offsets.begin());

    return convert_table_for_return(env, partitioned_table);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_roundRobinPartition(JNIEnv* env,
                                                                           jclass,
                                                                           jlong input_table,
                                                                           jint num_partitions,
                                                                           jint start_partition,
                                                                           jintArray output_offsets)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, output_offsets, "output_offsets is null", NULL);
  JNI_ARG_CHECK(env, num_partitions > 0, "num_partitions <= 0", NULL);
  JNI_ARG_CHECK(env, start_partition >= 0, "start_partition is negative", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto n_input_table = reinterpret_cast<cudf::table_view*>(input_table);

    auto [partitioned_table, partition_offsets] =
      cudf::round_robin_partition(*n_input_table, num_partitions, start_partition);

    cudf::jni::native_jintArray n_output_offsets(env, output_offsets);
    std::copy(partition_offsets.begin(), partition_offsets.end(), n_output_offsets.begin());

    return convert_table_for_return(env, partitioned_table);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_groupByAggregate(JNIEnv* env,
                                           jclass,
                                           jlong input_table,
                                           jintArray keys,
                                           jintArray aggregate_column_indices,
                                           jlongArray agg_instances,
                                           jboolean ignore_null_keys,
                                           jboolean jkey_sorted,
                                           jbooleanArray jkeys_sort_desc,
                                           jbooleanArray jkeys_null_first)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, agg_instances, "agg_instances are null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view* n_input_table = reinterpret_cast<cudf::table_view*>(input_table);
    cudf::jni::native_jintArray n_keys(env, keys);
    cudf::jni::native_jintArray n_values(env, aggregate_column_indices);
    cudf::jni::native_jpointerArray<cudf::aggregation> n_agg_instances(env, agg_instances);
    std::vector<cudf::column_view> n_keys_cols;
    n_keys_cols.reserve(n_keys.size());
    for (int i = 0; i < n_keys.size(); i++) {
      n_keys_cols.push_back(n_input_table->column(n_keys[i]));
    }

    cudf::table_view n_keys_table(n_keys_cols);
    auto column_order    = cudf::jni::resolve_column_order(env, jkeys_sort_desc, n_keys.size());
    auto null_precedence = cudf::jni::resolve_null_precedence(env, jkeys_null_first, n_keys.size());
    cudf::groupby::groupby grouper(
      n_keys_table,
      ignore_null_keys ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE,
      jkey_sorted ? cudf::sorted::YES : cudf::sorted::NO,
      column_order,
      null_precedence);

    // Aggregates are passed in already grouped by column, so we just need to fill it in
    // as we go.
    std::vector<cudf::groupby::aggregation_request> requests;

    int previous_index = -1;
    for (int i = 0; i < n_values.size(); i++) {
      cudf::groupby::aggregation_request req;
      int col_index = n_values[i];

      cudf::groupby_aggregation* agg = dynamic_cast<cudf::groupby_aggregation*>(n_agg_instances[i]);
      JNI_ARG_CHECK(
        env, agg != nullptr, "aggregation is not an instance of groupby_aggregation", nullptr);
      std::unique_ptr<cudf::groupby_aggregation> cloned(
        dynamic_cast<cudf::groupby_aggregation*>(agg->clone().release()));

      if (col_index == previous_index) {
        requests.back().aggregations.push_back(std::move(cloned));
      } else {
        req.values = n_input_table->column(col_index);
        req.aggregations.push_back(std::move(cloned));
        requests.push_back(std::move(req));
      }
      previous_index = col_index;
    }

    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> result =
      grouper.aggregate(requests);

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    int agg_result_size = result.second.size();
    for (int agg_result_index = 0; agg_result_index < agg_result_size; agg_result_index++) {
      int col_agg_size = result.second[agg_result_index].results.size();
      for (int col_agg_index = 0; col_agg_index < col_agg_size; col_agg_index++) {
        result_columns.push_back(std::move(result.second[agg_result_index].results[col_agg_index]));
      }
    }
    return convert_table_for_return(env, result.first, std::move(result_columns));
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_groupByScan(JNIEnv* env,
                                      jclass,
                                      jlong input_table,
                                      jintArray keys,
                                      jintArray aggregate_column_indices,
                                      jlongArray agg_instances,
                                      jboolean ignore_null_keys,
                                      jboolean jkey_sorted,
                                      jbooleanArray jkeys_sort_desc,
                                      jbooleanArray jkeys_null_first)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, agg_instances, "agg_instances are null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view* n_input_table = reinterpret_cast<cudf::table_view*>(input_table);
    cudf::jni::native_jintArray n_keys(env, keys);
    cudf::jni::native_jintArray n_values(env, aggregate_column_indices);
    cudf::jni::native_jpointerArray<cudf::aggregation> n_agg_instances(env, agg_instances);
    std::vector<cudf::column_view> n_keys_cols;
    n_keys_cols.reserve(n_keys.size());
    for (int i = 0; i < n_keys.size(); i++) {
      n_keys_cols.push_back(n_input_table->column(n_keys[i]));
    }

    cudf::table_view n_keys_table(n_keys_cols);
    auto column_order    = cudf::jni::resolve_column_order(env, jkeys_sort_desc, n_keys.size());
    auto null_precedence = cudf::jni::resolve_null_precedence(env, jkeys_null_first, n_keys.size());
    cudf::groupby::groupby grouper(
      n_keys_table,
      ignore_null_keys ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE,
      jkey_sorted ? cudf::sorted::YES : cudf::sorted::NO,
      column_order,
      null_precedence);

    // Aggregates are passed in already grouped by column, so we just need to fill it in
    // as we go.
    std::vector<cudf::groupby::scan_request> requests;

    int previous_index = -1;
    for (int i = 0; i < n_values.size(); i++) {
      cudf::groupby::scan_request req;
      int col_index = n_values[i];

      cudf::groupby_scan_aggregation* agg =
        dynamic_cast<cudf::groupby_scan_aggregation*>(n_agg_instances[i]);
      JNI_ARG_CHECK(
        env, agg != nullptr, "aggregation is not an instance of groupby_scan_aggregation", nullptr);
      std::unique_ptr<cudf::groupby_scan_aggregation> cloned(
        dynamic_cast<cudf::groupby_scan_aggregation*>(agg->clone().release()));

      if (col_index == previous_index) {
        requests.back().aggregations.push_back(std::move(cloned));
      } else {
        req.values = n_input_table->column(col_index);
        req.aggregations.push_back(std::move(cloned));
        requests.push_back(std::move(req));
      }
      previous_index = col_index;
    }

    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> result =
      grouper.scan(requests);

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    int agg_result_size = result.second.size();
    for (int agg_result_index = 0; agg_result_index < agg_result_size; agg_result_index++) {
      int col_agg_size = result.second[agg_result_index].results.size();
      for (int col_agg_index = 0; col_agg_index < col_agg_size; col_agg_index++) {
        result_columns.push_back(std::move(result.second[agg_result_index].results[col_agg_index]));
      }
    }
    return convert_table_for_return(env, result.first, std::move(result_columns));
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_groupByReplaceNulls(JNIEnv* env,
                                              jclass,
                                              jlong input_table,
                                              jintArray keys,
                                              jintArray replace_column_indices,
                                              jbooleanArray is_preceding,
                                              jboolean ignore_null_keys,
                                              jboolean jkey_sorted,
                                              jbooleanArray jkeys_sort_desc,
                                              jbooleanArray jkeys_null_first)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, replace_column_indices, "input replace_column_indices are null", NULL);
  JNI_NULL_CHECK(env, is_preceding, "is_preceding are null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view* n_input_table = reinterpret_cast<cudf::table_view*>(input_table);
    cudf::jni::native_jintArray n_keys(env, keys);
    cudf::jni::native_jintArray n_values(env, replace_column_indices);
    cudf::jni::native_jbooleanArray n_is_preceding(env, is_preceding);
    std::vector<cudf::column_view> n_keys_cols;
    n_keys_cols.reserve(n_keys.size());
    for (int i = 0; i < n_keys.size(); i++) {
      n_keys_cols.push_back(n_input_table->column(n_keys[i]));
    }

    cudf::table_view n_keys_table(n_keys_cols);
    auto column_order    = cudf::jni::resolve_column_order(env, jkeys_sort_desc, n_keys.size());
    auto null_precedence = cudf::jni::resolve_null_precedence(env, jkeys_null_first, n_keys.size());
    cudf::groupby::groupby grouper(
      n_keys_table,
      ignore_null_keys ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE,
      jkey_sorted ? cudf::sorted::YES : cudf::sorted::NO,
      column_order,
      null_precedence);

    // Aggregates are passed in already grouped by column, so we just need to fill it in
    // as we go.
    std::vector<cudf::column_view> n_replace_cols;
    n_replace_cols.reserve(n_values.size());
    for (int i = 0; i < n_values.size(); i++) {
      n_replace_cols.push_back(n_input_table->column(n_values[i]));
    }
    cudf::table_view n_replace_table(n_replace_cols);

    std::vector<cudf::replace_policy> policies = n_is_preceding.transform_if_else(
      cudf::replace_policy::PRECEDING, cudf::replace_policy::FOLLOWING);

    auto [keys, results] = grouper.replace_nulls(n_replace_table, policies);
    return convert_table_for_return(env, keys, results);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_filter(JNIEnv* env,
                                                              jclass,
                                                              jlong input_jtable,
                                                              jlong mask_jcol)
{
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, mask_jcol, "mask column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::table_view const*>(input_jtable);
    auto const mask  = reinterpret_cast<cudf::column_view const*>(mask_jcol);
    return convert_table_for_return(env, cudf::apply_boolean_mask(*input, *mask));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jint JNICALL Java_ai_rapids_cudf_Table_distinctCount(JNIEnv* env,
                                                               jclass,
                                                               jlong input_jtable,
                                                               jboolean nulls_equal)
{
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::table_view const*>(input_jtable);

    return cudf::distinct_count(
      *input, nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_dropDuplicates(
  JNIEnv* env, jclass, jlong input_jtable, jintArray key_columns, jint keep, jboolean nulls_equal)
{
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, key_columns, "input key_columns is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::table_view const*>(input_jtable);

    static_assert(sizeof(jint) == sizeof(cudf::size_type), "Integer types mismatched.");
    auto const native_keys_indices = cudf::jni::native_jintArray(env, key_columns);
    auto const keys_indices =
      std::vector<cudf::size_type>(native_keys_indices.begin(), native_keys_indices.end());
    auto const keep_option = [&] {
      switch (keep) {
        case 0: return cudf::duplicate_keep_option::KEEP_ANY;
        case 1: return cudf::duplicate_keep_option::KEEP_FIRST;
        case 2: return cudf::duplicate_keep_option::KEEP_LAST;
        case 3: return cudf::duplicate_keep_option::KEEP_NONE;
        default:
          JNI_THROW_NEW(env,
                        cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                        "Invalid `keep` option",
                        cudf::duplicate_keep_option::KEEP_ANY);
      }
    }();

    auto result =
      cudf::distinct(*input,
                     keys_indices,
                     keep_option,
                     nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL,
                     cudf::nan_equality::ALL_EQUAL,
                     cudf::get_default_stream(),
                     cudf::get_current_device_resource_ref());
    return convert_table_for_return(env, result);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_gather(
  JNIEnv* env, jclass, jlong j_input, jlong j_map, jboolean check_bounds)
{
  JNI_NULL_CHECK(env, j_input, "input table is null", 0);
  JNI_NULL_CHECK(env, j_map, "map column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::table_view const*>(j_input);
    auto const map   = reinterpret_cast<cudf::column_view const*>(j_map);
    auto bounds_policy =
      check_bounds ? cudf::out_of_bounds_policy::NULLIFY : cudf::out_of_bounds_policy::DONT_CHECK;
    return convert_table_for_return(env, cudf::gather(*input, *map, bounds_policy));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_scatterTable(
  JNIEnv* env, jclass, jlong j_input, jlong j_map, jlong j_target)
{
  JNI_NULL_CHECK(env, j_input, "input table is null", 0);
  JNI_NULL_CHECK(env, j_map, "map column is null", 0);
  JNI_NULL_CHECK(env, j_target, "target table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input  = reinterpret_cast<cudf::table_view const*>(j_input);
    auto const map    = reinterpret_cast<cudf::column_view const*>(j_map);
    auto const target = reinterpret_cast<cudf::table_view const*>(j_target);
    return convert_table_for_return(env, cudf::scatter(*input, *map, *target));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_scatterScalars(
  JNIEnv* env, jclass, jlongArray j_input, jlong j_map, jlong j_target)
{
  JNI_NULL_CHECK(env, j_input, "input scalars array is null", 0);
  JNI_NULL_CHECK(env, j_map, "map column is null", 0);
  JNI_NULL_CHECK(env, j_target, "target table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const scalars_array = cudf::jni::native_jpointerArray<cudf::scalar>(env, j_input);
    std::vector<std::reference_wrapper<cudf::scalar const>> input;
    std::transform(
      scalars_array.begin(), scalars_array.end(), std::back_inserter(input), [](auto& scalar) {
        return std::ref(*scalar);
      });
    auto const map    = reinterpret_cast<cudf::column_view const*>(j_map);
    auto const target = reinterpret_cast<cudf::table_view const*>(j_target);
    return convert_table_for_return(env, cudf::scatter(input, *map, *target));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_repeatStaticCount(JNIEnv* env,
                                                                         jclass,
                                                                         jlong input_jtable,
                                                                         jint count)
{
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::table_view const*>(input_jtable);
    return convert_table_for_return(env, cudf::repeat(*input, count));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_repeatColumnCount(JNIEnv* env,
                                                                         jclass,
                                                                         jlong input_jtable,
                                                                         jlong count_jcol)
{
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, count_jcol, "count column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::table_view const*>(input_jtable);
    auto const count = reinterpret_cast<cudf::column_view const*>(count_jcol);
    return convert_table_for_return(env, cudf::repeat(*input, *count));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_bound(JNIEnv* env,
                                                        jclass,
                                                        jlong input_jtable,
                                                        jlong values_jtable,
                                                        jbooleanArray desc_flags,
                                                        jbooleanArray are_nulls_smallest,
                                                        jboolean is_upper_bound)
{
  JNI_NULL_CHECK(env, input_jtable, "input table is null", 0);
  JNI_NULL_CHECK(env, values_jtable, "values table is null", 0);
  using cudf::column;
  using cudf::table_view;
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    table_view* input  = reinterpret_cast<table_view*>(input_jtable);
    table_view* values = reinterpret_cast<table_view*>(values_jtable);
    cudf::jni::native_jbooleanArray const n_desc_flags(env, desc_flags);
    cudf::jni::native_jbooleanArray const n_are_nulls_smallest(env, are_nulls_smallest);

    std::vector<cudf::order> column_desc_flags{
      n_desc_flags.transform_if_else(cudf::order::DESCENDING, cudf::order::ASCENDING)};
    std::vector<cudf::null_order> column_null_orders{
      n_are_nulls_smallest.transform_if_else(cudf::null_order::BEFORE, cudf::null_order::AFTER)};

    JNI_ARG_CHECK(env,
                  (column_desc_flags.size() == column_null_orders.size()),
                  "null-order and sort-order size mismatch",
                  0);

    return release_as_jlong(
      is_upper_bound ? cudf::upper_bound(*input, *values, column_desc_flags, column_null_orders)
                     : cudf::lower_bound(*input, *values, column_desc_flags, column_null_orders));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jobjectArray JNICALL Java_ai_rapids_cudf_Table_contiguousSplit(JNIEnv* env,
                                                                         jclass,
                                                                         jlong input_table,
                                                                         jintArray split_indices)
{
  JNI_NULL_CHECK(env, input_table, "native handle is null", 0);
  JNI_NULL_CHECK(env, split_indices, "split indices are null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view* n_table = reinterpret_cast<cudf::table_view*>(input_table);
    cudf::jni::native_jintArray n_split_indices(env, split_indices);

    std::vector<cudf::size_type> indices(n_split_indices.data(),
                                         n_split_indices.data() + n_split_indices.size());

    std::vector<cudf::packed_table> result = cudf::contiguous_split(*n_table, indices);
    cudf::jni::native_jobjectArray<jobject> n_result =
      cudf::jni::contiguous_table_array(env, result.size());
    for (size_t i = 0; i < result.size(); i++) {
      n_result.set(
        i, cudf::jni::contiguous_table_from(env, result[i].data, result[i].table.num_rows()));
    }
    return n_result.wrapped();
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_makeChunkedPack(
  JNIEnv* env, jclass, jlong input_table, jlong bounce_buffer_size, jlong memoryResourceHandle)
{
  JNI_NULL_CHECK(env, input_table, "native handle is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::table_view* n_table = reinterpret_cast<cudf::table_view*>(input_table);
    // `temp_mr` is the memory resource that `cudf::chunked_pack` will use to create temporary
    // and scratch memory only.
    rmm::device_async_resource_ref temp_mr =
      memoryResourceHandle != 0
        ? rmm::device_async_resource_ref{cudf::jni::get_resource(memoryResourceHandle)}
        : cudf::get_current_device_resource_ref();
    auto chunked_pack =
      cudf::chunked_pack::create(*n_table, bounce_buffer_size, cudf::get_default_stream(), temp_mr);
    return reinterpret_cast<jlong>(chunked_pack.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_rollingWindowAggregate(JNIEnv* env,
                                                 jclass,
                                                 jlong j_input_table,
                                                 jintArray j_keys,
                                                 jlongArray j_default_output,
                                                 jintArray j_aggregate_column_indices,
                                                 jlongArray j_agg_instances,
                                                 jintArray j_min_periods,
                                                 jintArray j_preceding,
                                                 jintArray j_following,
                                                 jbooleanArray j_unbounded_preceding,
                                                 jbooleanArray j_unbounded_following,
                                                 jboolean ignore_null_keys)
{
  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, j_aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_agg_instances, "agg_instances are null", NULL);
  JNI_NULL_CHECK(env, j_default_output, "default_outputs are null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    using cudf::jni::valid_window_parameters;

    // Convert from j-types to native.
    cudf::table_view* input_table{reinterpret_cast<cudf::table_view*>(j_input_table)};
    cudf::jni::native_jintArray keys{env, j_keys};
    cudf::jni::native_jintArray values{env, j_aggregate_column_indices};
    cudf::jni::native_jpointerArray<cudf::aggregation> agg_instances(env, j_agg_instances);
    cudf::jni::native_jpointerArray<cudf::column_view> default_output(env, j_default_output);
    cudf::jni::native_jintArray min_periods{env, j_min_periods};
    cudf::jni::native_jintArray preceding{env, j_preceding};
    cudf::jni::native_jintArray following{env, j_following};
    cudf::jni::native_jbooleanArray unbounded_preceding{env, j_unbounded_preceding};
    cudf::jni::native_jbooleanArray unbounded_following{env, j_unbounded_following};

    if (not valid_window_parameters(values, agg_instances, min_periods, preceding, following)) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "Number of aggregation columns must match number of agg ops, and window-specs",
                    nullptr);
    }

    // Extract table-view.
    cudf::table_view groupby_keys{
      input_table->select(std::vector<cudf::size_type>(keys.data(), keys.data() + keys.size()))};

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    for (int i(0); i < values.size(); ++i) {
      cudf::rolling_aggregation* agg = dynamic_cast<cudf::rolling_aggregation*>(agg_instances[i]);
      JNI_ARG_CHECK(
        env, agg != nullptr, "aggregation is not an instance of rolling_aggregation", nullptr);

      int agg_column_index               = values[i];
      auto const preceding_window_bounds = unbounded_preceding[i]
                                             ? cudf::window_bounds::unbounded()
                                             : cudf::window_bounds::get(preceding[i]);
      auto const following_window_bounds = unbounded_following[i]
                                             ? cudf::window_bounds::unbounded()
                                             : cudf::window_bounds::get(following[i]);

      if (default_output[i] != nullptr) {
        result_columns.emplace_back(
          cudf::grouped_rolling_window(groupby_keys,
                                       input_table->column(agg_column_index),
                                       *default_output[i],
                                       preceding_window_bounds,
                                       following_window_bounds,
                                       min_periods[i],
                                       *agg));
      } else {
        result_columns.emplace_back(
          cudf::grouped_rolling_window(groupby_keys,
                                       input_table->column(agg_column_index),
                                       preceding_window_bounds,
                                       following_window_bounds,
                                       min_periods[i],
                                       *agg));
      }
    }

    auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
    return convert_table_for_return(env, result_table);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_ai_rapids_cudf_Table_rangeRollingWindowAggregate(JNIEnv* env,
                                                      jclass,
                                                      jlong j_input_table,
                                                      jintArray j_keys,
                                                      jintArray j_orderby_column_indices,
                                                      jbooleanArray j_is_orderby_ascending,
                                                      jbooleanArray j_is_orderby_nulls_first,
                                                      jintArray j_orderby_offsets,
                                                      jintArray j_aggregate_column_indices,
                                                      jlongArray j_agg_instances,
                                                      jintArray j_min_periods,
                                                      jlongArray j_preceding,
                                                      jlongArray j_following,
                                                      jintArray j_preceding_extent,
                                                      jintArray j_following_extent,
                                                      jboolean ignore_null_keys)
{
  JNI_NULL_CHECK(env, j_input_table, "input table is null", NULL);
  JNI_NULL_CHECK(env, j_keys, "input keys are null", NULL);
  JNI_NULL_CHECK(env, j_orderby_column_indices, "input orderby_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_is_orderby_ascending, "input orderby_ascending is null", NULL);
  JNI_NULL_CHECK(env, j_is_orderby_nulls_first, "input orderby_nulls_first is null", NULL);
  JNI_NULL_CHECK(env, j_orderby_offsets, "input orderby_offsets are null", NULL);
  JNI_NULL_CHECK(env, j_aggregate_column_indices, "input aggregate_column_indices are null", NULL);
  JNI_NULL_CHECK(env, j_agg_instances, "agg_instances are null", NULL);
  JNI_NULL_CHECK(env, j_preceding, "preceding are null", NULL);
  JNI_NULL_CHECK(env, j_following, "following are null", NULL);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    using cudf::jni::valid_window_parameters;

    // Convert from j-types to native.
    cudf::table_view* input_table{reinterpret_cast<cudf::table_view*>(j_input_table)};
    cudf::jni::native_jintArray keys{env, j_keys};
    cudf::jni::native_jintArray orderbys{env, j_orderby_column_indices};
    cudf::jni::native_jbooleanArray orderbys_ascending{env, j_is_orderby_ascending};
    cudf::jni::native_jbooleanArray orderbys_nulls_first{env, j_is_orderby_nulls_first};
    cudf::jni::native_jintArray orderby_offsets{env, j_orderby_offsets};
    cudf::jni::native_jintArray values{env, j_aggregate_column_indices};
    cudf::jni::native_jpointerArray<cudf::aggregation> agg_instances(env, j_agg_instances);
    cudf::jni::native_jintArray min_periods{env, j_min_periods};
    cudf::jni::native_jintArray preceding_extent{env, j_preceding_extent};
    cudf::jni::native_jintArray following_extent{env, j_following_extent};
    cudf::jni::native_jpointerArray<cudf::scalar> preceding(env, j_preceding);
    cudf::jni::native_jpointerArray<cudf::scalar> following(env, j_following);

    if (not valid_window_parameters(values, agg_instances, min_periods, preceding, following)) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "Number of aggregation columns must match number of agg ops, and window-specs",
                    nullptr);
    }
    JNI_ARG_CHECK(env,
                  orderby_offsets.size() == values.size() + 1,
                  "orderby_offsets length must be one more than the number of aggregation columns",
                  nullptr);
    JNI_ARG_CHECK(env,
                  orderbys.size() == orderbys_ascending.size() &&
                    orderbys.size() == orderbys_nulls_first.size(),
                  "orderby column-index, ascending, and nulls-first arrays must have equal length",
                  nullptr);
    JNI_ARG_CHECK(env,
                  orderby_offsets[values.size()] == orderbys.size(),
                  "last orderby_offsets entry must equal the total number of order-by columns",
                  nullptr);
    // The CSR offsets must be non-decreasing and start at zero. Combined with the last-entry check
    // above, this guarantees every per-op slice satisfies 0 <= ob_begin <= ob_end <=
    // orderbys.size().
    JNI_ARG_CHECK(
      env, orderby_offsets[0] == 0, "first orderby_offsets entry must be zero", nullptr);
    for (int i = 0; i < values.size(); ++i) {
      JNI_ARG_CHECK(env,
                    orderby_offsets[i] <= orderby_offsets[i + 1],
                    "orderby_offsets must be non-decreasing",
                    nullptr);
    }

    // Extract table-view.
    cudf::table_view groupby_keys{
      input_table->select(std::vector<cudf::size_type>(keys.data(), keys.data() + keys.size()))};

    // Range extents are defined as:
    // a) 0 == CURRENT ROW
    // b) 1 == BOUNDED
    // c) 2 == UNBOUNDED
    auto constexpr CURRENT_ROW = 0;
    auto constexpr BOUNDED     = 1;
    auto constexpr UNBOUNDED   = 2;

    // Clone a (borrowed) rolling_aggregation into an owned unique_ptr, as required by
    // cudf::rolling_request. The Java side retains ownership of the original aggregation instances.
    // Each agg is verified to be a rolling_aggregation before this is called, so clone() preserves
    // its dynamic type and the cast succeeds, mirroring the groupby_aggregation clone idiom above.
    auto const clone_rolling = [](cudf::rolling_aggregation const& agg) {
      return std::unique_ptr<cudf::rolling_aggregation>(
        dynamic_cast<cudf::rolling_aggregation*>(agg.clone().release()));
    };

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    for (int i(0); i < values.size(); ++i) {
      int agg_column_index = values[i];

      cudf::rolling_aggregation* agg = dynamic_cast<cudf::rolling_aggregation*>(agg_instances[i]);
      JNI_ARG_CHECK(
        env, agg != nullptr, "aggregation is not an instance of rolling_aggregation", nullptr);

      // orderby_offsets is CSR-style with length == values.size() + 1; ob_begin/ob_end
      // delimit this op's order-by slice, and ob_count is the slice length.
      int const ob_begin = orderby_offsets[i];
      int const ob_end   = orderby_offsets[i + 1];
      int const ob_count = ob_end - ob_begin;

      if (ob_count == 1) {
        // Single order-by column: use the legacy range_window_bounds overload, which deduces null
        // ordering from the data and supports bounded scalar ranges.
        cudf::column_view const& order_by_column = input_table->column(orderbys[ob_begin]);
        cudf::data_type order_by_type            = order_by_column.type();
        cudf::data_type duration_type            = order_by_type;

        // Must set the duration type for only the non-BOUNDED cases.
        if (preceding_extent[i] != BOUNDED || following_extent[i] != BOUNDED) {
          switch (order_by_type.id()) {
            case cudf::type_id::TIMESTAMP_DAYS:
              duration_type = cudf::data_type{cudf::type_id::DURATION_DAYS};
              break;
            case cudf::type_id::TIMESTAMP_SECONDS:
              duration_type = cudf::data_type{cudf::type_id::DURATION_SECONDS};
              break;
            case cudf::type_id::TIMESTAMP_MILLISECONDS:
              duration_type = cudf::data_type{cudf::type_id::DURATION_MILLISECONDS};
              break;
            case cudf::type_id::TIMESTAMP_MICROSECONDS:
              duration_type = cudf::data_type{cudf::type_id::DURATION_MICROSECONDS};
              break;
            case cudf::type_id::TIMESTAMP_NANOSECONDS:
              duration_type = cudf::data_type{cudf::type_id::DURATION_NANOSECONDS};
              break;
            default: break;
          }
        }

        auto const make_window_bounds = [&](auto const& range_extent, auto const* p_scalar) {
          if (range_extent == CURRENT_ROW) {
            return cudf::range_window_bounds::current_row(duration_type);
          } else if (range_extent == UNBOUNDED) {
            return cudf::range_window_bounds::unbounded(duration_type);
          } else {
            return cudf::range_window_bounds::get(*p_scalar);
          }
        };

        result_columns.emplace_back(cudf::grouped_range_rolling_window(
          groupby_keys,
          order_by_column,
          orderbys_ascending[ob_begin] ? cudf::order::ASCENDING : cudf::order::DESCENDING,
          input_table->column(agg_column_index),
          make_window_bounds(preceding_extent[i], preceding[i]),
          make_window_bounds(following_extent[i], following[i]),
          min_periods[i],
          *agg));
      } else {
        // Multi-column order-by: use the table overload. Only peer-frame bounds (UNBOUNDED and
        // CURRENT_ROW) are supported; bounded scalar ranges are rejected on the Java side, but
        // guard here as defense in depth.
        JNI_ARG_CHECK(env,
                      preceding_extent[i] != BOUNDED && following_extent[i] != BOUNDED,
                      "Multi-column RANGE windows do not support bounded scalar ranges",
                      nullptr);

        std::vector<cudf::size_type> ob_indices(orderbys.data() + ob_begin,
                                                orderbys.data() + ob_end);
        cudf::table_view order_by_table{input_table->select(ob_indices)};

        std::vector<cudf::order> orders;
        std::vector<cudf::null_order> null_orders;
        orders.reserve(ob_count);
        null_orders.reserve(ob_count);
        for (int j = ob_begin; j < ob_end; ++j) {
          orders.push_back(orderbys_ascending[j] ? cudf::order::ASCENDING
                                                 : cudf::order::DESCENDING);
          null_orders.push_back(orderbys_nulls_first[j] ? cudf::null_order::BEFORE
                                                        : cudf::null_order::AFTER);
        }

        auto const to_range_window_type = [&](int range_extent) -> cudf::range_window_type {
          if (range_extent == CURRENT_ROW) {
            return cudf::current_row{};
          } else {
            return cudf::unbounded{};
          }
        };

        std::vector<cudf::rolling_request> requests;
        requests.push_back(cudf::rolling_request{
          input_table->column(agg_column_index), min_periods[i], clone_rolling(*agg)});

        auto const result = cudf::grouped_range_rolling_window(
          groupby_keys,
          order_by_table,
          cudf::host_span<cudf::order const>{orders},
          cudf::host_span<cudf::null_order const>{null_orders},
          to_range_window_type(preceding_extent[i]),
          to_range_window_type(following_extent[i]),
          cudf::host_span<cudf::rolling_request const>{requests});
        auto result_cols = result->release();
        result_columns.emplace_back(std::move(result_cols[0]));
      }
    }

    auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
    return convert_table_for_return(env, result_table);
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_explode(JNIEnv* env,
                                                               jclass,
                                                               jlong input_jtable,
                                                               jint column_index)
{
  JNI_NULL_CHECK(env, input_jtable, "explode: input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_table = reinterpret_cast<cudf::table_view const*>(input_jtable);
    auto const col_index   = static_cast<cudf::size_type>(column_index);
    return convert_table_for_return(env, cudf::explode(*input_table, col_index));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_explodePosition(JNIEnv* env,
                                                                       jclass,
                                                                       jlong input_jtable,
                                                                       jint column_index)
{
  JNI_NULL_CHECK(env, input_jtable, "explode: input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_table = reinterpret_cast<cudf::table_view const*>(input_jtable);
    auto const col_index   = static_cast<cudf::size_type>(column_index);
    return convert_table_for_return(env, cudf::explode_position(*input_table, col_index));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_explodeOuter(JNIEnv* env,
                                                                    jclass,
                                                                    jlong input_jtable,
                                                                    jint column_index)
{
  JNI_NULL_CHECK(env, input_jtable, "explode: input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_table = reinterpret_cast<cudf::table_view const*>(input_jtable);
    auto const col_index   = static_cast<cudf::size_type>(column_index);
    return convert_table_for_return(env, cudf::explode_outer(*input_table, col_index));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_explodeOuterPosition(JNIEnv* env,
                                                                            jclass,
                                                                            jlong input_jtable,
                                                                            jint column_index)
{
  JNI_NULL_CHECK(env, input_jtable, "explode: input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_table = reinterpret_cast<cudf::table_view const*>(input_jtable);
    auto const col_index   = static_cast<cudf::size_type>(column_index);
    return convert_table_for_return(env, cudf::explode_outer_position(*input_table, col_index));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_Table_rowBitCount(JNIEnv* env, jclass, jlong j_table)
{
  JNI_NULL_CHECK(env, j_table, "table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_table = reinterpret_cast<cudf::table_view const*>(j_table);
    return release_as_jlong(cudf::row_bit_count(*input_table));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jobject JNICALL
Java_ai_rapids_cudf_Table_contiguousSplitGroups(JNIEnv* env,
                                                jclass,
                                                jlong jinput_table,
                                                jintArray jkey_indices,
                                                jintArray jprojection_column_indices,
                                                jboolean jignore_null_keys,
                                                jboolean jkey_sorted,
                                                jbooleanArray jkeys_sort_desc,
                                                jbooleanArray jkeys_null_first,
                                                jboolean genUniqKeys)
{
  JNI_NULL_CHECK(env, jinput_table, "table native handle is null", 0);
  JNI_NULL_CHECK(env, jkey_indices, "key indices are null", 0);
  // Two main steps to split the groups in the input table.
  //    1) Calls `cudf::groupby::groupby::get_groups` to get the group offsets and
  //       the grouped table.
  //    2) Calls `cudf::contiguous_split` to execute the split over the grouped table
  //       according to the group offsets.
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jintArray n_key_indices(env, jkey_indices);
    auto const input_table = reinterpret_cast<cudf::table_view const*>(jinput_table);

    // Prepares arguments for the groupby:
    //   (keys, null_handling, keys_are_sorted, column_order, null_precedence)
    std::vector<cudf::size_type> key_indices = n_key_indices.to_vector();

    // check number of key columns vs number of input columns
    size_t num_cols_input = static_cast<size_t>(input_table->num_columns());
    if (key_indices.size() > num_cols_input) {
      JNI_THROW_NEW(
        env,
        cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
        "The number of key columns is greater than the number of columns in the input table",
        0);
    }

    auto keys = key_indices.empty() ? cudf::table_view{} : input_table->select(key_indices);
    auto null_handling =
      jignore_null_keys ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
    auto keys_are_sorted = jkey_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto column_order = cudf::jni::resolve_column_order(env, jkeys_sort_desc, key_indices.size());
    auto null_precedence =
      cudf::jni::resolve_null_precedence(env, jkeys_null_first, key_indices.size());

    // Constructs a groupby
    cudf::groupby::groupby grouper(
      keys, null_handling, keys_are_sorted, column_order, null_precedence);

    // 1) Gets the groups(keys, offsets, values) from groupby.
    //
    // If the `jprojection_column_indices` is null, uses all_columns as projection columns;
    // If the `jprojection_column_indices` is not null, use the `jprojection_column_indices` columns
    // as projection columns.
    std::vector<cudf::size_type> projection_indices;
    auto num_project_cols = [&]() -> size_t {
      if (jprojection_column_indices == NULL) {
        // if a column is not in key columns, then it's a projection column
        auto num_non_key_cols = num_cols_input - key_indices.size();
        projection_indices.reserve(num_non_key_cols);
        cudf::size_type index = 0;
        while (projection_indices.size() < num_non_key_cols) {
          if (std::find(key_indices.begin(), key_indices.end(), index) == key_indices.end()) {
            // not key column, so adds it as projection column.
            projection_indices.emplace_back(index);
          }
          index++;
        }
        return num_non_key_cols;
      } else {
        // use the specified projection columns as output columns
        cudf::jni::native_jintArray n_project_indices(env, jprojection_column_indices);
        projection_indices = n_project_indices.to_vector();
        return static_cast<size_t>(n_project_indices.size());
      }
    }();

    cudf::table_view projection_view = input_table->select(projection_indices);
    // execute grouping
    cudf::groupby::groupby::groups groups = grouper.get_groups(projection_view);

    // if jprojection_column_indices is null, output both key columns and projection columns;
    // otherwise, only output projection columns.
    auto num_grouped_cols =
      num_project_cols + ((jprojection_column_indices == NULL) ? key_indices.size() : 0);

    std::vector<cudf::column_view> grouped_cols(num_grouped_cols);

    if (jprojection_column_indices == NULL) {
      // When builds the table view from keys and non-keys of 'groups', restores the
      // original order of columns (same order with that in input table).
      // key columns
      auto key_view    = groups.keys->view();
      auto key_view_it = key_view.begin();
      for (auto key_id : key_indices) {
        grouped_cols[key_id] = *key_view_it;
        key_view_it++;
      }
      // non-key columns
      auto projection_view = groups.values->view();
      auto project_view_it = projection_view.begin();
      for (auto value_id : projection_indices) {
        grouped_cols[value_id] = *project_view_it;
        project_view_it++;
      }
    } else {
      // specified projection_indices, do not output keys columns by default
      auto projection_view = groups.values->view();
      auto project_view_it = projection_view.begin();
      for (size_t i = 0; i < num_project_cols; ++i) {
        grouped_cols[i] = *project_view_it;
        project_view_it++;
      }
    }

    cudf::table_view grouped_table(grouped_cols);
    // When no key columns, uses the input table instead, because the output
    // of 'get_groups' is empty.
    auto& grouped_view = key_indices.empty() ? *input_table : grouped_table;

    // Resolves the split indices from offsets vector directly to avoid copying. Since
    // the offsets vector may be very large if there are too many small groups.
    std::vector<cudf::size_type>& split_indices = groups.offsets;
    // Offsets layout is [0, split indices..., num_rows] or [0] for empty keys, so
    // need to removes the first and last elements. First remove last one.
    split_indices.pop_back();

    // generate uniq keys by using `gather` method, this means remove the duplicated keys
    std::unique_ptr<cudf::table> group_by_result_table;
    if (genUniqKeys) {
      // generate gather map column from `split_indices`
      auto begin      = std::cbegin(split_indices);
      auto end        = std::cend(split_indices);
      auto const size = cudf::distance(begin, end);
      auto const vec  = thrust::host_vector<cudf::size_type>(begin, end);
      auto buf =
        rmm::device_buffer{vec.data(), size * sizeof(cudf::size_type), cudf::get_default_stream()};
      auto gather_map_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, size, std::move(buf), rmm::device_buffer{}, 0);

      // gather the first key in each group to remove duplicated ones.
      group_by_result_table = cudf::gather(groups.keys->view(), gather_map_col->view());
    }

    // remove the first 0 if it exists
    if (!split_indices.empty()) { split_indices.erase(split_indices.begin()); }

    // 2) Splits the groups.
    std::vector<cudf::packed_table> result = cudf::contiguous_split(grouped_view, split_indices);
    // Release the grouped table right away after split done.
    groups.keys.reset(nullptr);
    groups.values.reset(nullptr);

    //  Returns the split result.
    cudf::jni::native_jobjectArray<jobject> n_result =
      cudf::jni::contiguous_table_array(env, result.size());
    for (size_t i = 0; i < result.size(); i++) {
      n_result.set(
        i, cudf::jni::contiguous_table_from(env, result[i].data, result[i].table.num_rows()));
    }

    jobjectArray groups_array = n_result.wrapped();

    if (genUniqKeys) {
      jlongArray keys_array = convert_table_for_return(env, group_by_result_table);
      return cudf::jni::contig_split_group_by_result_from(env, groups_array, keys_array);
    } else {
      return cudf::jni::contig_split_group_by_result_from(env, groups_array);
    }
  }
  JNI_CATCH(env, NULL);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_Table_sample(
  JNIEnv* env, jclass, jlong j_input, jlong n, jboolean replacement, jlong seed)
{
  JNI_NULL_CHECK(env, j_input, "input table is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::table_view const*>(j_input);
    auto sample_with_replacement =
      replacement ? cudf::sample_with_replacement::TRUE : cudf::sample_with_replacement::FALSE;
    return convert_table_for_return(env, cudf::sample(*input, n, sample_with_replacement, seed));
  }
  JNI_CATCH(env, 0);
}
}  // extern "C"
