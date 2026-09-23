/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "writer.hpp"

#include "host_staging.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.hpp>

#define USE_OWN_ARROW
using FFI_ArrowSchema      = ArrowSchema;
using FFI_ArrowArray       = ArrowArray;
using FFI_ArrowArrayStream = ArrowArrayStream;
#include <vortex_cuda.h>

#include <algorithm>
#include <memory>
#include <unordered_set>

namespace ndsh {

using cudf::size_type;
using cudf::table_view;
using cudf::type_id;
using cudf::io::io_type;
using cudf::io::sink_info;

vortex_writer_options_builder vortex_writer_options::builder(sink_info const& sink,
                                                             table_view const& table)
{
  return vortex_writer_options_builder{sink, table};
}

namespace {

using session_ptr = std::unique_ptr<vx_session, decltype(&vx_session_free)>;
using array_ptr   = std::unique_ptr<vx_array const, decltype(&vx_array_free)>;
using dtype_ptr   = std::unique_ptr<vx_dtype const, decltype(&vx_dtype_free)>;
using sink_ptr    = std::unique_ptr<vx_array_sink, decltype(&vx_array_sink_abort)>;

void check_error(vx_error*& error, std::string const& operation)
{
  auto owned = std::unique_ptr<vx_error, decltype(&vx_error_free)>{std::exchange(error, nullptr),
                                                                   vx_error_free};
  if (owned) {
    auto const message = vx_error_message(owned.get());
    CUDF_FAIL((operation + ": " + std::string{message.ptr, message.len}), std::runtime_error);
  }
}

bool supported_type(type_id id)
{
  switch (id) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::UINT8:
    case type_id::UINT16:
    case type_id::UINT32:
    case type_id::UINT64:
    case type_id::FLOAT32:
    case type_id::FLOAT64:
    case type_id::BOOL8:
    case type_id::STRING:
    case type_id::TIMESTAMP_DAYS:
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128: return true;
    default: return false;
  }
}

void write(vortex_writer_options const& options,
           cuda::stream_ref stream,
           rmm::device_async_resource_ref mr)
{
  auto const& sink_info = options.get_sink();
  CUDF_EXPECTS(sink_info.type() == io_type::FILEPATH && sink_info.num_sinks() == 1,
               "write_vortex requires exactly one local-file sink");
  auto const& path = sink_info.filepaths().front();
  CUDF_EXPECTS(
    !path.empty() && path.find('\0') == std::string::npos && path.find("://") == std::string::npos,
    "write_vortex requires a nonempty local path without NUL bytes or a URI scheme");
  auto const table      = options.get_table();
  auto const chunk_rows = options.get_rows_per_chunk();
  CUDF_EXPECTS(chunk_rows > 0, "write_vortex rows_per_chunk must be positive");
  CUDF_EXPECTS(table.num_columns() > 0, "write_vortex requires at least one column");
  for (auto const& column : table) {
    CUDF_EXPECTS(supported_type(column.type().id()), "write_vortex unsupported column type");
  }
  auto names = options.get_names();
  if (names.empty()) {
    for (size_type i = 0; i < table.num_columns(); ++i) {
      names.push_back("_col" + std::to_string(i));
    }
  }
  CUDF_EXPECTS(names.size() == static_cast<std::size_t>(table.num_columns()),
               "write_vortex requires one name per column");
  std::unordered_set<std::string> unique_names;
  std::vector<cudf::column_metadata> metadata;
  metadata.reserve(names.size());
  for (auto const& name : names) {
    CUDF_EXPECTS(name.find('\0') == std::string::npos,
                 "write_vortex column names must not contain NUL bytes");
    CUDF_EXPECTS(unique_names.insert(name).second, "write_vortex column names must be unique");
    metadata.emplace_back(name);
  }

  int device = -1;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  CUDF_EXPECTS(device == 0, "write_vortex currently requires CUDA device 0");
  unsigned int flags = 0;
  CUDF_CUDA_TRY(cudaStreamGetFlags(stream.get(), &flags));
  vx_error* error = nullptr;
  session_ptr session{vx_cuda_session_new(&error), vx_session_free};
  check_error(error, "create Vortex writer session");
  CUDF_EXPECTS(
    session != nullptr, "Vortex writer session creation returned null", std::runtime_error);

  vx_view const file_path{path.data(), path.size()};
  sink_ptr sink{nullptr, vx_array_sink_abort};
  auto const operation = "write_vortex(" + path + ")";
  size_type offset     = 0;
  do {
    auto const end    = offset + std::min(chunk_rows, table.num_rows() - offset);
    auto const chunks = cudf::slice(table, {offset, end}, stream);
    // Arrow import consumes both array and schema, so build a fresh schema for each chunk.
    // Derive it from the original table to keep nullability stable across slices.
    auto schema = cudf::to_arrow_schema(table, metadata);
    auto host   = ndsh::detail::stage_host_chunk(chunks.front(), stream, mr);
    CUDF_EXPECTS(host->device_type == ARROW_DEVICE_CPU,
                 operation + ": cuDF host export returned non-host data",
                 std::runtime_error);
    auto array = array_ptr{
      vx_array_from_arrow(session.get(), &host->array, schema.get(), false, &error), vx_array_free};
    check_error(error, operation);
    CUDF_EXPECTS(array != nullptr, operation + ": Arrow import returned null", std::runtime_error);
    if (!sink) {
      auto dtype = dtype_ptr{vx_array_dtype(array.get()), vx_dtype_free};
      sink.reset(vx_cuda_array_sink_open_file_block_rows(
        session.get(), file_path, dtype.get(), static_cast<std::size_t>(chunk_rows), &error));
      check_error(error, operation);
      CUDF_EXPECTS(sink != nullptr, operation + ": writer returned null", std::runtime_error);
    }
    vx_array_sink_push(sink.get(), array.get(), &error);
    check_error(error, operation);
    offset = end;
  } while (offset < table.num_rows());
  // close consumes the sink even on failure.
  vx_array_sink_close(sink.release(), &error);
  check_error(error, operation);
}

}  // namespace

void write_vortex(vortex_writer_options const& options,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  write(options, stream, mr);
}

}  // namespace ndsh
