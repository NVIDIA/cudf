/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vortex_io.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/interop.hpp>
#include <cudf/io/vortex.hpp>

#include <cuda/stream>
#include <nvtx3/nvtx3.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.hpp>

// Use cuDF's nanoarrow definitions instead of redeclaring the Arrow C ABI structs.
#define USE_OWN_ARROW
using FFI_ArrowSchema      = ArrowSchema;
using FFI_ArrowArray       = ArrowArray;
using FFI_ArrowArrayStream = ArrowArrayStream;
#include <vortex_cuda.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ndsh {
namespace {

using session_ptr = std::unique_ptr<vx_session, decltype(&vx_session_free)>;

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"Vortex I/O CUDA error: "} + cudaGetErrorString(status));
  }
}

void check_device()
{
  int device = -1;
  check_cuda(cudaGetDevice(&device));
  if (device != 0) {
    throw std::invalid_argument("Vortex benchmark I/O currently requires CUDA device 0");
  }
}

void retain_vortex_cuda_pool_memory()
{
  std::uint64_t release_threshold = 8ULL << 30;
  cudaMemPool_t pool{};
  check_cuda(cudaDeviceGetMemPool(&pool, 0));
  check_cuda(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));
}

vx_view path_view(std::string const& path)
{
  if (path.empty() || path.find('\0') != std::string::npos) {
    throw std::invalid_argument("Vortex I/O requires a nonempty local path without NUL bytes");
  }
  return {path.data(), path.size()};
}

void check_error(vx_error*& error, std::string const& operation)
{
  auto owned = std::unique_ptr<vx_error, decltype(&vx_error_free)>{std::exchange(error, nullptr),
                                                                   vx_error_free};
  if (owned) {
    auto const message = vx_error_message(owned.get());
    throw std::runtime_error(operation + ": " + std::string{message.ptr, message.len});
  }
}

// Declare after buffer owners to drain before unwinding releases them. Failed
// synchronization is fatal: producer memory may still be in use.
class stream_drain {
 public:
  explicit stream_drain(cudaStream_t stream) : stream_{stream} {}
  ~stream_drain()
  {
    if (pending_ && cudaStreamSynchronize(stream_) != cudaSuccess) { std::terminate(); }
  }
  void wait()
  {
    check_cuda(cudaStreamSynchronize(stream_));
    pending_ = false;
  }

 private:
  cudaStream_t stream_;
  bool pending_{true};
};

struct device_stream {
  nanoarrow::device::UniqueDeviceArrayStream value;
  void check(int status, std::string const& operation)
  {
    if (status != 0) {
      auto const* error = value->get_last_error ? value->get_last_error(value.get()) : nullptr;
      throw std::runtime_error(operation + ": " + (error ? error : "Arrow device stream failed"));
    }
  }
};

struct device_batch {
  nanoarrow::device::UniqueDeviceArray value;
  // Destroy the view and its deleter-owned import scratch before releasing Arrow buffers.
  std::optional<cudf::unique_table_view_t> view;

  device_batch()                        = default;
  device_batch(device_batch&&) noexcept = default;
  // Vector growth only needs move construction; memberwise assignment would release value first.
  device_batch& operator=(device_batch&&) = delete;
};

}  // namespace

// Implementation details with external linkage for focused tests, not public adapter API.
namespace detail {

void check_flat_schema(ArrowSchema const& schema)
{
  if (!schema.format || std::string_view{schema.format} != "+s" || schema.n_children < 0 ||
      schema.n_children > std::numeric_limits<cudf::size_type>::max()) {
    throw std::runtime_error("read_vortex requires a table-shaped Arrow schema");
  }
  if (schema.n_children == 0) {
    throw std::runtime_error("read_vortex requires at least one column");
  }
  for (int64_t i = 0; i < schema.n_children; ++i) {
    auto const& field  = *schema.children[i];
    auto const& values = field.dictionary ? *field.dictionary : field;
    if (values.n_children != 0 || values.dictionary) {
      throw std::runtime_error("read_vortex currently supports flat columns only");
    }
  }
}

}  // namespace detail

namespace {

std::unique_ptr<cudf::table> empty_table(ArrowSchema const& schema,
                                         cudaStream_t stream,
                                         rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray empty;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(empty.get(), &schema, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(empty.get()));
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(empty.get(), nullptr));
  std::unique_ptr<cudf::table> result;
  stream_drain drain{stream};
  result = cudf::from_arrow(&schema, empty.get(), cuda::stream_ref{stream}, mr);
  drain.wait();
  return result;
}

}  // namespace

struct vortex_io::impl {
  session_ptr session{nullptr, vx_session_free};
  cudaStream_t stream;
  rmm::device_async_resource_ref mr;

  impl(cudaStream_t stream, rmm::device_async_resource_ref mr) : stream{stream}, mr{mr}
  {
    check_device();
    retain_vortex_cuda_pool_memory();
    // Validate the supplied consumer stream before creating Vortex's independent stream pool.
    unsigned int flags = 0;
    check_cuda(cudaStreamGetFlags(stream, &flags));
    vx_error* error = nullptr;
    session.reset(vx_cuda_session_new(&error));
    check_error(error, "create Vortex CUDA session");
    if (!session) { throw std::runtime_error("Vortex CUDA session creation returned null"); }
  }
};

vortex_io::vortex_io(cudaStream_t stream, rmm::device_async_resource_ref mr)
  : impl_{std::make_unique<impl>(stream, mr)}
{
}

vortex_io::~vortex_io() = default;

void vortex_io::write_vortex(std::string const& path,
                             cudf::table_view table,
                             std::vector<std::string> const& column_names,
                             cudf::size_type chunk_rows) const
{
  auto const options = cudf::io::vortex_writer_options::builder(cudf::io::sink_info{path}, table)
                         .names(column_names)
                         .rows_per_chunk(chunk_rows)
                         .build();
  cudf::io::write_vortex(options, cuda::stream_ref{impl_->stream}, impl_->mr);
}

cudf::io::table_with_metadata vortex_io::read_vortex(std::string const& path,
                                                     std::size_t batch_rows,
                                                     std::vector<std::string> const& columns,
                                                     bool direct_io) const
{
  nvtx3::scoped_range read_range{"vortex.read"};
  check_device();
  auto const file_path = path_view(path);
  if (batch_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("read_vortex batch_rows exceeds the cuDF row limit");
  }
  auto const operation = "read_vortex(" + path + ")";
  device_stream input;
  vx_error* error = nullptr;
  vx_cuda_scan_options options{};
  options.flags      = direct_io ? VX_CUDA_SCAN_FLAG_DIRECT_IO : 0;
  options.batch_rows = batch_rows;
  std::vector<vx_view> column_views;
  column_views.reserve(columns.size());
  for (auto const& name : columns) {
    column_views.push_back({name.data(), name.size()});
  }
  // FFI copies names during the call; neither the views nor their bytes escape.
  int status;
  {
    nvtx3::scoped_range range{"vortex.scan_open"};
    status = vx_cuda_scan_path_arrow_device_stream_projected(impl_->session.get(),
                                                             file_path,
                                                             &options,
                                                             column_views.data(),
                                                             column_views.size(),
                                                             input.value.get(),
                                                             &error);
  }
  check_error(error, operation);
  if (status != 0) { throw std::runtime_error(operation + ": Vortex scan failed"); }

  nanoarrow::UniqueSchema schema;
  input.check(input.value->get_schema(input.value.get(), schema.get()), operation);
  detail::check_flat_schema(*schema.get());
  cudf::io::table_with_metadata result;
  for (int64_t i = 0; i < schema->n_children; ++i) {
    auto& info        = result.metadata.schema_info.emplace_back();
    auto const& field = *schema->children[i];
    info.name         = field.name ? field.name : "";
    info.is_nullable  = (field.flags & ARROW_FLAG_NULLABLE) != 0;
  }

  std::vector<device_batch> batches;
  stream_drain drain{impl_->stream};
  int64_t rows = 0;
  while (true) {
    batches.emplace_back();
    auto& batch = batches.back();
    {
      nvtx3::scoped_range range{"vortex.get_next"};
      input.check(input.value->get_next(input.value.get(), batch.value.get()), operation);
    }
    if (!batch.value->array.release) {
      batches.pop_back();
      break;
    }
    if (batch.value->device_type != ARROW_DEVICE_CUDA || batch.value->device_id != 0) {
      throw std::runtime_error("read_vortex expected an Arrow CUDA batch on device 0");
    }
    auto const count = batch.value->array.length;
    if (count < 0 || count > std::numeric_limits<cudf::size_type>::max() - rows) {
      throw std::overflow_error(operation + ": result exceeds the cuDF row limit");
    }
    rows += count;
    {
      nvtx3::scoped_range range{"vortex.arrow_device_import"};
      batch.view.emplace(cudf::from_arrow_device(
        schema.get(), batch.value.get(), cuda::stream_ref{impl_->stream}, impl_->mr));
    }
    for (auto const& column : **batch.view) {
      if (column.type().id() == cudf::type_id::DICTIONARY32) {
        throw std::runtime_error(operation + ": decoded scan returned a dictionary column");
      }
    }
  }
  {
    nvtx3::scoped_range range{"vortex.materialize"};
    if (batches.empty()) {
      result.tbl = empty_table(*schema.get(), impl_->stream, impl_->mr);
    } else if (batches.size() == 1) {
      result.tbl = std::make_unique<cudf::table>(
        **batches.front().view, cuda::stream_ref{impl_->stream}, impl_->mr);
    } else {
      std::vector<cudf::table_view> views;
      views.reserve(batches.size());
      for (auto const& batch : batches) {
        views.push_back(**batch.view);
      }
      result.tbl = cudf::concatenate(views, cuda::stream_ref{impl_->stream}, impl_->mr);
    }
  }
  {
    nvtx3::scoped_range range{"vortex.consumer_sync"};
    drain.wait();
  }
  result.metadata.num_rows_per_source = {static_cast<std::size_t>(rows)};
  // Import scratch and Vortex buffers are released only after materialization completes.
  {
    nvtx3::scoped_range range{"vortex.release_batches"};
    batches.clear();
    // Include async frees of cuDF import scratch in consumer-stream completion.
    drain.wait();
  }
  return result;
}

}  // namespace ndsh
