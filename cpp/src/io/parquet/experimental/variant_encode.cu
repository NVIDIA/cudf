/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/experimental/variant.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cstring>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace cudf {
namespace io::parquet::experimental {
namespace {

// ──────────────────────────────────────────────────────────────────────────────
// Encoding constants (from the VARIANT spec; mirrors variant_extract.cu)
// ──────────────────────────────────────────────────────────────────────────────

constexpr uint8_t k_null_value         = 0x00;
constexpr uint8_t k_bool_true_header   = 0x04;  // primitive type 1 (BOOLEAN_TRUE)
constexpr uint8_t k_bool_false_header  = 0x08;  // primitive type 2 (BOOLEAN_FALSE)
constexpr uint8_t k_int8_header        = 0x0c;
constexpr uint8_t k_int16_header       = 0x10;
constexpr uint8_t k_int32_header       = 0x14;
constexpr uint8_t k_int64_header       = 0x18;
constexpr uint8_t k_float64_header     = 0x1c;  // primitive type 7
constexpr uint8_t k_float32_header     = 0x38;  // primitive type 14
constexpr uint8_t k_long_string_header = 0x40;  // primitive type 16

// short_string: basic_type=1, length in bits 2..7 of the value_metadata byte
constexpr uint8_t k_short_string_basic_type = 0x01;

// Object with field_id_size=1, field_offset_size=4, num_elements_size=1:
//   value_header = (is_large=0 << 4) | (field_id_size-1=0 << 2) | (field_offset_size-1=3) = 0x03
//   value_metadata = (value_header << 2) | basic_type::object(2) = 0x0e
constexpr uint8_t k_object_value_metadata = 0x0e;

constexpr int block_size = 256;

// ──────────────────────────────────────────────────────────────────────────────
// Per-field device helpers
// ──────────────────────────────────────────────────────────────────────────────

__device__ int64_t field_encoded_size(column_device_view const& col, size_type row)
{
  if (col.type().id() == type_id::EMPTY || !col.is_valid(row)) { return 1; }
  switch (col.type().id()) {
    case type_id::BOOL8: return 1;
    case type_id::INT8: return 2;
    case type_id::INT16: return 3;
    case type_id::INT32: return 5;
    case type_id::INT64: return 9;
    case type_id::FLOAT32: return 5;
    case type_id::FLOAT64: return 9;
    case type_id::STRING: {
      auto const len = static_cast<int64_t>(col.element<cudf::string_view>(row).size_bytes());
      return len < 64 ? 1 + len : 5 + len;
    }
    default: return 1;
  }
}

// Write the encoded bytes for one field into `out`, returns bytes written.
__device__ size_type write_field_value(uint8_t* out, column_device_view const& col, size_type row)
{
  if (col.type().id() == type_id::EMPTY || !col.is_valid(row)) {
    out[0] = k_null_value;
    return 1;
  }
  switch (col.type().id()) {
    case type_id::BOOL8: {
      out[0] = col.element<bool>(row) ? k_bool_true_header : k_bool_false_header;
      return 1;
    }
    case type_id::INT8: {
      auto const v = col.element<int8_t>(row);
      out[0]       = k_int8_header;
      cuda::std::memcpy(out + 1, &v, 1);
      return 2;
    }
    case type_id::INT16: {
      auto const v = col.element<int16_t>(row);
      out[0]       = k_int16_header;
      cuda::std::memcpy(out + 1, &v, 2);
      return 3;
    }
    case type_id::INT32: {
      auto const v = col.element<int32_t>(row);
      out[0]       = k_int32_header;
      cuda::std::memcpy(out + 1, &v, 4);
      return 5;
    }
    case type_id::INT64: {
      auto const v = col.element<int64_t>(row);
      out[0]       = k_int64_header;
      cuda::std::memcpy(out + 1, &v, 8);
      return 9;
    }
    case type_id::FLOAT32: {
      auto const v = col.element<float>(row);
      out[0]       = k_float32_header;
      cuda::std::memcpy(out + 1, &v, 4);
      return 5;
    }
    case type_id::FLOAT64: {
      auto const v = col.element<double>(row);
      out[0]       = k_float64_header;
      cuda::std::memcpy(out + 1, &v, 8);
      return 9;
    }
    case type_id::STRING: {
      auto const sv  = col.element<cudf::string_view>(row);
      auto const len = static_cast<size_type>(sv.size_bytes());
      if (len < 64) {
        out[0] = static_cast<uint8_t>(k_short_string_basic_type | (len << 2));
        cuda::std::memcpy(out + 1, sv.data(), len);
        return 1 + len;
      } else {
        out[0]             = k_long_string_header;
        uint32_t const u32 = static_cast<uint32_t>(len);
        cuda::std::memcpy(out + 1, &u32, 4);
        cuda::std::memcpy(out + 5, sv.data(), len);
        return 5 + len;
      }
    }
    default: out[0] = k_null_value; return 1;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// encode_variant kernels
// ──────────────────────────────────────────────────────────────────────────────

// Pass 1: compute per-row value blob sizes.
CUDF_KERNEL __launch_bounds__(block_size) void object_value_sizes_kernel(
  cudf::table_device_view tbl,
  device_span<int32_t const> sort_order,
  device_span<size_type> d_val_sizes)
{
  auto const num_rows = static_cast<size_type>(d_val_sizes.size());
  auto const N        = tbl.num_columns();
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    // Header: value_metadata(1) + num_elements(1) + field_ids(N) + field_offsets((N+1)*4)
    int64_t size = static_cast<int64_t>(2 + N + (N + 1) * 4);
    for (int i = 0; i < N; i++) {
      size += field_encoded_size(tbl.column(sort_order[i]), row);
    }
    // Saturate to INT32_MAX so the host-side total-bytes overflow check fires.
    d_val_sizes[row] = size <= static_cast<int64_t>(cuda::std::numeric_limits<size_type>::max())
                         ? static_cast<size_type>(size)
                         : cuda::std::numeric_limits<size_type>::max();
  }
}

// Pass 2: write metadata and value blobs.
// Metadata is constant, stored at stride `meta_size` bytes per row.
CUDF_KERNEL __launch_bounds__(block_size) void object_encode_write_kernel(
  cudf::table_device_view tbl,
  device_span<int32_t const> sort_order,
  device_span<size_type const> d_val_offsets,
  uint8_t* d_val_buf,
  uint8_t* d_meta_buf,
  uint8_t const* d_meta_template,
  size_type meta_size)
{
  auto const num_rows = static_cast<size_type>(d_val_offsets.size()) - 1;
  auto const N        = tbl.num_columns();
  auto const tid      = cudf::detail::grid_1d::global_thread_id<block_size>();
  auto const stride   = cudf::detail::grid_1d::grid_stride<block_size>();

  for (auto row = tid; row < num_rows; row += stride) {
    // Broadcast constant metadata template
    cuda::std::memcpy(d_meta_buf + row * meta_size, d_meta_template, meta_size);

    // Write the VARIANT object value
    uint8_t* p = d_val_buf + d_val_offsets[row];

    *p++ = k_object_value_metadata;
    *p++ = static_cast<uint8_t>(N);

    // field_ids: sorted column IDs are 0..N-1 (dictionary is in sorted order)
    for (int i = 0; i < N; i++) {
      *p++ = static_cast<uint8_t>(i);
    }

    // Write (N+1) field offsets, 4 bytes each (LE).
    // field_encoded_size is called once per field here (and again below in write_field_value),
    // which avoids the 1 KiB field_sizes[256] stack array that would spill to local memory.
    uint32_t running = 0;
    for (int i = 0; i <= N; i++) {
      cuda::std::memcpy(p, &running, 4);
      p += 4;
      if (i < N) {
        running += static_cast<uint32_t>(field_encoded_size(tbl.column(sort_order[i]), row));
      }
    }

    // Write field values in sorted order
    for (int i = 0; i < N; i++) {
      p += write_field_value(p, tbl.column(sort_order[i]), row);
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Host: build metadata blob and column sort order
// ──────────────────────────────────────────────────────────────────────────────

// Returns (metadata_bytes, sort_order) where sort_order[i] is the original column index of
// the i-th lexicographically sorted column. Column names appear in sorted order in the blob.
std::pair<std::vector<uint8_t>, std::vector<int32_t>> build_metadata_blob(
  cudf::host_span<std::string const> column_names)
{
  auto const N = static_cast<int>(column_names.size());

  std::vector<int32_t> sort_order(N);
  std::iota(sort_order.begin(), sort_order.end(), 0);
  std::stable_sort(sort_order.begin(), sort_order.end(), [&](int a, int b) {
    return column_names[a] < column_names[b];
  });

  // Choose offset_size based on total UTF-8 key length
  std::size_t total_key_bytes = 0;
  for (auto const& name : column_names) {
    total_key_bytes += name.size();
  }
  int const offset_size = total_key_bytes <= 0xFFu ? 1 : total_key_bytes <= 0xFFFFu ? 2 : 4;

  // header: bits[7:6] = offset_size-1, bits[3:0] = version=1
  uint8_t const header = static_cast<uint8_t>(((offset_size - 1) << 6) | 0x01u);

  std::vector<uint8_t> blob;
  blob.push_back(header);

  // num_keys (offset_size bytes LE)
  for (int b = 0; b < offset_size; b++) {
    blob.push_back(static_cast<uint8_t>((static_cast<uint32_t>(N) >> (8 * b)) & 0xFFu));
  }

  // offsets[0..N] (offset_size bytes each LE), relative to start of string_data
  std::size_t running = 0;
  auto push_offset    = [&](std::size_t v) {
    for (int b = 0; b < offset_size; b++) {
      blob.push_back(static_cast<uint8_t>((v >> (8 * b)) & 0xFFu));
    }
  };
  push_offset(0);
  for (int i = 0; i < N; i++) {
    running += column_names[sort_order[i]].size();
    push_offset(running);
  }

  // string_data: keys in sorted order
  for (int i = 0; i < N; i++) {
    auto const& name = column_names[sort_order[i]];
    blob.insert(blob.end(), name.begin(), name.end());
  }

  return {std::move(blob), std::move(sort_order)};
}

// Build a list<uint8> column for constant-stride blobs (all rows the same M bytes).
// Offsets are [0, M, 2M, ..., num_rows*M]; data is filled by caller via kernel.
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> make_constant_list_buffers(
  size_type num_rows, size_type M, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(static_cast<int64_t>(num_rows) * M <= std::numeric_limits<size_type>::max(),
               "metadata size exceeds 2 GiB limit",
               std::overflow_error);
  auto offsets = make_numeric_column(
    data_type{type_id::INT32}, num_rows + 1, mask_state::UNALLOCATED, stream, mr);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   offsets->mutable_view().begin<int32_t>(),
                   offsets->mutable_view().end<int32_t>(),
                   int32_t{0},
                   static_cast<int32_t>(M));

  auto data = make_numeric_column(
    data_type{type_id::UINT8}, num_rows * M, mask_state::UNALLOCATED, stream, mr);
  return {std::move(offsets), std::move(data)};
}

}  // namespace

namespace detail {

// ──────────────────────────────────────────────────────────────────────────────
// encode_variant
// ──────────────────────────────────────────────────────────────────────────────

std::unique_ptr<column> encode_variant(cudf::table_view const& input,
                                       cudf::host_span<std::string const> column_names,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const N        = input.num_columns();
  auto const num_rows = input.num_rows();

  CUDF_EXPECTS(static_cast<std::size_t>(N) == column_names.size(),
               "encode_variant: column_names.size() must equal input.num_columns()",
               std::invalid_argument);
  CUDF_EXPECTS(
    N < 256, "encode_variant: table must have fewer than 256 columns", std::invalid_argument);

  for (int i = 0; i < N; i++) {
    auto const id = input.column(i).type().id();
    CUDF_EXPECTS(id == type_id::EMPTY || id == type_id::BOOL8 || id == type_id::INT8 ||
                   id == type_id::INT16 || id == type_id::INT32 || id == type_id::INT64 ||
                   id == type_id::FLOAT32 || id == type_id::FLOAT64 || id == type_id::STRING,
                 "encode_variant: unsupported column type — supported: EMPTY, BOOL8, "
                 "INT8/16/32/64, FLOAT32/64, STRING",
                 std::invalid_argument);
  }

  // ── Metadata blob, sort order, and duplicate check (sort once) ──
  auto [meta_bytes, sort_order] = build_metadata_blob(column_names);
  for (int i = 1; i < N; i++) {
    CUDF_EXPECTS(column_names[sort_order[i]] != column_names[sort_order[i - 1]],
                 "encode_variant: duplicate column names are not allowed",
                 std::invalid_argument);
  }

  auto make_empty_list = [&] {
    return make_lists_column(
      0, make_empty_column(type_id::INT32), make_empty_column(type_id::UINT8), 0, {});
  };

  if (num_rows == 0) {
    std::vector<std::unique_ptr<column>> ch;
    ch.push_back(make_empty_list());
    ch.push_back(make_empty_list());
    return make_structs_column(0, std::move(ch), 0, {}, stream, mr);
  }

  auto const M = static_cast<size_type>(meta_bytes.size());

  rmm::device_uvector<uint8_t> d_meta_template = cudf::detail::make_device_uvector_async(
    meta_bytes, stream, cudf::get_current_device_resource_ref());

  rmm::device_uvector<int32_t> d_sort_order = cudf::detail::make_device_uvector_async(
    sort_order, stream, cudf::get_current_device_resource_ref());

  // ── Column device views on device (via table_device_view) ──
  auto d_table = cudf::table_device_view::create(input, stream);

  // ── Metadata buffers: constant M bytes per row ──
  auto [meta_offsets_col, meta_data_col] = make_constant_list_buffers(num_rows, M, stream, mr);

  // ── Value sizes (pass 1) ──
  rmm::device_uvector<size_type> d_val_sizes(
    num_rows, stream, cudf::get_current_device_resource_ref());
  {
    cudf::detail::grid_1d grid{num_rows, block_size};
    object_value_sizes_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      *d_table, d_sort_order, d_val_sizes);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  auto [val_offsets_col, total_val_bytes] =
    cudf::detail::make_offsets_child_column(d_val_sizes.begin(), d_val_sizes.end(), stream, mr);
  CUDF_EXPECTS(val_offsets_col->type().id() == type_id::INT32,
               "VARIANT value bytes exceed 2 GiB limit",
               std::overflow_error);

  auto val_data_col = make_numeric_column(data_type{type_id::UINT8},
                                          static_cast<size_type>(total_val_bytes),
                                          mask_state::UNALLOCATED,
                                          stream,
                                          mr);

  // ── Write pass (pass 2): metadata + values ──
  {
    device_span<size_type const> d_val_offsets{val_offsets_col->view().data<size_type>(),
                                               static_cast<std::size_t>(num_rows + 1)};
    cudf::detail::grid_1d grid{num_rows, block_size};
    object_encode_write_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
      *d_table,
      d_sort_order,
      d_val_offsets,
      val_data_col->mutable_view().data<uint8_t>(),
      meta_data_col->mutable_view().data<uint8_t>(),
      d_meta_template.data(),
      M);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // ── Assemble STRUCT<list<uint8>, list<uint8>> (output rows are never null) ──
  auto meta_col =
    make_lists_column(num_rows, std::move(meta_offsets_col), std::move(meta_data_col), 0, {});
  auto val_col =
    make_lists_column(num_rows, std::move(val_offsets_col), std::move(val_data_col), 0, {});

  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(meta_col));
  children.push_back(std::move(val_col));
  return make_structs_column(num_rows, std::move(children), 0, {}, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> encode_variant(cudf::table_view const& input,
                                       cudf::host_span<std::string const> column_names,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::encode_variant(input, column_names, stream, mr);
}

}  // namespace io::parquet::experimental
}  // namespace cudf
