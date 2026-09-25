/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/protobuf/device_helpers.cuh"
#include "io/protobuf/host_helpers.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <cuda/std/type_traits>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf::io::protobuf::detail {

// ============================================================================
// Pass 2: Extract data kernels
// ============================================================================

// ============================================================================
// Data Extraction Location Providers
// ============================================================================

struct top_level_location_provider {
  cudf::size_type const* offsets;
  cudf::size_type base_offset;
  field_location const* locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto loc = locations[flat_index(static_cast<size_t>(thread_idx),
                                    static_cast<size_t>(num_fields),
                                    static_cast<size_t>(field_idx))];
    if (loc.offset >= 0) { data_offset = offsets[thread_idx] - base_offset + loc.offset; }
    return loc;
  }
};

struct repeated_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  repeated_occurrence const* occurrences;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto occ    = occurrences[thread_idx];
    data_offset = row_offsets[occ.row_idx] - base_offset + occ.offset;
    return {occ.offset, occ.length};
  }
};

struct nested_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto ploc = parent_locations[thread_idx];
    auto cloc = child_locations[flat_index(static_cast<size_t>(thread_idx),
                                           static_cast<size_t>(num_fields),
                                           static_cast<size_t>(field_idx))];
    if (ploc.offset >= 0 && cloc.offset >= 0) {
      data_offset = row_offsets[thread_idx] - base_offset + ploc.offset + cloc.offset;
    } else {
      cloc.offset = -1;
    }
    return cloc;
  }
};

struct nested_repeated_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  repeated_occurrence const* occurrences;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto occ  = occurrences[thread_idx];
    auto ploc = parent_locations[occ.row_idx];
    if (ploc.offset >= 0) {
      data_offset = row_offsets[occ.row_idx] - base_offset + ploc.offset + occ.offset;
      return {occ.offset, occ.length};
    }
    data_offset = 0;
    return {-1, 0};
  }
};

struct repeated_msg_child_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* msg_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto mloc = msg_locations[thread_idx];
    auto cloc = child_locations[flat_index(static_cast<size_t>(thread_idx),
                                           static_cast<size_t>(num_fields),
                                           static_cast<size_t>(field_idx))];
    if (mloc.offset >= 0 && cloc.offset >= 0) {
      data_offset = row_offsets[thread_idx] - base_offset + mloc.offset + cloc.offset;
    } else {
      cloc.offset = -1;
    }
    return cloc;
  }
};

template <typename OutputType, bool ZigZag = false, typename LocationProvider>
CUDF_KERNEL void extract_varint_kernel(uint8_t const* message_data,
                                       LocationProvider loc_provider,
                                       int total_items,
                                       OutputType* out,
                                       bool* valid,
                                       int* error_flag,
                                       bool has_default      = false,
                                       int64_t default_value = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  // For BOOL8 (uint8_t), protobuf spec says any non-zero varint is true.
  // A raw static_cast<uint8_t> would silently truncate values >= 256 to 0.
  auto const write_value = [](OutputType* dst, uint64_t val) {
    if constexpr (cuda::std::is_same_v<OutputType, uint8_t>) {
      *dst = static_cast<uint8_t>(val != 0 ? 1 : 0);
    } else {
      *dst = static_cast<OutputType>(val);
    }
  };

  if (loc.offset < 0) {
    if (has_default) {
      write_value(&out[idx], static_cast<uint64_t>(default_value));
      if (valid) valid[idx] = true;
    } else {
      if (valid) valid[idx] = false;
    }
    return;
  }

  uint8_t const* cur     = message_data + data_offset;
  uint8_t const* cur_end = cur + loc.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, cur_end, v, n)) {
    set_error_once(error_flag, proto_decode_error::VARINT);
    if (valid) valid[idx] = false;
    return;
  }

  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  write_value(&out[idx], v);
  if (valid) valid[idx] = true;
}

template <typename OutputType, int WT, typename LocationProvider>
CUDF_KERNEL void extract_fixed_kernel(uint8_t const* message_data,
                                      LocationProvider loc_provider,
                                      int total_items,
                                      OutputType* out,
                                      bool* valid,
                                      int* error_flag,
                                      bool has_default         = false,
                                      OutputType default_value = OutputType{})
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  if (loc.offset < 0) {
    if (has_default) {
      out[idx] = default_value;
      if (valid) valid[idx] = true;
    } else {
      if (valid) valid[idx] = false;
    }
    return;
  }

  uint8_t const* cur = message_data + data_offset;
  OutputType value;

  if constexpr (WT == wire_type_value(proto_wire_type::I32BIT)) {
    if (loc.length < 4) {
      set_error_once(error_flag, proto_decode_error::FIXED_LEN);
      if (valid) valid[idx] = false;
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (loc.length < 8) {
      set_error_once(error_flag, proto_decode_error::FIXED_LEN);
      if (valid) valid[idx] = false;
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }

  out[idx] = value;
  if (valid) valid[idx] = true;
}

// ============================================================================
// Batched scalar extraction — one 2D kernel for N fields of the same type
// ============================================================================

struct batched_scalar_desc {
  int loc_field_idx;  // index into the locations array (column within d_locations)
  void* output;       // pre-allocated output buffer (T*)
  bool* valid;        // pre-allocated validity buffer
  bool has_default;
  int64_t default_int;
  double default_float;
};

template <typename OutputType, bool ZigZag = false>
CUDF_KERNEL void extract_varint_batched_kernel(uint8_t const* message_data,
                                               cudf::size_type const* row_offsets,
                                               cudf::size_type base_offset,
                                               field_location const* locations,
                                               int num_loc_fields,
                                               batched_scalar_desc const* descs,
                                               int num_descs,
                                               int num_rows,
                                               int* error_flag)
{
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int fi  = static_cast<int>(blockIdx.y);
  if (row >= num_rows || fi >= num_descs) return;

  auto const& desc = descs[fi];
  auto loc         = locations[row * num_loc_fields + desc.loc_field_idx];
  auto* out        = static_cast<OutputType*>(desc.output);

  auto const write_value = [](OutputType* dst, uint64_t val) {
    if constexpr (cuda::std::is_same_v<OutputType, uint8_t>) {
      *dst = static_cast<uint8_t>(val != 0 ? 1 : 0);
    } else {
      *dst = static_cast<OutputType>(val);
    }
  };

  if (loc.offset < 0) {
    if (desc.has_default) {
      write_value(&out[row], static_cast<uint64_t>(desc.default_int));
      desc.valid[row] = true;
    } else {
      desc.valid[row] = false;
    }
    return;
  }

  int32_t data_offset = row_offsets[row] - base_offset + loc.offset;
  uint8_t const* cur  = message_data + data_offset;
  uint8_t const* end  = cur + loc.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, end, v, n)) {
    set_error_once(error_flag, proto_decode_error::VARINT);
    desc.valid[row] = false;
    return;
  }
  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  write_value(&out[row], v);
  desc.valid[row] = true;
}

template <typename OutputType, int WT>
CUDF_KERNEL void extract_fixed_batched_kernel(uint8_t const* message_data,
                                              cudf::size_type const* row_offsets,
                                              cudf::size_type base_offset,
                                              field_location const* locations,
                                              int num_loc_fields,
                                              batched_scalar_desc const* descs,
                                              int num_descs,
                                              int num_rows,
                                              int* error_flag)
{
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int fi  = static_cast<int>(blockIdx.y);
  if (row >= num_rows || fi >= num_descs) return;

  auto const& desc = descs[fi];
  auto loc         = locations[row * num_loc_fields + desc.loc_field_idx];
  auto* out        = static_cast<OutputType*>(desc.output);

  if (loc.offset < 0) {
    if (desc.has_default) {
      if constexpr (cuda::std::is_integral_v<OutputType>) {
        out[row] = static_cast<OutputType>(desc.default_int);
      } else {
        out[row] = static_cast<OutputType>(desc.default_float);
      }
      desc.valid[row] = true;
    } else {
      desc.valid[row] = false;
    }
    return;
  }

  int32_t data_offset = row_offsets[row] - base_offset + loc.offset;
  uint8_t const* cur  = message_data + data_offset;
  OutputType value;

  if constexpr (WT == wire_type_value(proto_wire_type::I32BIT)) {
    if (loc.length < 4) {
      set_error_once(error_flag, proto_decode_error::FIXED_LEN);
      desc.valid[row] = false;
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (loc.length < 8) {
      set_error_once(error_flag, proto_decode_error::FIXED_LEN);
      desc.valid[row] = false;
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }
  out[row]        = value;
  desc.valid[row] = true;
}

// ============================================================================

template <typename LocationProvider>
CUDF_KERNEL void extract_lengths_kernel(LocationProvider loc_provider,
                                        int total_items,
                                        int32_t* out_lengths,
                                        bool has_default       = false,
                                        int32_t default_length = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  if (loc.offset >= 0) {
    out_lengths[idx] = loc.length;
  } else if (has_default) {
    out_lengths[idx] = default_length;
  } else {
    out_lengths[idx] = 0;
  }
}

// ============================================================================
// Host-side template helpers that launch CUDA kernels
// ============================================================================

template <typename T>
inline std::pair<rmm::device_buffer, cudf::size_type> make_null_mask_from_valid(
  rmm::device_uvector<T> const& valid,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + valid.size();
  auto pred  = [ptr = valid.data()] __device__(cudf::size_type i) {
    return static_cast<bool>(ptr[i]);
  };
  return cudf::detail::valid_if(begin, end, pred, stream, mr);
}

template <typename T, typename LaunchFn>
std::unique_ptr<cudf::column> extract_and_build_scalar_column(cudf::data_type dt,
                                                              int num_rows,
                                                              LaunchFn&& launch_extract,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> out(num_rows, stream, mr);
  if (num_rows == 0) {
    return std::make_unique<cudf::column>(dt, 0, out.release(), rmm::device_buffer{}, 0);
  }
  rmm::device_uvector<bool> valid(num_rows, stream, cudf::get_current_device_resource_ref());
  launch_extract(out.data(), valid.data());
  auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
  return std::make_unique<cudf::column>(dt, num_rows, out.release(), std::move(mask), null_count);
}

template <typename T, typename LocationProvider>
inline void extract_integer_into_buffers(uint8_t const* message_data,
                                         LocationProvider const& loc_provider,
                                         int num_rows,
                                         int blocks,
                                         int threads,
                                         bool has_default,
                                         int64_t default_value,
                                         int encoding,
                                         bool enable_zigzag,
                                         T* out_ptr,
                                         bool* valid_ptr,
                                         int* error_ptr,
                                         rmm::cuda_stream_view stream)
{
  if (enable_zigzag && encoding == encoding_value(proto_encoding::ZIGZAG)) {
    extract_varint_kernel<T, true, LocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(message_data,
                                               loc_provider,
                                               num_rows,
                                               out_ptr,
                                               valid_ptr,
                                               error_ptr,
                                               has_default,
                                               default_value);
  } else if (encoding == encoding_value(proto_encoding::FIXED)) {
    if constexpr (sizeof(T) == 4) {
      extract_fixed_kernel<T, wire_type_value(proto_wire_type::I32BIT), LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                 loc_provider,
                                                 num_rows,
                                                 out_ptr,
                                                 valid_ptr,
                                                 error_ptr,
                                                 has_default,
                                                 static_cast<T>(default_value));
    } else {
      static_assert(sizeof(T) == 8, "extract_integer_into_buffers only supports 32/64-bit");
      extract_fixed_kernel<T, wire_type_value(proto_wire_type::I64BIT), LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                 loc_provider,
                                                 num_rows,
                                                 out_ptr,
                                                 valid_ptr,
                                                 error_ptr,
                                                 has_default,
                                                 static_cast<T>(default_value));
    }
  } else {
    extract_varint_kernel<T, false, LocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(message_data,
                                               loc_provider,
                                               num_rows,
                                               out_ptr,
                                               valid_ptr,
                                               error_ptr,
                                               has_default,
                                               default_value);
  }
}

template <typename T, typename LocationProvider>
std::unique_ptr<cudf::column> extract_and_build_integer_column(cudf::data_type dt,
                                                               uint8_t const* message_data,
                                                               LocationProvider const& loc_provider,
                                                               int num_rows,
                                                               int blocks,
                                                               int threads,
                                                               rmm::device_uvector<int>& d_error,
                                                               bool has_default,
                                                               int64_t default_value,
                                                               int encoding,
                                                               bool enable_zigzag,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  return extract_and_build_scalar_column<T>(
    dt,
    num_rows,
    [&](T* out_ptr, bool* valid_ptr) {
      extract_integer_into_buffers<T, LocationProvider>(message_data,
                                                        loc_provider,
                                                        num_rows,
                                                        blocks,
                                                        threads,
                                                        has_default,
                                                        default_value,
                                                        encoding,
                                                        enable_zigzag,
                                                        out_ptr,
                                                        valid_ptr,
                                                        d_error.data(),
                                                        stream);
    },
    stream,
    mr);
}

struct extract_strided_count {
  repeated_field_info const* info;
  int field_idx;
  int num_fields;

  __device__ int32_t operator()(int row) const
  {
    return info[flat_index(static_cast<size_t>(row),
                           static_cast<size_t>(num_fields),
                           static_cast<size_t>(field_idx))]
      .count;
  }
};

template <typename LengthProvider, typename CopyProvider, typename ValidityFn>
inline std::unique_ptr<cudf::column> extract_and_build_string_or_bytes_column(
  bool as_bytes,
  uint8_t const* message_data,
  int num_rows,
  LengthProvider const& length_provider,
  CopyProvider const& copy_provider,
  ValidityFn validity_fn,
  bool has_default,
  decode_protobuf_options::byte_vector const& default_bytes,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  int32_t def_len = has_default ? static_cast<int32_t>(default_bytes.size()) : 0;
  auto temp_mr    = cudf::get_current_device_resource_ref();
  rmm::device_uvector<uint8_t> d_default(0, stream, temp_mr);
  if (has_default && def_len > 0) {
    d_default = cudf::detail::make_device_uvector_async(default_bytes, stream, temp_mr);
  }

  rmm::device_uvector<int32_t> lengths(num_rows, stream, temp_mr);
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((num_rows + threads - 1u) / threads);
  extract_lengths_kernel<LengthProvider><<<blocks, threads, 0, stream.value()>>>(
    length_provider, num_rows, lengths.data(), has_default, def_len);

  auto [offsets_col, total_size] =
    cudf::strings::detail::make_offsets_child_column(lengths.begin(), lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_size, stream, mr);
  if (total_size > 0) {
    auto const* offsets_data = offsets_col->view().data<cudf::size_type>();
    auto* chars_ptr          = chars.data();
    auto const* default_ptr  = d_default.data();

    auto src_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<void const*>(
        [message_data, copy_provider, has_default, default_ptr, def_len] __device__(
          int idx) -> void const* {
          int32_t data_offset = 0;
          auto loc            = copy_provider.get(idx, data_offset);
          if (loc.offset < 0) {
            return (has_default && def_len > 0) ? static_cast<void const*>(default_ptr) : nullptr;
          }
          return static_cast<void const*>(message_data + data_offset);
        }));
    auto dst_iter = cudf::detail::make_counting_transform_iterator(
      0, cuda::proclaim_return_type<void*>([chars_ptr, offsets_data] __device__(int idx) -> void* {
        return static_cast<void*>(chars_ptr + offsets_data[idx]);
      }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<size_t>(
        [copy_provider, has_default, def_len] __device__(int idx) -> size_t {
          int32_t data_offset = 0;
          auto loc            = copy_provider.get(idx, data_offset);
          if (loc.offset < 0) {
            return (has_default && def_len > 0) ? static_cast<size_t>(def_len) : 0;
          }
          return static_cast<size_t>(loc.length);
        }));

    size_t temp_storage_bytes = 0;
    cub::DeviceMemcpy::Batched(
      nullptr, temp_storage_bytes, src_iter, dst_iter, size_iter, num_rows, stream.value());
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, temp_mr);
    cub::DeviceMemcpy::Batched(temp_storage.data(),
                               temp_storage_bytes,
                               src_iter,
                               dst_iter,
                               size_iter,
                               num_rows,
                               stream.value());
  }

  if (num_rows == 0) {
    if (as_bytes) {
      auto bytes_child = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::UINT8}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      return cudf::make_lists_column(
        0, std::move(offsets_col), std::move(bytes_child), 0, rmm::device_buffer{});
    }
    return cudf::make_strings_column(
      0, std::move(offsets_col), chars.release(), 0, rmm::device_buffer{});
  }

  rmm::device_uvector<bool> valid(num_rows, stream, temp_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    valid.data(),
                    validity_fn);
  auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
  if (as_bytes) {
    auto bytes_child = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, total_size, chars.release(), rmm::device_buffer{}, 0);
    return cudf::make_lists_column(
      num_rows, std::move(offsets_col), std::move(bytes_child), null_count, std::move(mask));
  }

  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
}

template <typename LocationProvider>
inline std::unique_ptr<cudf::column> extract_typed_column(
  cudf::data_type dt,
  int encoding,
  uint8_t const* message_data,
  LocationProvider const& loc_provider,
  int num_items,
  int blocks,
  int threads_per_block,
  bool has_default,
  int64_t default_int,
  double default_float,
  bool default_bool,
  decode_protobuf_options::byte_vector const& default_string,
  int schema_idx,
  std::vector<decode_protobuf_options::enum_value_vector> const& enum_valid_values,
  std::vector<decode_protobuf_options::enum_name_vector> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices = nullptr,
  bool propagate_invalid_rows    = true)
{
  switch (dt.id()) {
    case cudf::type_id::BOOL8: {
      int64_t def_val = has_default ? (default_bool ? 1 : 0) : 0;
      return extract_and_build_scalar_column<uint8_t>(
        dt,
        num_items,
        [&](uint8_t* out_ptr, bool* valid_ptr) {
          extract_varint_kernel<uint8_t, false, LocationProvider>
            <<<blocks, threads_per_block, 0, stream.value()>>>(message_data,
                                                               loc_provider,
                                                               num_items,
                                                               out_ptr,
                                                               valid_ptr,
                                                               d_error.data(),
                                                               has_default,
                                                               def_val);
        },
        stream,
        mr);
    }
    case cudf::type_id::INT32: {
      if (num_items == 0) {
        return std::make_unique<cudf::column>(dt, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      }
      rmm::device_uvector<int32_t> out(num_items, stream, mr);
      rmm::device_uvector<bool> valid(num_items, stream, mr);
      extract_integer_into_buffers<int32_t, LocationProvider>(message_data,
                                                              loc_provider,
                                                              num_items,
                                                              blocks,
                                                              threads_per_block,
                                                              has_default,
                                                              default_int,
                                                              encoding,
                                                              true,
                                                              out.data(),
                                                              valid.data(),
                                                              d_error.data(),
                                                              stream);
      if (schema_idx < static_cast<int>(enum_valid_values.size())) {
        auto const& valid_enums = enum_valid_values[schema_idx];
        if (!valid_enums.empty()) {
          validate_enum_and_propagate_rows(out,
                                           valid,
                                           valid_enums,
                                           d_row_force_null,
                                           num_items,
                                           top_row_indices,
                                           propagate_invalid_rows,
                                           stream);
        }
      }
      auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
      return std::make_unique<cudf::column>(
        dt, num_items, out.release(), std::move(mask), null_count);
    }
    case cudf::type_id::UINT32:
      return extract_and_build_integer_column<uint32_t>(dt,
                                                        message_data,
                                                        loc_provider,
                                                        num_items,
                                                        blocks,
                                                        threads_per_block,
                                                        d_error,
                                                        has_default,
                                                        default_int,
                                                        encoding,
                                                        false,
                                                        stream,
                                                        mr);
    case cudf::type_id::INT64:
      return extract_and_build_integer_column<int64_t>(dt,
                                                       message_data,
                                                       loc_provider,
                                                       num_items,
                                                       blocks,
                                                       threads_per_block,
                                                       d_error,
                                                       has_default,
                                                       default_int,
                                                       encoding,
                                                       true,
                                                       stream,
                                                       mr);
    case cudf::type_id::UINT64:
      return extract_and_build_integer_column<uint64_t>(dt,
                                                        message_data,
                                                        loc_provider,
                                                        num_items,
                                                        blocks,
                                                        threads_per_block,
                                                        d_error,
                                                        has_default,
                                                        default_int,
                                                        encoding,
                                                        false,
                                                        stream,
                                                        mr);
    case cudf::type_id::FLOAT32: {
      float def_float_val = has_default ? static_cast<float>(default_float) : 0.0f;
      return extract_and_build_scalar_column<float>(
        dt,
        num_items,
        [&](float* out_ptr, bool* valid_ptr) {
          extract_fixed_kernel<float, wire_type_value(proto_wire_type::I32BIT), LocationProvider>
            <<<blocks, threads_per_block, 0, stream.value()>>>(message_data,
                                                               loc_provider,
                                                               num_items,
                                                               out_ptr,
                                                               valid_ptr,
                                                               d_error.data(),
                                                               has_default,
                                                               def_float_val);
        },
        stream,
        mr);
    }
    case cudf::type_id::FLOAT64: {
      double def_double = has_default ? default_float : 0.0;
      return extract_and_build_scalar_column<double>(
        dt,
        num_items,
        [&](double* out_ptr, bool* valid_ptr) {
          extract_fixed_kernel<double, wire_type_value(proto_wire_type::I64BIT), LocationProvider>
            <<<blocks, threads_per_block, 0, stream.value()>>>(message_data,
                                                               loc_provider,
                                                               num_items,
                                                               out_ptr,
                                                               valid_ptr,
                                                               d_error.data(),
                                                               has_default,
                                                               def_double);
        },
        stream,
        mr);
    }
    default: return make_null_column(dt, num_items, stream, mr);
  }
}

template <typename T>
inline std::unique_ptr<cudf::column> build_repeated_scalar_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  device_nested_field_descriptor const& field_desc,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_null_count = binary_input.null_count();

  if (total_count == 0) {
    rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
    thrust::fill(rmm::exec_policy_nosync(stream, mr), offsets.begin(), offsets.end(), 0);
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      num_rows + 1,
                                                      offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);
    auto elem_type   = field_desc.output_type_id == static_cast<int>(cudf::type_id::LIST)
                         ? cudf::type_id::UINT8
                         : static_cast<cudf::type_id>(field_desc.output_type_id);
    auto child_col   = make_empty_column_safe(cudf::data_type{elem_type}, stream, mr);

    if (input_null_count > 0) {
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_lists_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(child_col),
                                     input_null_count,
                                     std::move(null_mask));
    } else {
      return cudf::make_lists_column(
        num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
    }
  }

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, mr),
                         d_field_counts.begin(),
                         d_field_counts.end(),
                         list_offs.begin(),
                         0);

  int32_t total_count_i32 = static_cast<int32_t>(total_count);
  thrust::fill_n(
    rmm::exec_policy_nosync(stream, mr), list_offs.data() + num_rows, 1, total_count_i32);

  rmm::device_uvector<T> values(total_count, stream, mr);

  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((total_count + threads - 1u) / threads);

  int encoding = field_desc.encoding;
  bool zigzag  = (encoding == encoding_value(proto_encoding::ZIGZAG));

  constexpr bool is_floating_point = std::is_same_v<T, float> || std::is_same_v<T, double>;
  bool use_fixed_kernel = is_floating_point || (encoding == encoding_value(proto_encoding::FIXED));

  repeated_location_provider loc_provider{list_offsets, base_offset, d_occurrences.data()};
  if (use_fixed_kernel) {
    if constexpr (sizeof(T) == 4) {
      extract_fixed_kernel<T, wire_type_value(proto_wire_type::I32BIT), repeated_location_provider>
        <<<blocks, threads, 0, stream.value()>>>(
          message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
    } else {
      extract_fixed_kernel<T, wire_type_value(proto_wire_type::I64BIT), repeated_location_provider>
        <<<blocks, threads, 0, stream.value()>>>(
          message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
    }
  } else if (zigzag) {
    extract_varint_kernel<T, true, repeated_location_provider>
      <<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
  } else {
    extract_varint_kernel<T, false, repeated_location_provider>
      <<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    list_offs.release(),
                                                    rmm::device_buffer{},
                                                    0);
  auto child_col   = std::make_unique<cudf::column>(
    cudf::data_type{static_cast<cudf::type_id>(field_desc.output_type_id)},
    total_count,
    values.release(),
    rmm::device_buffer{},
    0);

  if (input_null_count > 0) {
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_lists_column(num_rows,
                                   std::move(offsets_col),
                                   std::move(child_col),
                                   input_null_count,
                                   std::move(null_mask));
  }

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
}

}  // namespace cudf::io::protobuf::detail
