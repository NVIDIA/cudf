/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utilities.hpp"
#include "from_arrow_host.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Return bitmask word at the given index in the source
 *
 * This is a 64-bit version of cudf::detail::get_mask_offset_word
 * since the source may have a range more than max(int) bits.
 */
__device__ inline bitmask_type get_mask_word(bitmask_type const* __restrict__ source,
                                             int64_t destination_word_index,
                                             int64_t source_begin_bit,
                                             int64_t source_end_bit)
{
  constexpr auto bitmask_bits = size_in_bits<bitmask_type>();
  auto const word_index       = destination_word_index + (source_begin_bit / bitmask_bits);
  auto const curr_word        = source[word_index];
  auto const end_index        = (source_end_bit - 1) / bitmask_bits;
  auto const next_word        = (end_index > word_index) ? source[word_index + 1] : bitmask_type{0};
  auto const shift            = static_cast<bitmask_type>(source_begin_bit % bitmask_bits);
  return __funnelshift_r(curr_word, next_word, shift);
}

/**
 * @brief Copy a shifted bitmask in device memory
 *
 * Called by get_mask_buffer below when a bit-shift within a bitmask_type is required.
 *
 * @param destination The destination bitmask.
 * @param source The source bitmask.
 * @param source_begin_bit The beginning bit of the source bitmask.
 * @param source_end_bit The end bit of the source bitmask.
 * @param number_of_mask_words The number of mask words.
 */
CUDF_KERNEL void copy_shifted_bitmask(bitmask_type* __restrict__ destination,
                                      bitmask_type const* __restrict__ source,
                                      int64_t source_begin_bit,
                                      int64_t source_end_bit,
                                      size_type number_of_mask_words)
{
  auto const stride = cudf::detail::grid_1d::grid_stride();
  for (thread_index_type destination_word_index = grid_1d::global_thread_id();
       destination_word_index < number_of_mask_words;
       destination_word_index += stride) {
    destination[destination_word_index] =
      detail::get_mask_word(source, destination_word_index, source_begin_bit, source_end_bit);
  }
}

// copies the bitmask to device and automatically applies the offset
std::pair<std::unique_ptr<rmm::device_buffer>, size_type> get_mask_buffer(
  ArrowArray const* input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  if (input->length == 0) { return {std::make_unique<rmm::device_buffer>(0, stream, mr), 0}; }

  auto bitmap = static_cast<uint8_t const*>(input->buffers[validity_buffer_idx]);
  if (bitmap == nullptr || input->null_count == 0) {
    return {std::make_unique<rmm::device_buffer>(0, stream, mr), 0};
  }

  constexpr auto bits_in_byte = static_cast<int64_t>(size_in_bits<uint8_t>());

  auto const num_rows     = static_cast<size_type>(input->length);
  auto const offset_index = input->offset / bits_in_byte;
  auto const mask_words   = num_bitmask_words(num_rows);
  auto const padded_words = bitmask_allocation_size_bytes(num_rows) / sizeof(bitmask_type);
  auto const bit_index    = input->offset % bits_in_byte;
  auto const copy_size    = cudf::util::div_rounding_up_safe(num_rows + bit_index, bits_in_byte);

  auto mask = rmm::device_uvector<bitmask_type>(padded_words, stream, mr);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(mask.data(), bitmap + offset_index, copy_size, stream));

  if (mask_words > 0 && bit_index > 0) {
    auto dest_mask = rmm::device_uvector<bitmask_type>(padded_words, stream, mr);
    cudf::detail::grid_1d config(mask_words, 256);
    copy_shifted_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      dest_mask.data(), mask.data(), bit_index, bit_index + num_rows, mask_words);
    CUDF_CHECK_CUDA(stream.value());
    mask = std::move(dest_mask);
  }

  auto const null_count =
    mask_words > 0 ? cudf::detail::count_unset_bits(mask.data(), 0, num_rows, stream) : 0;

  return {std::make_unique<rmm::device_buffer>(std::move(mask.release())), null_count};
}

std::unique_ptr<column> get_column_copy(ArrowSchemaView const* schema,
                                        ArrowArray const* input,
                                        data_type type,
                                        bool skip_mask,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

struct dispatch_copy_from_arrow_host {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, CUDF_ENABLE_IF(not is_rep_layout_compatible<T>() && !is_fixed_point<T>())>
  std::unique_ptr<column> operator()(ArrowSchemaView const*, ArrowArray const*, data_type, bool)
  {
    CUDF_FAIL("Unsupported type in copy_from_arrow_host.");
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>() || is_fixed_point<T>())>
  std::unique_ptr<column> operator()(ArrowSchemaView const*,
                                     ArrowArray const* input,
                                     data_type type,
                                     bool skip_mask)
  {
    using DeviceType = device_storage_type_t<T>;

    auto const num_rows = static_cast<size_type>(input->length);
    auto const data_buffer =
      static_cast<DeviceType const*>(input->buffers[fixed_width_data_buffer_idx]);

    auto col = make_fixed_width_column(type, num_rows, mask_state::UNALLOCATED, stream, mr);
    auto mutable_column_view = col->mutable_view();
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(mutable_column_view.data<DeviceType>(),
                                             data_buffer + input->offset,
                                             sizeof(DeviceType) * num_rows,
                                             stream));

    if (!skip_mask) {
      auto [mask, null_count] = get_mask_buffer(input, stream, mr);
      col->set_null_mask(std::move(*mask), null_count);
    }

    return col;
  }
};

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<bool>(ArrowSchemaView const*,
                                                                        ArrowArray const* input,
                                                                        data_type type,
                                                                        bool skip_mask)
{
  auto data_buffer = static_cast<uint8_t const*>(input->buffers[fixed_width_data_buffer_idx]);

  constexpr auto bits_in_byte = static_cast<int64_t>(size_in_bits<uint8_t>());

  auto const num_rows     = static_cast<size_type>(input->length);
  auto const offset_index = input->offset / bits_in_byte;
  auto const data_words   = num_bitmask_words(num_rows);
  auto const bit_index    = input->offset % bits_in_byte;
  auto const copy_size    = cudf::util::div_rounding_up_safe(num_rows + bit_index, bits_in_byte);

  auto data = rmm::device_uvector<bitmask_type>(data_words, stream, mr);
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_async(data.data(), data_buffer + offset_index, copy_size, stream));

  if (data_words > 0 && bit_index > 0) {
    auto dest_data = rmm::device_uvector<bitmask_type>(data_words, stream, mr);
    cudf::detail::grid_1d config(data_words, 256);
    copy_shifted_bitmask<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
      dest_data.data(), data.data(), bit_index, bit_index + num_rows, data_words);
    CUDF_CHECK_CUDA(stream.value());
    data = std::move(dest_data);
  }

  auto out_col = mask_to_bools(static_cast<bitmask_type*>(data.data()), 0, num_rows, stream, mr);

  if (!skip_mask) {
    auto [out_mask, null_count] = get_mask_buffer(input, stream, mr);
    out_col->set_null_mask(std::move(*out_mask), null_count);
  }

  return out_col;
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::string_view>(
  ArrowSchemaView const* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  CUDF_EXPECTS(
    input->length + 1 <= static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()),
    "Number of rows exceeds cuDF's maximum supported row count (cudf::size_type).",
    std::overflow_error);

  if (input->length == 0) { return make_empty_column(type_id::STRING); }
  auto [mask, null_count] = !skip_mask
                              ? get_mask_buffer(input, stream, mr)
                              : std::pair{std::make_unique<rmm::device_buffer>(0, stream, mr), 0};
  return string_column_from_arrow_host(schema, input, std::move(mask), null_count, stream, mr);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::dictionary32>(
  ArrowSchemaView const* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  ArrowSchemaView keys_schema_view;
  NANOARROW_THROW_NOT_OK(
    ArrowSchemaViewInit(&keys_schema_view, schema->schema->dictionary, nullptr));

  auto const keys_type = arrow_to_cudf_type(&keys_schema_view);
  auto keys_column =
    get_column_copy(&keys_schema_view, input->dictionary, keys_type, true, stream, mr);

  auto const dict_indices_type = [&schema]() -> data_type {
    // cudf dictionary requires a signed type for the indices
    switch (schema->storage_type) {
      case NANOARROW_TYPE_INT8: return data_type(type_id::INT8);
      case NANOARROW_TYPE_INT16: return data_type(type_id::INT16);
      case NANOARROW_TYPE_INT32: return data_type(type_id::INT32);
      case NANOARROW_TYPE_INT64: return data_type(type_id::INT64);
      default: CUDF_FAIL("Unsupported type_id for dictionary indices", cudf::data_type_error);
    }
  }();

  auto indices_column = get_column_copy(schema, input, dict_indices_type, skip_mask, stream, mr);

  return make_dictionary_column(std::move(keys_column), std::move(indices_column), stream, mr);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::struct_view>(
  ArrowSchemaView const* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  std::vector<std::unique_ptr<column>> child_columns;
  std::transform(input->children,
                 input->children + input->n_children,
                 schema->schema->children,
                 std::back_inserter(child_columns),
                 [this, input](ArrowArray const* child, ArrowSchema const* child_schema) {
                   ArrowSchemaView view;
                   NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
                   auto child_type = arrow_to_cudf_type(&view);

                   ArrowArray child_array(*child);
                   child_array.offset += input->offset;
                   child_array.length = std::min(input->length, child_array.length);

                   return get_column_copy(&view, &child_array, child_type, false, stream, mr);
                 });

  auto [out_mask, null_count] =
    !skip_mask ? get_mask_buffer(input, stream, mr)
               : std::pair{std::make_unique<rmm::device_buffer>(0, stream, mr), 0};

  return make_structs_column(
    input->length, std::move(child_columns), null_count, std::move(*out_mask), stream, mr);
}

template <>
std::unique_ptr<column> dispatch_copy_from_arrow_host::operator()<cudf::list_view>(
  ArrowSchemaView const* schema, ArrowArray const* input, data_type type, bool skip_mask)
{
  CUDF_EXPECTS(
    input->length + 1 <= static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()),
    "Number of rows exceeds cuDF's maximum supported row count (cudf::size_type).",
    std::overflow_error);

  auto [offsets_column, offset, length] = get_offsets_column(schema, input, stream, mr);

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema->schema->children[0], nullptr));
  auto child_type = arrow_to_cudf_type(&view);

  ArrowArray child_array(*input->children[0]);
  child_array.offset += offset;
  child_array.length = std::min(length, child_array.length);

  auto child_column = get_column_copy(&view, &child_array, child_type, skip_mask, stream, mr);

  auto [out_mask, null_count] =
    !skip_mask ? get_mask_buffer(input, stream, mr)
               : std::pair{std::make_unique<rmm::device_buffer>(0, stream, mr), 0};

  return make_lists_column(static_cast<size_type>(input->length),
                           std::move(offsets_column),
                           std::move(child_column),
                           null_count,
                           std::move(*out_mask));
}

/**
 * @brief Convert ArrowArray to cudf column utility
 *
 * This function is simply a convenience wrapper around the dispatch functor with
 * some extra handling to avoid having to reproduce it for all of the nested types.
 * It also allows us to centralize the location where the recursive calls happen
 * so that we only need to forward declare this one function, rather than multiple
 * functions which handle the overloads for nested types (list, struct, etc.)
 *
 * @param schema Arrow schema includes the column type
 * @param input Column data, nulls, offset
 * @param type The cudf column type to map input to
 * @param skip_mask True if the mask is handled by the caller
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for all device memory allocations
 */
std::unique_ptr<column> get_column_copy(ArrowSchemaView const* schema,
                                        ArrowArray const* input,
                                        data_type type,
                                        bool skip_mask,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    input->length <= static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()),
    "Number of rows exceeds cuDF's maximum supported row count (cudf::size_type).",
    std::overflow_error);

  if (type.id() == type_id::EMPTY) {
    return std::make_unique<column>(data_type(type_id::EMPTY),
                                    input->length,
                                    rmm::device_buffer{},
                                    rmm::device_buffer{},
                                    input->length);
  }

  auto result = type_dispatcher(
    type, dispatch_copy_from_arrow_host{stream, mr}, schema, input, type, skip_mask);
  if (!result->has_nulls() && !result->nullable() &&           // no nulls, not nullable
      ((schema->schema->flags & ARROW_FLAG_NULLABLE) != 0)) {  // but arrow wants nullable
    result->set_null_mask(
      cudf::detail::create_null_mask(result->size(), cudf::mask_state::ALL_VALID, stream, mr), 0);
  }
  return result;
}

/**
 * @brief Utility to copy and normalize the offsets in the given array
 */
template <typename OffsetType>
std::tuple<std::unique_ptr<column>, int64_t, int64_t> copy_offsets_column(
  ArrowSchemaView const* schema,
  ArrowArray const* offsets,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto offsets_buffer =
    static_cast<OffsetType const*>(offsets->buffers[fixed_width_data_buffer_idx]);
  auto const offset = offsets_buffer[offsets->offset];
  auto const length = offsets_buffer[offsets->offset + offsets->length - 1] - offset;

  // dispatch directly since we know the type
  auto result = dispatch_copy_from_arrow_host{stream, mr}.template operator()<OffsetType>(
    schema, offsets, data_type{type_to_id<OffsetType>()}, true);
  if (offset != 0) {
    auto begin = result->mutable_view().template begin<OffsetType>();
    auto end   = begin + offsets->length;
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      begin,
                      end,
                      begin,
                      [offset] __device__(auto o) { return o - offset; });
  }
  return std::tuple{std::move(result), offset, length};
}

}  // namespace

/**
 * @brief Utility to copy the offsets from the given input (strings or list) to a
 * cudf column
 */
std::tuple<std::unique_ptr<column>, int64_t, int64_t> get_offsets_column(
  ArrowSchemaView const* schema,
  ArrowArray const* input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  void const* offsets_buffer     = input->buffers[fixed_width_data_buffer_idx];
  void const* offsets_buffers[2] = {nullptr, offsets_buffer};
  ArrowArray offsets_array       = {
          .length     = input->length + 1,
          .null_count = 0,
          .offset     = input->offset,
          .n_buffers  = 2,
          .n_children = 0,
          .buffers    = offsets_buffers,
  };

  if (schema->type == NANOARROW_TYPE_STRING || schema->type == NANOARROW_TYPE_LIST) {
    return copy_offsets_column<int32_t>(schema, &offsets_array, stream, mr);
  }
  if (schema->type == NANOARROW_TYPE_LARGE_STRING) {
    return copy_offsets_column<int64_t>(schema, &offsets_array, stream, mr);
  }

  CUDF_EXPECTS(schema->type == NANOARROW_TYPE_LARGE_LIST, "Unknown offsets parent type");

  // Large-lists must be copied to int32 column
  auto int32_offsets = std::vector<int32_t>();
  int32_offsets.reserve(input->length + 1);
  auto int64_offsets = static_cast<int64_t const*>(offsets_buffer);
  auto const offset  = int64_offsets[input->offset];
  auto const length  = int64_offsets[input->offset + input->length] - offset;

  constexpr auto max_offset = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
  CUDF_EXPECTS(
    length <= max_offset, "large list offsets exceed 32-bit integer bounds", std::overflow_error);

  // normalize the offsets while copying from int64 to int32
  std::transform(int64_offsets + input->offset,
                 int64_offsets + input->offset + input->length + 1,
                 std::back_inserter(int32_offsets),
                 [offset](int64_t o) { return static_cast<int32_t>(o - offset); });

  offsets_buffers[fixed_width_data_buffer_idx] = int32_offsets.data();
  offsets_array.offset                         = 0;  // already accounted for by the above transform
  auto result = dispatch_copy_from_arrow_host{stream, mr}.template operator()<int32_t>(
    schema, &offsets_array, data_type(type_id::INT32), true);
  return std::tuple{std::move(result), offset, length};
}

std::unique_ptr<table> from_arrow_host(ArrowSchema const* schema,
                                       ArrowDeviceArray const* input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL",
               std::invalid_argument);
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CPU,
               "ArrowDeviceArray must have CPU device type for `from_arrow_host`",
               std::invalid_argument);

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));

  std::vector<std::unique_ptr<column>> columns;

  auto type = arrow_to_cudf_type(&view);
  CUDF_EXPECTS(type == data_type(type_id::STRUCT),
               "Must pass a struct to `from_arrow_host`",
               cudf::data_type_error);

  std::transform(input->array.children,
                 input->array.children + input->array.n_children,
                 view.schema->children,
                 std::back_inserter(columns),
                 [&stream, &mr](ArrowArray const* child, ArrowSchema const* child_schema) {
                   ArrowSchemaView view;
                   NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, child_schema, nullptr));
                   auto type = arrow_to_cudf_type(&view);
                   return get_column_copy(&view, child, type, false, stream, mr);
                 });

  // A zero-column struct still has a length, preserve it as the table row count.
  if (columns.empty()) {
    CUDF_EXPECTS(
      input->array.length <= static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max()),
      "Number of rows exceeds cuDF's maximum supported row count (cudf::size_type).",
      std::overflow_error);
    return std::make_unique<table>(std::move(columns), static_cast<size_type>(input->array.length));
  }
  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<column> from_arrow_host_column(ArrowSchema const* schema,
                                               ArrowDeviceArray const* input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(schema != nullptr && input != nullptr,
               "input ArrowSchema and ArrowDeviceArray must not be NULL",
               std::invalid_argument);
  CUDF_EXPECTS(input->device_type == ARROW_DEVICE_CPU,
               "ArrowDeviceArray must have CPU device type for `from_arrow_host_column`",
               std::invalid_argument);

  ArrowSchemaView view;
  NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&view, schema, nullptr));

  auto type = arrow_to_cudf_type(&view);
  return get_column_copy(&view, &input->array, type, false, stream, mr);
}

std::unique_ptr<column> get_column_from_host_copy(ArrowSchemaView const* schema,
                                                  ArrowArray const* input,
                                                  data_type type,
                                                  bool skip_mask,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  return get_column_copy(schema, input, type, skip_mask, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> from_arrow_host(ArrowSchema const* schema,
                                       ArrowDeviceArray const* input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow_host(schema, input, stream, mr);
}

std::unique_ptr<column> from_arrow_host_column(ArrowSchema const* schema,
                                               ArrowDeviceArray const* input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_arrow_host_column(schema, input, stream, mr);
}

std::unique_ptr<table> from_arrow(ArrowSchema const* schema,
                                  ArrowArray const* input,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  ArrowDeviceArray const device_input = {
    .array       = *input,
    .device_id   = -1,
    .device_type = ARROW_DEVICE_CPU,
  };
  return detail::from_arrow_host(schema, &device_input, stream, mr);
}

std::unique_ptr<column> from_arrow_column(ArrowSchema const* schema,
                                          ArrowArray const* input,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  ArrowDeviceArray const device_input = {
    .array       = *input,
    .device_id   = -1,
    .device_type = ARROW_DEVICE_CPU,
  };
  return detail::from_arrow_host_column(schema, &device_input, stream, mr);
}

}  // namespace cudf
