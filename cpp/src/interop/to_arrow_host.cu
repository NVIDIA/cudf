/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utilities.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>
#include <sys/mman.h>

#include <iostream>

namespace cudf {
namespace detail {

namespace {

/*
  Enable Transparent Huge Pages (THP) for large (>4MB) allocations.
  `buf` is returned untouched.
  Enabling THP can improve performance of device-host memory transfers
  significantly, see <https://github.com/rapidsai/cudf/pull/13914>.
*/
void enable_hugepage(ArrowBuffer* buffer)
{
  if (buffer->size_bytes < (1u << 22u)) {  // Smaller than 4 MB
    return;
  }

#ifdef MADV_HUGEPAGE
  auto const pagesize = sysconf(_SC_PAGESIZE);
  void* addr          = const_cast<uint8_t*>(buffer->data);
  auto length{static_cast<std::size_t>(buffer->size_bytes)};
  if (std::align(pagesize, pagesize, addr, length)) {
    // Intentionally not checking for errors that may be returned by older kernel versions;
    // optimistically tries enabling huge pages.
    madvise(addr, length, MADV_HUGEPAGE);
  }
#endif
}

struct dispatch_to_arrow_host {
  cudf::column_view column;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  int populate_validity_bitmap(ArrowBitmap* bitmap) const
  {
    if (!column.has_nulls()) { return NANOARROW_OK; }

    NANOARROW_RETURN_NOT_OK(ArrowBitmapResize(bitmap, static_cast<int64_t>(column.size()), 0));
    enable_hugepage(&bitmap->buffer);
    CUDF_CUDA_TRY(cudf::detail::memcpy_async(
      bitmap->buffer.data,
      (column.offset() > 0) ? cudf::detail::copy_bitmask(column, stream, mr).data()
                            : column.null_mask(),
      bitmap->buffer.size_bytes,
      stream));
    stream.synchronize();  // ensures the bitmap is not destroyed before the copy is completed
    return NANOARROW_OK;
  }

  template <typename T>
  int populate_data_buffer(device_span<T const> input, ArrowBuffer* buffer) const
  {
    NANOARROW_RETURN_NOT_OK(ArrowBufferResize(buffer, input.size_bytes(), 1));
    enable_hugepage(buffer);
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(buffer->data, input.data(), input.size_bytes(), stream));
    return NANOARROW_OK;
  }

  template <typename T,
            CUDF_ENABLE_IF(!is_rep_layout_compatible<T>() && !cudf::is_fixed_point<T>())>
  int operator()(ArrowArray*) const
  {
    CUDF_FAIL("Unsupported type for to_arrow_host", cudf::data_type_error);
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>() || is_fixed_point<T>())>
  int operator()(ArrowArray* out) const
  {
    nanoarrow::UniqueArray tmp;

    auto const storage_type = id_to_arrow_storage_type(column.type().id());
    NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), storage_type, column));

    NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
    using DataType = device_storage_type_t<T>;
    NANOARROW_RETURN_NOT_OK(
      populate_data_buffer(device_span<DataType const>(column.data<DataType>(), column.size()),
                           ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }
};

int get_column(cudf::column_view column,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr,
               ArrowArray* out);

template <>
int dispatch_to_arrow_host::operator()<bool>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_BOOL, column));

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
  auto bitmask = detail::bools_to_mask(column, stream, mr);
  NANOARROW_RETURN_NOT_OK(populate_data_buffer(
    device_span<uint8_t const>(reinterpret_cast<uint8_t const*>(bitmask.first->data()),
                               bitmask.first->size()),
    ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::string_view>(ArrowArray* out) const
{
  ArrowType nanoarrow_type = NANOARROW_TYPE_STRING;
  if (column.num_children() > 0 &&
      column.child(cudf::strings_column_view::offsets_column_index).type().id() ==
        cudf::type_id::INT64) {
    nanoarrow_type = NANOARROW_TYPE_LARGE_STRING;
  }

  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), nanoarrow_type, column));

  if (column.size() == 0) {
    // initialize the offset buffer with a single zero by convention
    if (nanoarrow_type == NANOARROW_TYPE_LARGE_STRING) {
      NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppendInt64(ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx), 0));
    } else {
      NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppendInt32(ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx), 0));
    }

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));

  auto const scv     = cudf::strings_column_view(column);
  auto const offsets = scv.offsets();
  if (offsets.type().id() == cudf::type_id::INT64) {
    NANOARROW_RETURN_NOT_OK(populate_data_buffer(
      device_span<int64_t const>(offsets.data<int64_t>() + scv.offset(), scv.size() + 1),
      ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
  } else {
    NANOARROW_RETURN_NOT_OK(populate_data_buffer(
      device_span<int32_t const>(offsets.data<int32_t>() + scv.offset(), scv.size() + 1),
      ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
  }

  NANOARROW_RETURN_NOT_OK(
    populate_data_buffer(device_span<char const>(scv.chars_begin(stream), scv.chars_size(stream)),
                         ArrowArrayBuffer(tmp.get(), 2)));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::list_view>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_LIST, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), 1));

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
  auto const lcv = cudf::lists_column_view(column);

  if (column.size() == 0) {
    // initialize the offsets buffer with a single zero by convention for 0 length
    NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt32(ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx), 0));
  } else {
    NANOARROW_RETURN_NOT_OK(
      populate_data_buffer(device_span<int32_t const>(lcv.offsets_begin(), (column.size() + 1)),
                           ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
  }

  NANOARROW_RETURN_NOT_OK(get_column(lcv.child(), stream, mr, tmp->children[0]));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::dictionary32>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;

  auto const dcv          = cudf::dictionary_column_view(column);
  auto const dict_indices = dcv.is_empty() ? cudf::make_empty_column(cudf::type_id::INT32)->view()
                                           : dcv.get_indices_annotated();
  auto const keys =
    dcv.is_empty() ? cudf::make_empty_column(cudf::type_id::INT64)->view() : dcv.keys();

  NANOARROW_RETURN_NOT_OK(
    initialize_array(tmp.get(), id_to_arrow_type(dict_indices.type().id()), column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateDictionary(tmp.get()));

  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));
  switch (dict_indices.type().id()) {
    case type_id::INT8:
    case type_id::UINT8:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int8_t const>(dict_indices.data<int8_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    case type_id::INT16:
    case type_id::UINT16:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int16_t const>(dict_indices.data<int16_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    case type_id::INT32:
    case type_id::UINT32:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int32_t const>(dict_indices.data<int32_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    case type_id::INT64:
    case type_id::UINT64:
      NANOARROW_RETURN_NOT_OK(populate_data_buffer(
        device_span<int64_t const>(dict_indices.data<int64_t>(), dict_indices.size()),
        ArrowArrayBuffer(tmp.get(), fixed_width_data_buffer_idx)));
      break;
    default: CUDF_FAIL("unsupported type for dictionary indices");
  }

  NANOARROW_RETURN_NOT_OK(get_column(keys, stream, mr, tmp->dictionary));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_host::operator()<cudf::struct_view>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;

  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_STRUCT, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), column.num_children()));
  NANOARROW_RETURN_NOT_OK(populate_validity_bitmap(ArrowArrayValidityBitmap(tmp.get())));

  auto const scv = cudf::structs_column_view(column);

  for (size_t i = 0; i < size_t(tmp->n_children); ++i) {
    ArrowArray* child_ptr = tmp->children[i];
    auto const child      = scv.get_sliced_child(i, stream);
    NANOARROW_RETURN_NOT_OK(get_column(child, stream, mr, child_ptr));
  }

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

int get_column(cudf::column_view column,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr,
               ArrowArray* out)
{
  return column.type().id() != type_id::EMPTY
           ? type_dispatcher(column.type(), dispatch_to_arrow_host{column, stream, mr}, out)
           : initialize_array(out, NANOARROW_TYPE_NA, column);
}

unique_device_array_t create_device_array(nanoarrow::UniqueArray&& out)
{
  ArrowError err;
  if (ArrowArrayFinishBuildingDefault(out.get(), &err) != NANOARROW_OK) {
    std::cerr << err.message << std::endl;
    CUDF_FAIL("failed to build");
  }

  unique_device_array_t result(new ArrowDeviceArray, [](ArrowDeviceArray* arr) {
    if (arr->array.release != nullptr) { ArrowArrayRelease(&arr->array); }
    delete arr;
  });

  result->device_id   = -1;
  result->device_type = ARROW_DEVICE_CPU;
  result->sync_event  = nullptr;
  ArrowArrayMove(out.get(), &result->array);
  return result;
}

}  // namespace

unique_device_array_t to_arrow_host(cudf::table_view const& table,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_STRUCT));

  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), table.num_columns()));
  tmp->length     = table.num_rows();
  tmp->null_count = 0;

  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto child = tmp->children[i];
    auto col   = table.column(i);
    NANOARROW_THROW_NOT_OK(
      cudf::type_dispatcher(col.type(), detail::dispatch_to_arrow_host{col, stream, mr}, child));
  }

  // wait for all the stream operations to complete before we return.
  // this ensures that the host memory that we're returning will be populated
  // before we return from this function.
  stream.synchronize();

  return create_device_array(std::move(tmp));
}

unique_device_array_t to_arrow_host(cudf::column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray tmp;

  NANOARROW_THROW_NOT_OK(
    cudf::type_dispatcher(col.type(), detail::dispatch_to_arrow_host{col, stream, mr}, tmp.get()));

  // wait for all the stream operations to complete before we return.
  // this ensures that the host memory that we're returning will be populated
  // before we return from this function.
  stream.synchronize();

  return create_device_array(std::move(tmp));
}

/**
 * @brief Builds the ArrayBinaryView for each row in the d_strings column
 *
 * Smaller strings may fit inline in the ArrayBinaryItem object while
 * longer strings specify a buffer and offset for their character data.
 */
struct strings_to_binary_view {
  cudf::column_device_view d_strings;
  input_offsetalator d_offsets;               // offsets of longer strings in d_strings
  device_span<int64_t const> buffer_offsets;  // output buffers' offsets
  ArrowBinaryView* d_items;

  __device__ void operator()(cudf::size_type idx) const
  {
    auto& item = d_items[idx];
    if (d_strings.is_null(idx)) {
      item.inlined.size = 0;
      return;
    }
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    item.inlined.size = d_str.size_bytes();
    if (d_str.size_bytes() <= NANOARROW_BINARY_VIEW_INLINE_SIZE) {
      thrust::copy(thrust::seq, d_str.data(), d_str.data() + d_str.size_bytes(), item.inlined.data);
      thrust::uninitialized_fill(thrust::seq,
                                 item.inlined.data + item.inlined.size,
                                 item.inlined.data + NANOARROW_BINARY_VIEW_INLINE_SIZE,
                                 0);
    } else {
      thrust::copy(thrust::seq,
                   d_str.data(),
                   d_str.data() + NANOARROW_BINARY_VIEW_PREFIX_SIZE,
                   item.ref.prefix);
      auto const offset  = d_offsets[idx];
      auto const buf_idx = cuda::std::distance(
        buffer_offsets.begin(),
        thrust::upper_bound(thrust::seq, buffer_offsets.begin(), buffer_offsets.end(), offset));
      auto const new_offset = offset - (buf_idx == 0 ? 0 : buffer_offsets[buf_idx - 1]);
      item.ref.buffer_index = buf_idx;
      item.ref.offset       = static_cast<int32_t>(new_offset);
    }
  }
};

unique_device_array_t to_arrow_host_stringview(cudf::strings_column_view const& col,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray out;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(out.get(), NANOARROW_TYPE_STRING_VIEW));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(out.get()));

  if (col.size() == 0) { return create_device_array(std::move(out)); }

  dispatch_to_arrow_host fn{col.parent(), stream, mr};
  NANOARROW_THROW_NOT_OK(fn.populate_validity_bitmap(ArrowArrayValidityBitmap(out.get())));

  auto const d_strings = column_device_view::create(col.parent(), stream);
  auto d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(col.offsets(), col.offset());

  // count the number of long-ish strings -- ones that cannot be inlined
  auto const num_longer_strings = thrust::count_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{col.size()},
    [d_offsets] __device__(auto idx) {
      return d_offsets[idx + 1] - d_offsets[idx] > NANOARROW_BINARY_VIEW_INLINE_SIZE;
    });

  // gather all the long-ish strings into a single strings column
  auto [unused_col, longer_strings] = [&] {
    if (num_longer_strings == col.size()) {
      // we can use the input column as is for the remainder of this function
      return std::pair{cudf::make_empty_column(cudf::type_id::STRING), col};
    }
    auto indices = make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<cudf::strings::detail::string_index_pair>(
        [d_strings = *d_strings] __device__(auto idx) {
          if (d_strings.is_null(idx)) {
            return cudf::strings::detail::string_index_pair{nullptr, 0};
          }
          auto const d_str = d_strings.element<cudf::string_view>(idx);
          return (d_str.size_bytes() > NANOARROW_BINARY_VIEW_INLINE_SIZE)
                   ? cudf::strings::detail::string_index_pair{d_str.data(), d_str.size_bytes()}
                   : cudf::strings::detail::string_index_pair{"", 0};
        }));
    auto longer_strings = cudf::strings::detail::make_strings_column(
      indices, indices + col.size(), stream, cudf::get_current_device_resource_ref());
    stream.synchronize();
    auto const sv = cudf::strings_column_view(longer_strings->view());
    return std::pair{std::move(longer_strings), sv};
  }();
  auto [first, last] = cudf::strings::detail::get_first_and_last_offset(longer_strings, stream);
  auto const longer_chars_size = last - first;

  d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(longer_strings.offsets(),
                                                                      longer_strings.offset());

  // using max/2 here ensures no buffer is greater than 2GB
  constexpr int64_t max_size = std::numeric_limits<int32_t>::max() / 2;
  auto const num_buffers     = cudf::util::div_rounding_up_safe(longer_chars_size, max_size);
  auto buffer_offsets        = rmm::device_uvector<int64_t>(num_buffers, stream);
  // copy the bytes for the longer strings into Arrow variadic buffers
  if (longer_chars_size > 0) {
    // compute buffer boundaries (less than 2GB per buffer)
    auto buffer_indices  = rmm::device_uvector<int64_t>(num_buffers, stream);
    auto const bound_itr = make_counting_transform_iterator(
      0, cuda::proclaim_return_type<int64_t>([] __device__(auto idx) {
        return (idx + 1) * max_size;
      }));
    thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        d_offsets,
                        d_offsets + longer_strings.size(),
                        bound_itr,
                        bound_itr + num_buffers,
                        buffer_indices.begin());
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      buffer_indices.begin(),
                      buffer_indices.end(),
                      buffer_offsets.begin(),
                      [d_offsets] __device__(auto idx) { return d_offsets[idx]; });
    auto h_offsets = make_std_vector(buffer_offsets, stream);

    // build up the variadic buffers needed
    NANOARROW_THROW_NOT_OK(ArrowArrayAddVariadicBuffers(out.get(), num_buffers));
    auto const chars_data = longer_strings.chars_begin(stream);
    for (auto i = 0L; i < num_buffers; ++i) {
      auto variadic_buf = ArrowArrayBuffer(out.get(), NANOARROW_BINARY_VIEW_FIXED_BUFFERS + i);
      auto const offset = i == 0 ? 0 : h_offsets[i - 1];
      auto const size   = h_offsets[i] - offset;
      NANOARROW_THROW_NOT_OK(ArrowBufferReserve(variadic_buf, size));
      CUDF_CUDA_TRY(
        cudf::detail::memcpy_async(variadic_buf->data, chars_data + offset, size, stream));
      variadic_buf->size_bytes = size;
    }
  }

  // now build BinaryView objects from the strings in device memory
  auto d_items = rmm::device_uvector<ArrowBinaryView>(col.size(), stream);
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<cudf::size_type>{0},
                     col.size(),
                     strings_to_binary_view{*d_strings, d_offsets, buffer_offsets, d_items.data()});

  constexpr auto data_buffer_idx = 1;

  // finally, copy the BinaryView array into host memory
  auto data_buffer   = ArrowArrayBuffer(out.get(), data_buffer_idx);
  auto const bv_size = d_items.size() * sizeof(ArrowBinaryView);
  NANOARROW_THROW_NOT_OK(ArrowBufferReserve(data_buffer, bv_size));
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(data_buffer->data, d_items.data(), bv_size, stream));
  data_buffer->size_bytes = bv_size;

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(out.get(), NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  out->length     = col.size();
  out->null_count = col.null_count();
  out->offset     = 0;

  stream.synchronize();
  return create_device_array(std::move(out));
}
}  // namespace detail

unique_device_array_t to_arrow_host(cudf::column_view const& col,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_host(col, stream, mr);
}

unique_device_array_t to_arrow_host(cudf::table_view const& table,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_host(table, stream, mr);
}

unique_device_array_t to_arrow_host_stringview(cudf::strings_column_view const& col,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_host_stringview(col, stream, mr);
}
}  // namespace cudf
