/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utilities.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/for_each.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

namespace cudf {
namespace detail {
namespace {

template <typename T>
void device_buffer_finalize(ArrowBufferAllocator* allocator, uint8_t*, int64_t)
{
  auto* unique_buffer = reinterpret_cast<std::unique_ptr<T>*>(allocator->private_data);
  delete unique_buffer;
}

template <typename>
struct is_device_scalar : public std::false_type {};

template <typename T>
struct is_device_scalar<cudf::detail::device_scalar<T>> : public std::true_type {};

template <typename>
struct is_device_uvector : public std::false_type {};

template <typename T>
struct is_device_uvector<rmm::device_uvector<T>> : public std::true_type {};

template <typename T>
int set_buffer(std::unique_ptr<T> device_buf, int64_t i, ArrowArray* out)
{
  ArrowBuffer* buf = ArrowArrayBuffer(out, i);
  auto ptr         = reinterpret_cast<uint8_t*>(device_buf->data());
  buf->size_bytes  = [&] {
    if constexpr (is_device_scalar<T>::value) {
      return sizeof(typename T::value_type);
    } else if constexpr (is_device_uvector<T>::value) {
      return sizeof(typename T::value_type) * device_buf->size();
    } else {
      return device_buf->size();
    }
  }();
  // we make a new unique_ptr and move to it in case there was a custom deleter
  NANOARROW_RETURN_NOT_OK(
    ArrowBufferSetAllocator(buf,
                            ArrowBufferDeallocator(&device_buffer_finalize<T>,
                                                   new std::unique_ptr<T>(std::move(device_buf)))));
  buf->data = ptr;
  return NANOARROW_OK;
}

int set_null_mask(column::contents& contents, ArrowArray* out)
{
  if (contents.null_mask) {
    NANOARROW_RETURN_NOT_OK(set_buffer(std::move(contents.null_mask), validity_buffer_idx, out));
  }
  return NANOARROW_OK;
}

int set_contents(column::contents& contents, ArrowArray* out)
{
  NANOARROW_RETURN_NOT_OK(set_null_mask(contents, out));
  NANOARROW_RETURN_NOT_OK(set_buffer(std::move(contents.data), fixed_width_data_buffer_idx, out));
  return NANOARROW_OK;
}

struct dispatch_to_arrow_device {
  template <typename T,
            CUDF_ENABLE_IF(not is_rep_layout_compatible<T>() and not is_fixed_point<T>())>
  int operator()(cudf::column&&, cuda::stream_ref, rmm::device_async_resource_ref, ArrowArray*)
  {
    CUDF_FAIL("Unsupported type for to_arrow_device", cudf::data_type_error);
  }

  // cover rep layout compatible and decimal types
  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>() or is_fixed_point<T>())>
  int operator()(cudf::column&& column,
                 cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr,
                 ArrowArray* out)
  {
    nanoarrow::UniqueArray tmp;

    auto const storage_type = id_to_arrow_storage_type(column.type().id());
    NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), storage_type, column));

    auto contents = column.release();
    NANOARROW_RETURN_NOT_OK(set_contents(contents, tmp.get()));

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }
};

int handle_empty_type_column(ArrowArray* array, cudf::column& column)
{
  NANOARROW_RETURN_NOT_OK(initialize_array(array, NANOARROW_TYPE_NA, column.view()));
  auto child_contents = column.release();
  NANOARROW_RETURN_NOT_OK(set_contents(child_contents, array));
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device::operator()<bool>(cudf::column&& column,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr,
                                               ArrowArray* out)
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_BOOL, column));

  auto bitmask  = detail::bools_to_mask(column.view(), stream, mr);
  auto contents = column.release();
  NANOARROW_RETURN_NOT_OK(set_null_mask(contents, tmp.get()));
  NANOARROW_RETURN_NOT_OK(
    set_buffer(std::move(bitmask.first), fixed_width_data_buffer_idx, tmp.get()));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device::operator()<cudf::string_view>(cudf::column&& column,
                                                            cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr,
                                                            ArrowArray* out)
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
    // the scalar zero here is necessary because the spec for string arrays states
    // that the offsets buffer should contain "length + 1" signed integers. So in
    // the case of a 0 length string array, there should be exactly 1 value, zero,
    // in the offsets buffer. While some arrow implementations may accept a zero-sized
    // offsets buffer, best practices would be to allocate the buffer with the single value.
    if (nanoarrow_type == NANOARROW_TYPE_STRING) {
      auto zero = std::make_unique<cudf::detail::device_scalar<int32_t>>(0, stream, mr);
      NANOARROW_RETURN_NOT_OK(set_buffer(std::move(zero), fixed_width_data_buffer_idx, tmp.get()));
    } else {
      auto zero = std::make_unique<cudf::detail::device_scalar<int64_t>>(0, stream, mr);
      NANOARROW_RETURN_NOT_OK(set_buffer(std::move(zero), fixed_width_data_buffer_idx, tmp.get()));
    }

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }

  auto contents = column.release();
  NANOARROW_RETURN_NOT_OK(set_null_mask(contents, tmp.get()));

  auto offsets_contents =
    contents.children[cudf::strings_column_view::offsets_column_index]->release();
  NANOARROW_RETURN_NOT_OK(set_buffer(std::move(offsets_contents.data), 1, tmp.get()));
  NANOARROW_RETURN_NOT_OK(set_buffer(std::move(contents.data), 2, tmp.get()));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device::operator()<cudf::list_view>(cudf::column&& column,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr,
                                                          ArrowArray* out);

template <>
int dispatch_to_arrow_device::operator()<cudf::dictionary32>(cudf::column&& column,
                                                             cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr,
                                                             ArrowArray* out);

template <>
int dispatch_to_arrow_device::operator()<cudf::struct_view>(cudf::column&& column,
                                                            cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr,
                                                            ArrowArray* out)
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_STRUCT, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), column.num_children()));

  auto contents = column.release();
  NANOARROW_RETURN_NOT_OK(set_null_mask(contents, tmp.get()));

  for (size_t i = 0; i < size_t(tmp->n_children); ++i) {
    ArrowArray* child_ptr = tmp->children[i];
    auto& child           = contents.children[i];
    if (child->type().id() == cudf::type_id::EMPTY) {
      NANOARROW_RETURN_NOT_OK(handle_empty_type_column(child_ptr, *child));
    } else {
      NANOARROW_RETURN_NOT_OK(cudf::type_dispatcher(
        child->type(), dispatch_to_arrow_device{}, std::move(*child), stream, mr, child_ptr));
    }
  }

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device::operator()<cudf::list_view>(cudf::column&& column,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr,
                                                          ArrowArray* out)
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_LIST, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), 1));

  auto contents = column.release();
  NANOARROW_RETURN_NOT_OK(set_null_mask(contents, tmp.get()));

  auto offsets_contents =
    contents.children[cudf::lists_column_view::offsets_column_index]->release();
  NANOARROW_RETURN_NOT_OK(set_buffer(std::move(offsets_contents.data), 1, tmp.get()));

  auto& child = contents.children[cudf::lists_column_view::child_column_index];
  if (child->type().id() == cudf::type_id::EMPTY) {
    NANOARROW_RETURN_NOT_OK(handle_empty_type_column(tmp->children[0], *child));
  } else {
    NANOARROW_RETURN_NOT_OK(cudf::type_dispatcher(
      child->type(), dispatch_to_arrow_device{}, std::move(*child), stream, mr, tmp->children[0]));
  }

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device::operator()<cudf::dictionary32>(cudf::column&& column,
                                                             cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr,
                                                             ArrowArray* out)
{
  nanoarrow::UniqueArray tmp;

  cudf::dictionary_column_view const dview{column.view()};
  auto const is_empty = dview.is_empty();

  NANOARROW_RETURN_NOT_OK(initialize_array(
    tmp.get(),
    is_empty ? NANOARROW_TYPE_INT32 : id_to_arrow_type(dview.indices().type().id()),
    column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateDictionary(tmp.get()));

  auto contents = column.release();
  NANOARROW_RETURN_NOT_OK(set_null_mask(contents, tmp.get()));

  auto indices_contents =
    is_empty ? cudf::make_empty_column(cudf::type_id::INT32)->release()
             : contents.children[cudf::dictionary_column_view::indices_column_index]->release();
  NANOARROW_RETURN_NOT_OK(
    set_buffer(std::move(indices_contents.data), fixed_width_data_buffer_idx, tmp.get()));

  auto keys = is_empty
                ? cudf::make_empty_column(cudf::type_id::INT64)
                : std::move(contents.children[cudf::dictionary_column_view::keys_column_index]);
  NANOARROW_RETURN_NOT_OK(cudf::type_dispatcher(
    keys->type(), dispatch_to_arrow_device{}, std::move(*keys), stream, mr, tmp->dictionary));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

struct dispatch_to_arrow_device_view {
  cudf::column_view column;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  template <typename T,
            CUDF_ENABLE_IF(not is_rep_layout_compatible<T>() and not is_fixed_point<T>())>
  int operator()(ArrowArray*) const
  {
    CUDF_FAIL("Unsupported type for to_arrow_device", cudf::data_type_error);
  }

  template <typename T, CUDF_ENABLE_IF(is_rep_layout_compatible<T>() or is_fixed_point<T>())>
  int operator()(ArrowArray* out) const
  {
    nanoarrow::UniqueArray tmp;

    auto const storage_type = id_to_arrow_storage_type(column.type().id());
    NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), storage_type, column));
    NANOARROW_RETURN_NOT_OK(set_null_mask(column, tmp.get()));
    NANOARROW_RETURN_NOT_OK(set_view_to_buffer(column, tmp.get()));

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }

  int set_buffer_view(void const* in_ptr, size_t size, int64_t i, ArrowArray* out) const
  {
    ArrowBuffer* buf = ArrowArrayBuffer(out, i);
    buf->size_bytes  = size;

    // reset the deallocator to do nothing since this is a non-owning view
    NANOARROW_RETURN_NOT_OK(ArrowBufferSetAllocator(
      buf, ArrowBufferDeallocator([](ArrowBufferAllocator*, uint8_t*, int64_t) {}, nullptr)));

    buf->data = const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(in_ptr));
    return NANOARROW_OK;
  }

  int set_null_mask(column_view column, ArrowArray* out) const
  {
    if (column.nullable()) {
      NANOARROW_RETURN_NOT_OK(set_buffer_view(column.null_mask(),
                                              bitmask_allocation_size_bytes(column.size()),
                                              validity_buffer_idx,
                                              out));
    }
    return NANOARROW_OK;
  }

  int set_view_to_buffer(column_view column, ArrowArray* out) const
  {
    auto const type_size = cudf::size_of(column.type());
    return set_buffer_view(column.head<uint8_t>() + (type_size * column.offset()),
                           column.size() * type_size,
                           fixed_width_data_buffer_idx,
                           out);
  }
};

template <>
int dispatch_to_arrow_device_view::operator()<bool>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_BOOL, column));

  auto bitmask = detail::bools_to_mask(column, stream, mr);
  NANOARROW_RETURN_NOT_OK(
    set_buffer(std::move(bitmask.first), fixed_width_data_buffer_idx, tmp.get()));
  NANOARROW_RETURN_NOT_OK(set_null_mask(column, tmp.get()));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device_view::operator()<cudf::string_view>(ArrowArray* out) const
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
    // https://github.com/NVIDIA/cudf/pull/15047#discussion_r1546528552
    if (nanoarrow_type == NANOARROW_TYPE_LARGE_STRING) {
      auto zero = std::make_unique<cudf::detail::device_scalar<int64_t>>(0, stream, mr);
      NANOARROW_RETURN_NOT_OK(set_buffer(std::move(zero), fixed_width_data_buffer_idx, tmp.get()));
    } else {
      auto zero = std::make_unique<cudf::detail::device_scalar<int32_t>>(0, stream, mr);
      NANOARROW_RETURN_NOT_OK(set_buffer(std::move(zero), fixed_width_data_buffer_idx, tmp.get()));
    }

    ArrowArrayMove(tmp.get(), out);
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(set_null_mask(column, tmp.get()));

  auto const scv = cudf::strings_column_view(column);
  NANOARROW_RETURN_NOT_OK(set_view_to_buffer(scv.offsets(), tmp.get()));
  NANOARROW_RETURN_NOT_OK(
    set_buffer_view(scv.chars_begin(stream), scv.chars_size(stream), 2, tmp.get()));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device_view::operator()<cudf::list_view>(ArrowArray* out) const;

template <>
int dispatch_to_arrow_device_view::operator()<cudf::dictionary32>(ArrowArray* out) const;

template <>
int dispatch_to_arrow_device_view::operator()<cudf::struct_view>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;

  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_STRUCT, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), column.num_children()));
  NANOARROW_RETURN_NOT_OK(set_null_mask(column, tmp.get()));

  for (size_t i = 0; i < size_t(tmp->n_children); ++i) {
    ArrowArray* child_ptr = tmp->children[i];
    auto const child      = column.child(i);
    NANOARROW_RETURN_NOT_OK(cudf::type_dispatcher(
      child.type(), dispatch_to_arrow_device_view{child, stream, mr}, child_ptr));
  }

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device_view::operator()<cudf::list_view>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;

  NANOARROW_RETURN_NOT_OK(initialize_array(tmp.get(), NANOARROW_TYPE_LIST, column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), 1));
  NANOARROW_RETURN_NOT_OK(set_null_mask(column, tmp.get()));

  auto const lcv = cudf::lists_column_view(column);
  NANOARROW_RETURN_NOT_OK(set_view_to_buffer(lcv.offsets(), tmp.get()));

  auto child = lcv.child();
  NANOARROW_RETURN_NOT_OK(cudf::type_dispatcher(
    child.type(), dispatch_to_arrow_device_view{child, stream, mr}, tmp->children[0]));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

template <>
int dispatch_to_arrow_device_view::operator()<cudf::dictionary32>(ArrowArray* out) const
{
  nanoarrow::UniqueArray tmp;

  auto const dcv     = cudf::dictionary_column_view(column);
  auto const indices = dcv.is_empty() ? cudf::make_empty_column(cudf::type_id::INT32)->view()
                                      : dcv.get_indices_annotated();
  auto const keys =
    dcv.is_empty() ? cudf::make_empty_column(cudf::type_id::INT64)->view() : dcv.keys();

  NANOARROW_RETURN_NOT_OK(
    initialize_array(tmp.get(), id_to_arrow_type(indices.type().id()), column));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateDictionary(tmp.get()));
  NANOARROW_RETURN_NOT_OK(set_null_mask(column, tmp.get()));

  NANOARROW_RETURN_NOT_OK(set_view_to_buffer(indices, tmp.get()));

  NANOARROW_RETURN_NOT_OK(cudf::type_dispatcher(
    keys.type(), dispatch_to_arrow_device_view{keys, stream, mr}, tmp->dictionary));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

struct ArrowDeviceArrayPrivateData {
  ArrowArray parent;
  cudaEvent_t sync_event;
};

void ArrowDeviceArrayRelease(ArrowArray* array)
{
  auto private_data = reinterpret_cast<ArrowDeviceArrayPrivateData*>(array->private_data);
  CUDF_CUDA_TRY(cudaEventDestroy(private_data->sync_event));
  ArrowArrayRelease(&private_data->parent);
  delete private_data;
  array->release = nullptr;
}

unique_device_array_t create_device_array(nanoarrow::UniqueArray&& out, cuda::stream_ref stream)
{
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(out.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  auto private_data = std::make_unique<detail::ArrowDeviceArrayPrivateData>();
  CUDF_CUDA_TRY(cudaEventCreate(&private_data->sync_event));
  CUDF_CUDA_TRY(cudaEventRecord(private_data->sync_event, stream.get()));

  ArrowArrayMove(out.get(), &private_data->parent);
  unique_device_array_t result(new ArrowDeviceArray, [](ArrowDeviceArray* arr) {
    if (arr->array.release != nullptr) { ArrowArrayRelease(&arr->array); }
    delete arr;
  });
  result->device_id          = rmm::get_current_cuda_device().value();
  result->device_type        = ARROW_DEVICE_CUDA;
  result->sync_event         = &private_data->sync_event;
  result->array              = private_data->parent;  // makes a shallow copy
  result->array.private_data = private_data.release();
  result->array.release      = &detail::ArrowDeviceArrayRelease;
  return result;
}

}  // namespace

unique_device_array_t to_arrow_device(cudf::table&& table,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_STRUCT));

  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), table.num_columns()));
  tmp->length     = table.num_rows();
  tmp->null_count = 0;

  auto cols = table.release();
  for (size_t i = 0; i < cols.size(); ++i) {
    auto child = tmp->children[i];
    auto col   = cols[i].get();
    if (col->type().id() == cudf::type_id::EMPTY) {
      NANOARROW_THROW_NOT_OK(handle_empty_type_column(child, *col));
    } else {
      NANOARROW_THROW_NOT_OK(cudf::type_dispatcher(
        col->type(), detail::dispatch_to_arrow_device{}, std::move(*col), stream, mr, child));
    }
  }

  return create_device_array(std::move(tmp), stream);
}

unique_device_array_t to_arrow_device(cudf::column&& col,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray tmp;

  if (col.type().id() == cudf::type_id::EMPTY) {
    NANOARROW_THROW_NOT_OK(handle_empty_type_column(tmp.get(), col));
  } else {
    NANOARROW_THROW_NOT_OK(cudf::type_dispatcher(
      col.type(), detail::dispatch_to_arrow_device{}, std::move(col), stream, mr, tmp.get()));
  }

  return create_device_array(std::move(tmp), stream);
}

unique_device_array_t to_arrow_device(cudf::table_view const& table,
                                      cuda::stream_ref stream,
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
    NANOARROW_THROW_NOT_OK(cudf::type_dispatcher(
      col.type(), detail::dispatch_to_arrow_device_view{col, stream, mr}, child));
  }

  return create_device_array(std::move(tmp), stream);
}

unique_device_array_t to_arrow_device(cudf::column_view const& col,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray tmp;

  NANOARROW_THROW_NOT_OK(cudf::type_dispatcher(
    col.type(), detail::dispatch_to_arrow_device_view{col, stream, mr}, tmp.get()));

  return create_device_array(std::move(tmp), stream);
}

}  // namespace detail

unique_device_array_t to_arrow_device(cudf::table&& table,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_device(std::move(table), stream, mr);
}

unique_device_array_t to_arrow_device(cudf::column&& col,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_device(std::move(col), stream, mr);
}

unique_device_array_t to_arrow_device(cudf::table_view const& table,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_device(table, stream, mr);
}

unique_device_array_t to_arrow_device(cudf::column_view const& col,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_arrow_device(col, stream, mr);
}
}  // namespace cudf
