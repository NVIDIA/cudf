/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <functional>
#include <numeric>

namespace cudf {

template <typename A, typename B>
inline constexpr bool layout_compatible = (sizeof(A) == sizeof(B)) && (alignof(A) == alignof(B));

static_assert(
  layout_compatible<detail::column_device_view_base, column_device_view_core>,
  "detail::column_device_view_base and column_device_view_core must be layout-compatible");
static_assert(layout_compatible<column_device_view_core, column_device_view>,
              "column_device_view_core and column_device_view must be layout-compatible");

static_assert(
  layout_compatible<detail::column_device_view_base, mutable_column_device_view_core>,
  "detail::column_device_view_base and mutable_column_device_view_core must be layout-compatible");
static_assert(
  layout_compatible<mutable_column_device_view_core, mutable_column_device_view>,
  "mutable_column_device_view_core and mutable_column_device_view must be layout-compatible");

// Trivially copy all members but the children
column_device_view::column_device_view(column_view source)
  : column_device_view_core{source.type(),
                            source.size(),
                            source.head(),
                            source.null_count(),
                            source.null_mask(),
                            source.offset(),
                            nullptr,
                            source.num_children()}

{
}

// Free device memory allocated for children
void column_device_view::destroy() { delete this; }

namespace {
// helper function for column_device_view::create and mutable_column_device::create methods
template <typename ColumnView, typename ColumnDeviceView>
std::unique_ptr<ColumnDeviceView, std::function<void(ColumnDeviceView*)>>
create_device_view_from_view(ColumnView const& source,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  size_type num_children = source.num_children();
  // First calculate the size of memory needed to hold the child columns. This is done by calling
  // extent() for each of the children.
  auto get_extent = cudf::detail::make_counting_transform_iterator(
    0, [&source](auto i) { return ColumnDeviceView::extent(source.child(i)); });

  // pad the allocation for aligning the first pointer
  auto const descendant_storage_bytes = std::accumulate(
    get_extent, get_extent + num_children, std::size_t{alignof(ColumnDeviceView) - 1});

  // A buffer of CPU memory is allocated to hold the ColumnDeviceView
  // objects. Once filled, the CPU memory is copied to device memory
  // and then set into the d_children member pointer.
  auto staging_buffer = detail::make_pinned_vector_async<char>(descendant_storage_bytes, stream);

  // Each ColumnDeviceView instance may have child objects that
  // require setting some internal device pointers before being copied
  // from CPU to device.
  auto const descendant_storage =
    new rmm::device_uvector<char>(descendant_storage_bytes, stream, mr);

  auto deleter = [descendant_storage](ColumnDeviceView* v) {
    v->destroy();
    delete descendant_storage;
  };

  std::unique_ptr<ColumnDeviceView, decltype(deleter)> result{
    new ColumnDeviceView(source, staging_buffer.data(), descendant_storage->data()), deleter};

  // copy the CPU memory with all the children into device memory
  detail::cuda_memcpy<char>(*descendant_storage, staging_buffer, stream);
  return result;
}

}  // namespace

// Place any child objects in host memory (h_ptr) and use the device
// memory ptr (d_ptr) to set any child object pointers.
column_device_view::column_device_view(column_view source, void* h_ptr, void* d_ptr)
  : column_device_view_core{source.type(),
                            source.size(),
                            source.head(),
                            source.null_count(),
                            source.null_mask(),
                            source.offset(),
                            nullptr,
                            source.num_children()}
{
  _children = detail::child_columns_to_device_array<column_device_view>(
    source.child_begin(), source.child_end(), h_ptr, d_ptr);
}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>
column_device_view::create(column_view source,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  size_type num_children = source.num_children();
  if (num_children == 0) {
    // Can't use make_unique since the ctor is protected
    return std::unique_ptr<column_device_view>(new column_device_view(source));
  }

  return create_device_view_from_view<column_view, column_device_view>(source, stream, mr);
}

std::size_t column_device_view::extent(column_view const& source)
{
  auto get_extent = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, [&source](auto i) { return extent(source.child(i)); });

  return std::accumulate(
    get_extent, get_extent + source.num_children(), sizeof(column_device_view));
}

// For use with inplace-new to pre-fill memory to be copied to device
mutable_column_device_view::mutable_column_device_view(mutable_column_view source)
  : mutable_column_device_view_core{source.type(),
                                    source.size(),
                                    source.head(),
                                    source.null_mask(),
                                    source.offset(),
                                    nullptr,
                                    source.num_children()}
{
}

mutable_column_device_view::mutable_column_device_view(mutable_column_view source,
                                                       void* h_ptr,
                                                       void* d_ptr)
  : mutable_column_device_view_core{source.type(),
                                    source.size(),
                                    source.head(),
                                    source.null_mask(),
                                    source.offset(),
                                    nullptr,
                                    source.num_children()}
{
  _children = detail::child_columns_to_device_array<mutable_column_device_view>(
    source.child_begin(), source.child_end(), h_ptr, d_ptr);
}

// Handle freeing children
void mutable_column_device_view::destroy() { delete this; }

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>
mutable_column_device_view::create(mutable_column_view source,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  return source.num_children() == 0
           ? std::unique_ptr<mutable_column_device_view>(new mutable_column_device_view(source))
           : create_device_view_from_view<mutable_column_view, mutable_column_device_view>(
               source, stream, mr);
}

std::size_t mutable_column_device_view::extent(mutable_column_view source)
{
  auto get_extent = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, [&source](auto i) { return extent(source.child(i)); });

  return std::accumulate(
    get_extent, get_extent + source.num_children(), sizeof(mutable_column_device_view));
}

}  // namespace cudf
