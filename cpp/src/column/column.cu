/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/detail/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/copying.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace cudf {

// Copy ctor w/ optional stream/mr
column::column(column const& other, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  : _type{other._type},
    _size{other._size},
    _data{other._data, stream, mr},
    _null_mask{other._null_mask, stream, mr},
    _null_count{other._null_count}
{
  _children.reserve(other.num_children());
  for (auto const& c : other._children) {
    _children.emplace_back(std::make_unique<column>(*c, stream, mr));
  }
}

// Move constructor
column::column(column&& other) noexcept
  : _type{other._type},
    _size{other._size},
    _data{std::move(other._data)},
    _null_mask{std::move(other._null_mask)},
    _null_count{other._null_count},
    _children{std::move(other._children)}
{
  other._size       = 0;
  other._null_count = 0;
  other._type       = data_type{type_id::EMPTY};
}

// Release contents
column::contents column::release() noexcept
{
  _size       = 0;
  _null_count = 0;
  _type       = data_type{type_id::EMPTY};
  return column::contents{std::make_unique<rmm::device_buffer>(std::move(_data)),
                          std::make_unique<rmm::device_buffer>(std::move(_null_mask)),
                          std::move(_children)};
}

std::size_t column::alloc_size() const
{
  return _data.size() + _null_mask.size() +
         std::transform_reduce(
           _children.begin(), _children.end(), std::size_t{0}, std::plus{}, [](auto const& c) {
             return c->alloc_size();
           });
}

// Create immutable view
column_view column::view() const
{
  // Create views of children
  std::vector<column_view> child_views;
  child_views.reserve(_children.size());
  for (auto const& c : _children) {
    child_views.emplace_back(*c);
  }

  return column_view{type(),
                     size(),
                     _data.data(),
                     static_cast<bitmask_type const*>(_null_mask.data()),
                     null_count(),
                     0,
                     child_views};
}

// Create mutable view
mutable_column_view column::mutable_view()
{
  CUDF_FUNC_RANGE();

  // create views of children
  std::vector<mutable_column_view> child_views;
  child_views.reserve(_children.size());
  for (auto const& c : _children) {
    child_views.emplace_back(*c);
  }

  return mutable_column_view{type(),
                             size(),
                             _data.data(),
                             static_cast<bitmask_type*>(_null_mask.data()),
                             _null_count,
                             0,
                             child_views};
}

void column::set_null_mask(rmm::device_buffer&& new_null_mask, size_type new_null_count)
{
  if (new_null_count > 0) {
    CUDF_EXPECTS(new_null_mask.size() >= cudf::bitmask_allocation_size_bytes(this->size()),
                 "Column with null values must be nullable and the null mask \
                  buffer size should match the size of the column.");
  }
  _null_mask  = std::move(new_null_mask);  // move
  _null_count = new_null_count;
}

void column::set_null_mask(rmm::device_buffer const& new_null_mask,
                           size_type new_null_count,
                           cuda::stream_ref stream)
{
  if (new_null_count > 0) {
    CUDF_EXPECTS(new_null_mask.size() >= cudf::bitmask_allocation_size_bytes(this->size()),
                 "Column with null values must be nullable and the null mask \
                  buffer size should match the size of the column.");
  }
  _null_mask  = rmm::device_buffer{new_null_mask, stream};  // copy
  _null_count = new_null_count;
}

void column::set_null_count(size_type new_null_count)
{
  if (new_null_count > 0) { CUDF_EXPECTS(nullable(), "Invalid null count."); }
  _null_count = new_null_count;
}

namespace {
struct create_column_from_view {
  cudf::column_view view;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  template <typename ColumnType>
  std::unique_ptr<column> operator()()
    requires(std::is_same_v<ColumnType, cudf::string_view>)
  {
    cudf::strings_column_view sview(view);
    return cudf::strings::detail::copy_slice(sview, 0, view.size(), stream, mr);
  }

  template <typename ColumnType>
  std::unique_ptr<column> operator()()
    requires(std::is_same_v<ColumnType, cudf::dictionary32>)
  {
    std::vector<std::unique_ptr<column>> children;
    if (view.num_children()) {
      cudf::dictionary_column_view dict_view(view);
      auto indices_view = column_view(dict_view.indices().type(),
                                      dict_view.size(),
                                      dict_view.indices().head(),
                                      nullptr,
                                      0,
                                      dict_view.offset());
      children.emplace_back(std::make_unique<column>(indices_view, stream, mr));
      children.emplace_back(std::make_unique<column>(dict_view.keys(), stream, mr));
    }
    return std::make_unique<column>(view.type(),
                                    view.size(),
                                    rmm::device_buffer{0, stream, mr},
                                    cudf::detail::copy_bitmask(view, stream, mr),
                                    view.null_count(),
                                    std::move(children));
  }

  template <typename ColumnType>
  std::unique_ptr<column> operator()()
    requires(cudf::is_fixed_width<ColumnType>())
  {
    auto op       = [&](auto const& child) { return std::make_unique<column>(child, stream, mr); };
    auto begin    = thrust::make_transform_iterator(view.child_begin(), op);
    auto children = std::vector<std::unique_ptr<column>>(begin, begin + view.num_children());

    return std::make_unique<column>(
      view.type(),
      view.size(),
      rmm::device_buffer{
        static_cast<char const*>(view.head()) + (view.offset() * cudf::size_of(view.type())),
        view.size() * cudf::size_of(view.type()),
        stream,
        mr},
      cudf::detail::copy_bitmask(view, stream, mr),
      view.null_count(),
      std::move(children));
  }

  template <typename ColumnType>
  std::unique_ptr<column> operator()()
    requires(std::is_same_v<ColumnType, cudf::list_view>)
  {
    auto lists_view = lists_column_view(view);
    return cudf::lists::detail::copy_slice(lists_view, 0, view.size(), stream, mr);
  }

  template <typename ColumnType>
  std::unique_ptr<column> operator()()
    requires(std::is_same_v<ColumnType, cudf::struct_view>)
  {
    if (view.is_empty()) { return cudf::empty_like(view); }

    std::vector<std::unique_ptr<column>> children;
    children.reserve(view.num_children());
    auto begin = view.offset();
    auto end   = begin + view.size();

    std::transform(view.child_begin(),
                   view.child_end(),
                   std::back_inserter(children),
                   [begin, end, stream = this->stream, mr = this->mr](auto child) {
                     return std::make_unique<column>(
                       cudf::detail::slice(child, begin, end, stream), stream, mr);
                   });

    auto num_rows = view.size();

    return make_structs_column(num_rows,
                               std::move(children),
                               view.null_count(),
                               cudf::detail::copy_bitmask(view.null_mask(), begin, end, stream, mr),
                               stream,
                               mr);
  }
};
}  // anonymous namespace

// Copy from a view
column::column(column_view view, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  :  // Move is needed here because the dereference operator of unique_ptr returns
     // an lvalue reference, which would otherwise dispatch to the copy constructor
    column{std::move(*type_dispatcher(view.type(), create_column_from_view{view, stream, mr}))}
{
}

std::unique_ptr<column> rebind_stream(column&& col, cuda::stream_ref stream)
{
  auto const dtype      = col.type();
  auto const sz         = col.size();
  auto const null_count = col.null_count();
  auto contents         = col.release();
  contents.data->set_stream(stream);
  contents.null_mask->set_stream(stream);
  for (auto& child : contents.children) {
    child = rebind_stream(std::move(*child), stream);
  }
  return std::make_unique<column>(dtype,
                                  sz,
                                  std::move(*contents.data),
                                  std::move(*contents.null_mask),
                                  null_count,
                                  std::move(contents.children));
}

}  // namespace cudf
