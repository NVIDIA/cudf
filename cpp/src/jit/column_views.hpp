/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/buffer>
#include <cuda/stream>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <variant>

namespace cudf::jit {

struct mutable_fixed_width_column_view {
  mutable_column_view _view;

  auto to_device(cuda::stream_ref stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct fixed_width_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(data_type type,
                   size_type size,
                   cuda::device_buffer<std::byte> null_mask,
                   size_type null_count,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
  {
    return fixed_width_column{
      make_fixed_width_column(type, size, std::move(null_mask), null_count, stream, mr)};
  }

  auto mutable_view() const { return mutable_fixed_width_column_view{_col->mutable_view()}; }

  void set_null_count(size_type count) { _col->set_null_count(count); }

  bool nullable() const { return _col->nullable(); }

  bitmask_type* null_mask() { return _col->mutable_view().null_mask(); }
};

struct mutable_string_views_column_view {
  void* _data{nullptr};
  size_type _size{0};
  bitmask_type const* _null_mask{nullptr};
  size_type _offset{0};
  size_type _null_count{0};

  auto to_device(cuda::stream_ref stream) const
  {
    using view = mutable_column_device_view;
    return std::unique_ptr<view, std::function<void(view*)>>(
      new view{
        view::create(data_type{type_id::EMPTY}, _size, _data, _null_mask, _offset, nullptr, 0)},
      [](auto* p) { delete p; });
  }
};

struct string_views_column {
  cuda::device_buffer<string_view> _data;
  size_type _size{0};
  cuda::device_buffer<std::byte> _null_mask =
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);
  size_type _null_count{0};

  static auto make(size_type size,
                   cuda::device_buffer<std::byte> null_mask,
                   size_type null_count,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
  {
    cuda::device_buffer<string_view> data{stream, mr, static_cast<size_t>(size), cuda::no_init};
    return string_views_column{std::move(data), size, std::move(null_mask), null_count};
  }

  auto mutable_view()
  {
    return mutable_string_views_column_view{
      _data.data(),
      _size,
      reinterpret_cast<bitmask_type const*>(_null_mask.data()),
      0,
      _null_count};
  }

  void set_null_count(size_type count) { _null_count = count; }

  bool nullable() const { return _null_mask.size() != 0; }

  bitmask_type* null_mask() { return reinterpret_cast<bitmask_type*>(_null_mask.data()); }
};

struct mutable_strings_column_view {
  mutable_column_view _view;

  auto to_device(cuda::stream_ref stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct mutable_strings_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(size_type size,
                   rmm::device_buffer chars,
                   std::unique_ptr<column> offsets,
                   cuda::device_buffer<std::byte> null_mask,
                   size_type null_count)
  {
    return mutable_strings_column{make_strings_column(
      size, std::move(offsets), std::move(chars), null_count, std::move(null_mask))};
  }

  auto mutable_view() const { return mutable_strings_column_view{_col->mutable_view()}; }

  void set_null_count(size_type count) { _col->set_null_count(count); }

  bool nullable() const { return _col->nullable(); }

  bitmask_type* null_mask() { return _col->mutable_view().null_mask(); }
};

struct mutable_lists_column_view {
  mutable_column_view _view;

  auto to_device(cuda::stream_ref stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

// Lists retain the supplied offsets and own the directly written child.
struct mutable_lists_column {
  std::unique_ptr<column> _col;

  auto mutable_view() const { return mutable_lists_column_view{_col->mutable_view()}; }

  void set_null_count(size_type count) { _col->set_null_count(count); }

  bool nullable() const { return _col->nullable(); }

  bitmask_type* null_mask() { return _col->mutable_view().null_mask(); }
};

using output_column = std::
  variant<fixed_width_column, string_views_column, mutable_strings_column, mutable_lists_column>;
}  // namespace cudf::jit
