/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/errc.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>
#include <cuda/std/optional>

#include <cudf_fragments.hpp>
#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/nvvm_types.hpp>
#include <jit/parser.hpp>
#include <jit/row_ir.hpp>
#include <jit/span.cuh>
#include <jit/util.hpp>

#include <algorithm>
#include <numeric>
#include <span>
#include <type_traits>
#include <variant>

namespace cudf {
namespace jit {

[[maybe_unused]] static column_view as_column_view(scalar_column_view const& scalar)
{
  return scalar.as_column_view();
}

[[maybe_unused]] static column_view as_column_view(column_view const& column) { return column; }

struct mutable_fixed_width_column_view {
  mutable_column_view _view;

  auto to_device(rmm::cuda_stream_view stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct fixed_width_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(data_type type,
                   size_type size,
                   rmm::device_buffer null_mask,
                   size_type null_count,
                   rmm::cuda_stream_view stream,
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

  auto to_device(rmm::cuda_stream_view stream) const
  {
    using view = mutable_column_device_view;
    return std::unique_ptr<view, std::function<void(view*)>>(
      new view{
        view::create(data_type{type_id::EMPTY}, _size, _data, _null_mask, _offset, nullptr, 0)},
      [](auto* p) { delete p; });
  }
};

struct string_views_column {
  rmm::device_buffer _data;
  size_type _size{0};
  rmm::device_buffer _null_mask{};
  size_type _null_count{0};

  static auto make(size_type size,
                   rmm::device_buffer null_mask,
                   size_type null_count,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
  {
    rmm::device_buffer data{static_cast<size_t>(size) * sizeof(string_view), stream, mr};
    return string_views_column{std::move(data), size, std::move(null_mask), null_count};
  }

  auto mutable_view() const
  {
    return mutable_string_views_column_view{
      const_cast<void*>(_data.data()),
      _size,
      static_cast<bitmask_type*>(const_cast<void*>(_null_mask.data())),
      0,
      _null_count};
  }

  void set_null_count(size_type count) { _null_count = count; }

  bool nullable() const { return !_null_mask.is_empty(); }

  bitmask_type* null_mask() { return static_cast<bitmask_type*>(_null_mask.data()); }
};

struct mutable_strings_column_view {
  mutable_column_view _view;

  auto to_device(rmm::cuda_stream_view stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct mutable_strings_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(size_type size,
                   rmm::device_buffer chars,
                   std::unique_ptr<column> offsets,
                   rmm::device_buffer null_mask,
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

[[maybe_unused]] static std::string get_element_type_name(column_view const& view,
                                                          bool use_physical_type);

struct element_type_name_fn {
  template <typename T>
  std::string operator()(column_view const& view, bool use_physical_type) const
    requires(is_fixed_width<T>() || std::same_as<T, cudf::string_view>)
  {
    return type_to_name(use_physical_type ? jit::physical_type_of(view.type()) : view.type());
  }

  template <typename T>
  std::string operator()(column_view const& view, bool use_physical_type) const
    requires(std::same_as<T, cudf::dictionary32>)
  {
    return std::format(
      "cudf::dictionary_element<{}, {}>",
      get_element_type_name(view.child(cudf::dictionary_indices_column_index), use_physical_type),
      get_element_type_name(view.child(cudf::dictionary_keys_column_index), use_physical_type));
  }

  template <typename T>
  std::string operator()(column_view const& view, bool use_physical_type) const
    requires(!is_fixed_width<T>() && !std::same_as<T, cudf::string_view> &&
             !std::same_as<T, cudf::dictionary32>)
  {
    CUDF_FAIL("Unsupported type for JIT compilation: " + type_to_name(view.type()),
              std::invalid_argument);
  }
};

[[maybe_unused]] static std::string get_element_type_name(column_view const& view,
                                                          bool use_physical_type)
{
  return cudf::type_dispatcher(view.type(), element_type_name_fn{}, view, use_physical_type);
}

[[maybe_unused]] static std::string reflect_input_element(column_view const& c,
                                                          bool use_physical_type)
{
  return get_element_type_name(c, use_physical_type);
}

[[maybe_unused]] static std::string reflect_input_element(scalar_column_view const& c,
                                                          bool use_physical_type)
{
  return get_element_type_name(c.as_column_view(), use_physical_type);
}

[[maybe_unused]] static std::string reflect_output_element(fixed_width_column const& c,
                                                           bool use_physical_type)
{
  return get_element_type_name(c._col->view(), use_physical_type);
}

[[maybe_unused]] static std::string reflect_output_element(string_views_column const&,
                                                           [[maybe_unused]] bool use_physical_type)
{
  return "cudf::string_view";
}

[[maybe_unused]] static std::string reflect_output_element(mutable_strings_column const&,
                                                           [[maybe_unused]] bool use_physical_type)
{
  return "cuda::std::span<char>";
}

[[maybe_unused]] static std::string reflect_input_value_type(column_view const& c,
                                                             bool use_physical_type)
{
  return is_dictionary(c.type()) ? reflect_input_value_type(
                                     c.child(cudf::dictionary_keys_column_index), use_physical_type)
                                 : reflect_input_element(c, use_physical_type);
}

[[maybe_unused]] static std::string reflect_input_value_type(scalar_column_view const& c,
                                                             bool use_physical_type)
{
  return reflect_input_value_type(c.as_column_view(), use_physical_type);
}

[[maybe_unused]] static std::string reflect_output_value_type(fixed_width_column const& c,
                                                              bool use_physical_type)
{
  return reflect_output_element(c, use_physical_type);
}

[[maybe_unused]] static std::string reflect_output_value_type(string_views_column const& c,
                                                              bool use_physical_type)
{
  return reflect_output_element(c, use_physical_type);
}

[[maybe_unused]] static std::string reflect_output_value_type(mutable_strings_column const& c,
                                                              bool use_physical_type)
{
  return reflect_output_element(c, use_physical_type);
}

template <typename T>
[[maybe_unused]] static std::string reflect_nvvm_value_type(bool use_optional)
{
  if (use_optional) {
    return nvvm_type<cuda::std::optional<T>>();
  } else {
    return nvvm_type<T>();
  }
}

struct nvvm_abi_type_name_fn {
  template <typename T>
  std::string operator()(bool use_optional) const
  {
    if constexpr (is_fixed_width<T>() || std::same_as<T, cudf::string_view>) {
      return reflect_nvvm_value_type<T>(use_optional);
    } else {
      CUDF_FAIL("Unsupported type for JIT NVVM ABI reflection", std::invalid_argument);
    }
  }
};

[[maybe_unused]] static std::string reflect_input_value_nvvm_abi_type(column_view const& c,
                                                                      bool use_optional)
{
  return is_dictionary(c.type())
           ? reflect_input_value_nvvm_abi_type(c.child(cudf::dictionary_keys_column_index),
                                               use_optional)
           : cudf::type_dispatcher(c.type(), nvvm_abi_type_name_fn{}, use_optional);
}

[[maybe_unused]] static std::string reflect_input_value_nvvm_abi_type(scalar_column_view const& c,
                                                                      bool use_optional)
{
  return reflect_input_value_nvvm_abi_type(c.as_column_view(), use_optional);
}

[[maybe_unused]] static std::string reflect_output_value_nvvm_abi_type(fixed_width_column const&,
                                                                       bool use_optional)
{
  return reflect_nvvm_value_type<void*>(use_optional);
}

[[maybe_unused]] static std::string reflect_output_value_nvvm_abi_type(string_views_column const&,
                                                                       bool use_optional)
{
  return reflect_nvvm_value_type<void*>(use_optional);
}

[[maybe_unused]] static std::string reflect_output_value_nvvm_abi_type(
  mutable_strings_column const&, bool use_optional)
{
  return reflect_nvvm_value_type<void*>(use_optional);
}

[[maybe_unused]] static std::string reflect_input_column(column_view const&)
{
  return "cudf::column_device_view_core";
}

[[maybe_unused]] static std::string reflect_input_column(scalar_column_view const&)
{
  return "cudf::column_device_view_core";
}

[[maybe_unused]] static std::string reflect_output_column(fixed_width_column const&)
{
  return "cudf::mutable_column_device_view_core";
}

[[maybe_unused]] static std::string reflect_output_column(string_views_column const&)
{
  return "cudf::jit::mutable_vector_device_view";
}

[[maybe_unused]] static std::string reflect_output_column(mutable_strings_column const&)
{
  return "cudf::jit::mutable_strings_column_device_view";
}

}  // namespace jit
}  // namespace cudf
