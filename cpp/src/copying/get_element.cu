/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/is_element_valid.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/lists/detail/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <stdexcept>

namespace cudf {
namespace detail {

namespace {

struct get_element_functor {
  template <typename T, std::enable_if_t<is_fixed_width<T>() && !is_fixed_point<T>()>* p = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     size_type index,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    auto s = make_fixed_width_scalar(data_type(type_to_id<T>()), stream, mr);

    using ScalarType = cudf::scalar_type_t<T>;
    auto typed_s     = static_cast<ScalarType*>(s.get());

    auto device_s   = get_scalar_device_view(*typed_s);
    auto device_col = column_device_view::create(input, stream);

    device_single_thread(
      [device_s, d_col = *device_col, index] __device__() mutable {
        device_s.set_value(d_col.element<T>(index));
        device_s.set_valid(d_col.is_valid(index));
      },
      stream);
    return s;
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, string_view>>* p = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     size_type index,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    auto device_col = column_device_view::create(input, stream);

    auto const temp_mr = cudf::get_current_device_resource_ref();
    cudf::detail::device_scalar<string_view> temp_data(stream, temp_mr);
    cudf::detail::device_scalar<bool> temp_valid(stream, temp_mr);

    device_single_thread(
      [buffer   = temp_data.data(),
       validity = temp_valid.data(),
       d_col    = *device_col,
       index] __device__() mutable {
        *buffer   = d_col.element<string_view>(index);
        *validity = d_col.is_valid(index);
      },
      stream);

    return std::make_unique<string_scalar>(
      temp_data.value(stream), temp_valid.value(stream), stream, mr);
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, dictionary32>>* p = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     size_type index,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    auto dict_view    = dictionary_column_view(input);
    auto indices_iter = detail::indexalator_factory::make_input_iterator(dict_view.indices());
    numeric_scalar<size_type> key_index_scalar{
      index, true, stream, cudf::get_current_device_resource_ref()};
    auto d_key_index = get_scalar_device_view(key_index_scalar);
    auto d_col       = column_device_view::create(input, stream);

    // retrieve the indices value at index
    device_single_thread(
      [d_key_index, d_col = *d_col, indices_iter, index] __device__() mutable {
        d_key_index.set_value(indices_iter[index]);
        d_key_index.set_valid(d_col.is_valid(index));
      },
      stream);

    if (!key_index_scalar.is_valid(stream)) {
      return make_default_constructed_scalar(dict_view.keys().type(), stream, mr);
    }

    // retrieve the key element using the key-index
    return type_dispatcher(dict_view.keys().type(),
                           get_element_functor{},
                           dict_view.keys(),
                           key_index_scalar.value(stream),
                           stream,
                           mr);
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, list_view>>* p = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     size_type index,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    bool valid               = is_element_valid_sync(input, index, stream);
    auto const child_col_idx = lists_column_view::child_column_index;

    if (valid) {
      lists_column_view lcv(input);
      // Make a copy of the row
      auto row_slice_contents =
        lists::detail::copy_slice(lcv, index, index + 1, stream, mr)->release();
      // Construct scalar with row data
      return std::make_unique<list_scalar>(
        std::move(*row_slice_contents.children[child_col_idx]), valid, stream, mr);
    } else {
      auto empty_row_contents = empty_like(input)->release();
      return std::make_unique<list_scalar>(
        std::move(*empty_row_contents.children[child_col_idx]), valid, stream, mr);
    }
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* p = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     size_type index,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    using Type = typename T::rep;

    auto device_col = column_device_view::create(input, stream);

    auto result = std::make_unique<fixed_point_scalar<T>>(
      Type{}, numeric::scale_type{input.type().scale()}, false, stream, mr);

    device_single_thread(
      [buffer   = result->data(),
       validity = result->validity_data(),
       d_col    = *device_col,
       index] __device__() mutable {
        *buffer   = d_col.element<Type>(index);
        *validity = d_col.is_valid(index);
      },
      stream);

    return result;
  }

  template <typename T, std::enable_if_t<std::is_same_v<T, struct_view>>* p = nullptr>
  std::unique_ptr<scalar> operator()(column_view const& input,
                                     size_type index,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    bool valid = is_element_valid_sync(input, index, stream);
    auto row_contents =
      std::make_unique<column>(slice(input, index, index + 1, stream), stream, mr)->release();
    auto scalar_contents = table(std::move(row_contents.children));
    return std::make_unique<struct_scalar>(std::move(scalar_contents), valid, stream, mr);
  }
};

}  // namespace

std::unique_ptr<scalar> get_element(column_view const& input,
                                    size_type index,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(index >= 0 and index < input.size(), "Index out of bounds", std::out_of_range);
  return type_dispatcher(input.type(), get_element_functor{}, input, index, stream, mr);
}

}  // namespace detail

std::unique_ptr<scalar> get_element(column_view const& input,
                                    size_type index,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_element(input, index, stream, mr);
}

}  // namespace cudf
