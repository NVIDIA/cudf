/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/device/device_find.cuh>

namespace cudf {
namespace dictionary {
namespace detail {

namespace {

/**
 * @brief Find index of a given key within a dictionary's keys column
 *
 * The index is the position within the keys column where the given key (scalar) is found.
 * The result is an integer scalar identifying the index value.
 * If the key is not found, the resulting scalar is set `is_valid()=false`.
 */
struct find_index_fn {
  template <typename Element>
  std::unique_ptr<scalar> operator()(dictionary_column_view const& input,
                                     scalar const& key,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(not std::is_same_v<Element, dictionary32> and
             not std::is_same_v<Element, list_view> and not std::is_same_v<Element, struct_view>)
  {
    auto const num_keys = input.keys_size();
    if (!key.is_valid(stream) || num_keys == 0) {
      return std::make_unique<numeric_scalar<size_type>>(0, false, stream, mr);
    }

    CUDF_EXPECTS(cudf::have_same_types(input.parent(), key),
                 "search key type must match dictionary keys type",
                 std::invalid_argument);

    using ScalarType = cudf::scalar_type_t<Element>;
    auto const find_key =
      get_scalar_device_view(static_cast<ScalarType&>(const_cast<scalar&>(key)));
    auto keys_view  = column_device_view::create(input.keys(), stream);
    auto const keys = keys_view->begin<Element>();

    auto result   = std::make_unique<numeric_scalar<size_type>>(-1, true, stream, mr);
    auto find_fn  = [find_key] __device__(auto const& k) { return k == find_key.value(); };
    auto tmp_size = std::size_t{0};
    CUDF_CUDA_TRY(cub::DeviceFind::FindIf(
      nullptr, tmp_size, keys, result->data(), find_fn, num_keys, stream.value()));
    auto tmp = rmm::device_buffer(tmp_size, stream);
    CUDF_CUDA_TRY(cub::DeviceFind::FindIf(
      tmp.data(), tmp_size, keys, result->data(), find_fn, num_keys, stream.value()));
    if (result->value(stream) == num_keys) { result->set_valid_async(false, stream); }
    return result;
  }

  template <typename Element>
  std::unique_ptr<scalar> operator()(dictionary_column_view const&,
                                     scalar const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
    requires(std::is_same_v<Element, dictionary32> or std::is_same_v<Element, list_view> or
             std::is_same_v<Element, struct_view>)
  {
    CUDF_FAIL(
      "dictionary, list_view, and struct_view columns cannot be the keys column of a dictionary");
  }
};

}  // namespace

std::unique_ptr<scalar> get_index(dictionary_column_view const& dictionary,
                                  scalar const& key,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  if (dictionary.is_empty()) {
    return std::make_unique<numeric_scalar<int32_t>>(0, false, stream, mr);
  }
  return type_dispatcher<dispatch_storage_type>(
    dictionary.keys().type(), find_index_fn(), dictionary, key, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<scalar> get_index(dictionary_column_view const& dictionary,
                                  scalar const& key,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::get_index(dictionary, key, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
