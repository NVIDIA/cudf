/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/replace.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
struct find_replace_fn {
  column_device_view d_input;
  column_device_view d_values;
  column_device_view d_replacements;

  __device__ string_index_pair get_replacement(size_type idx)
  {
    if (d_replacements.is_null(idx)) { return string_index_pair{nullptr, 0}; }
    auto const d_str = d_replacements.element<string_view>(idx);
    return string_index_pair{d_str.data(), d_str.size_bytes()};
  }

  __device__ string_index_pair operator()(size_type idx)
  {
    if (d_input.is_null(idx)) { return string_index_pair{nullptr, 0}; }
    auto const d_str = d_input.element<string_view>(idx);
    // find d_str in d_values
    // if found return corresponding replacement
    // if not found, return d_str
    auto const begin = cuda::counting_iterator<size_type>{0};
    auto const end   = cuda::counting_iterator<size_type>{d_values.size()};
    auto const itr =
      thrust::find_if(thrust::seq, begin, end, [d_values = d_values, d_str](size_type i) -> bool {
        return d_str == d_values.element<string_view>(i);
      });
    return itr == end ? string_index_pair{d_str.data(), d_str.size_bytes()} : get_replacement(*itr);
  }
};

}  // namespace

std::unique_ptr<cudf::column> find_and_replace_all(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& values_to_replace,
  cudf::strings_column_view const& replacement_values,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto d_input             = cudf::column_device_view::create(input.parent(), stream);
  auto d_values_to_replace = cudf::column_device_view::create(values_to_replace.parent(), stream);
  auto d_replacements      = cudf::column_device_view::create(replacement_values.parent(), stream);

  auto indices = rmm::device_uvector<string_index_pair>(input.size(), stream);

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    indices.begin(),
                    find_replace_fn{*d_input, *d_values_to_replace, *d_replacements});

  return make_strings_column(indices.begin(), indices.end(), stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
