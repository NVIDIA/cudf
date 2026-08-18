/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/fill.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

struct fill_fn {
  column_device_view const d_strings;
  size_type const begin;
  size_type const end;
  string_scalar_device_view const d_value;

  __device__ string_index_pair operator()(size_type idx) const
  {
    auto d_str = string_view();
    if ((begin <= idx) && (idx < end)) {
      if (!d_value.is_valid()) { return string_index_pair{nullptr, 0}; }
      d_str = d_value.value();
    } else {
      if (d_strings.is_null(idx)) { return string_index_pair{nullptr, 0}; }
      d_str = d_strings.element<string_view>(idx);
    }
    return !d_str.empty() ? string_index_pair{d_str.data(), d_str.size_bytes()}
                          : string_index_pair{"", 0};
  }
};

}  // namespace

std::unique_ptr<column> fill(strings_column_view const& input,
                             size_type begin,
                             size_type end,
                             string_scalar const& value,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS((begin >= 0) && (end <= strings_count),
               "Parameters [begin,end) are outside the range of the provided strings column");
  CUDF_EXPECTS(begin <= end, "Parameters [begin,end) have invalid range values");
  if (begin == end) { return std::make_unique<column>(input.parent(), stream, mr); }

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_value   = cudf::get_scalar_device_view(const_cast<string_scalar&>(value));

  auto fn = fill_fn{*d_strings, begin, end, d_value};
  rmm::device_uvector<string_index_pair> indices(strings_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    indices.begin(),
                    fn);

  return make_strings_column(indices.begin(), indices.end(), stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
