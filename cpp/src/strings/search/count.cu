/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/find.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

struct counter_fn {
  column_device_view d_strings;
  string_view d_target;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx) || d_target.empty()) { return 0; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) { return 0; }

    auto const tgt_size = d_target.size_bytes();
    auto itr            = d_str.data();
    auto const end      = itr + d_str.size_bytes();
    size_type count     = 0;
    while (itr + tgt_size <= end) {
      if (d_target.compare(itr, tgt_size) == 0) {
        ++count;
        itr += tgt_size;
      } else {
        ++itr;
      }
    }
    return count;
  }
};
}  // namespace

std::unique_ptr<column> count(strings_column_view const& input,
                              string_scalar const& target,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(target.is_valid(stream), "parameter target must be valid", std::invalid_argument);
  auto d_target = string_view(target.data(), target.size());

  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_results = results->mutable_view().data<size_type>();

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    d_results,
                    counter_fn{*d_strings, d_target});

  return results;
}
}  // namespace detail

// external APIs

std::unique_ptr<column> count(strings_column_view const& strings,
                              string_scalar const& target,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::count(strings, target, stream, mr);
}

}  // namespace strings
}  // namespace cudf
