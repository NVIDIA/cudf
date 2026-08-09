/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
std::unique_ptr<column> find_multiple(strings_column_view const& input,
                                      strings_column_view const& targets,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const strings_count = input.size();
  auto const targets_count = targets.size();
  CUDF_EXPECTS(targets_count > 0, "Must include at least one search target", std::invalid_argument);
  CUDF_EXPECTS(
    !targets.has_nulls(), "Search targets cannot contain null strings", std::invalid_argument);

  auto const total_elements =
    static_cast<int64_t>(strings_count) * static_cast<int64_t>(targets_count);
  CUDF_EXPECTS(
    total_elements <= static_cast<decltype(total_elements)>(std::numeric_limits<size_type>::max()),
    "Size of output exceeds the column size limit",
    std::overflow_error);

  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_strings      = *strings_column;
  auto targets_column = column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;

  auto const total_count = static_cast<size_type>(total_elements);

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     total_count,
                                     rmm::device_buffer{0, stream, mr},
                                     0,
                                     stream,
                                     mr);

  // fill output column with position values
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{total_count},
                    results->mutable_view().begin<size_type>(),
                    [d_strings, d_targets, targets_count] __device__(size_type idx) -> size_type {
                      size_type str_idx = idx / targets_count;
                      if (d_strings.is_null(str_idx)) return -1;
                      string_view d_str = d_strings.element<string_view>(str_idx);
                      string_view d_tgt = d_targets.element<string_view>(idx % targets_count);
                      return d_str.find(d_tgt);
                    });
  results->set_null_count(0);

  auto offsets = cudf::detail::sequence(
    strings_count + 1,
    numeric_scalar<size_type>(0, true, stream, cudf::get_current_device_resource_ref()),
    numeric_scalar<size_type>(targets_count, true, stream, cudf::get_current_device_resource_ref()),
    stream,
    mr);
  return make_lists_column(
    strings_count, std::move(offsets), std::move(results), 0, rmm::device_buffer{0, stream, mr});
}

}  // namespace detail

// external API
std::unique_ptr<column> find_multiple(strings_column_view const& input,
                                      strings_column_view const& targets,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find_multiple(input, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
