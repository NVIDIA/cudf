/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/detail/merge.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
std::unique_ptr<column> merge(strings_column_view const& lhs,
                              strings_column_view const& rhs,
                              cudf::detail::index_vector const& row_order,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  using cudf::detail::side;
  if (row_order.is_empty()) { return make_empty_column(type_id::STRING); }
  auto const strings_count = static_cast<cudf::size_type>(row_order.size());

  auto const lhs_column = column_device_view::create(lhs.parent(), stream);
  auto const d_lhs      = *lhs_column;
  auto const rhs_column = column_device_view::create(rhs.parent(), stream);
  auto const d_rhs      = *rhs_column;

  auto const begin = row_order.begin();

  // build vector of strings
  rmm::device_uvector<string_index_pair> indices(strings_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{strings_count},
                    indices.begin(),
                    [d_lhs, d_rhs, begin] __device__(size_type idx) {
                      auto const [s, index] = begin[idx];
                      if (s == side::LEFT ? d_lhs.is_null(index) : d_rhs.is_null(index)) {
                        return string_index_pair{nullptr, 0};
                      }
                      auto d_str = (s == side::LEFT) ? d_lhs.element<string_view>(index)
                                                     : d_rhs.element<string_view>(index);
                      return d_str.size_bytes() == 0
                               ? string_index_pair{"", 0}  // ensures empty != null
                               : string_index_pair{d_str.data(), d_str.size_bytes()};
                    });

  // convert vector into strings column
  return cudf::make_strings_column(indices, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
