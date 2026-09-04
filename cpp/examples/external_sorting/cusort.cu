/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cusort.hpp"

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/gather.h>

#include <memory>

namespace cudf {
namespace examples {

std::unique_ptr<cudf::column> sample_splitters(cudf::table_view const& table_view,
                                               cudf::size_type num_splitters,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (table_view.num_rows() == 0 || num_splitters <= 0) {
    // Return empty column of the same type as the first column
    return cudf::empty_like(table_view.column(0));
  }

  // Sort this table by first column
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER};

  auto sorted_indices =
    cudf::sorted_order(table_view.select({0}), column_order, null_precedence, stream, mr);
  rmm::device_uvector<cudf::size_type> sampled_indices(num_splitters, stream, mr);
  auto const stride          = table_view.num_rows() / num_splitters;
  auto sample_iterator_begin = cuda::strided_iterator(cuda::counting_iterator(0), stride);
  auto sample_iterator_end =
    cuda::strided_iterator(cuda::counting_iterator(table_view.num_rows()), stride);
  thrust::gather(rmm::exec_policy_nosync(stream),
                 sample_iterator_begin,
                 sample_iterator_end,
                 sorted_indices->view().begin<cudf::size_type>(),
                 sampled_indices.begin());
  cudf::column sampled_indices_col(std::move(sampled_indices), rmm::device_buffer{}, 0);
  auto sampled_values = cudf::gather(table_view.select({0}),
                                     sampled_indices_col.view(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
  return std::move(sampled_values->release()[0]);
}

}  // namespace examples
}  // namespace cudf
