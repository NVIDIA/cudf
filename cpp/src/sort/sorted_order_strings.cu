/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_strings.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/stream>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Compares two rows of the input strings column by index
 *
 * The input column is known to have no nulls so the strings can be
 * dereferenced and compared without any null checking.
 */
struct strings_comparator_fn {
  column_device_view const d_strings;
  bool ascending;
  __device__ bool operator()(size_type lhs, size_type rhs) const
  {
    auto const lhs_str = d_strings.element<string_view>(lhs);
    auto const rhs_str = d_strings.element<string_view>(rhs);
    return ascending ? (lhs_str < rhs_str) : (rhs_str < lhs_str);
  }
};

}  // namespace

bool is_strings_sortable(column_view const& column)
{
  return column.type().id() == type_id::STRING && !column.has_nulls() && column.size() > 0;
}

template <sort_method method>
void sorted_order_strings(column_view const& input,
                          mutable_column_view& indices,
                          bool ascending,
                          cuda::stream_ref stream)
{
  auto const n       = input.size();
  auto const d_input = column_device_view::create(input, stream);

  // only the indices are sorted; the comparator dereferences the strings on demand so no
  // string_view array is ever materialized
  auto const in_keys  = cuda::counting_iterator<size_type>{0};
  auto const out_keys = indices.begin<size_type>();

  auto const comp = strings_comparator_fn{*d_input, ascending};
  // the environment provides cub with the stream and the memory resource it uses for
  // its temporary storage
  auto const mr_env = cuda::std::execution::prop{cuda::mr::get_memory_resource_t{},
                                                 cudf::get_current_device_resource_ref()};
  auto const env    = cuda::std::execution::env{cuda::stream_ref{stream.get()}, mr_env};
  // Compiling the cub sort APIs is expensive so use a constexpr condition
  // to only compile the one that is needed.
  if constexpr (method == sort_method::STABLE) {
    CUDF_CUDA_TRY(cub::DeviceMergeSort::StableSortKeysCopy(in_keys, out_keys, n, comp, env));
  } else {
    CUDF_CUDA_TRY(cub::DeviceMergeSort::SortKeysCopy(in_keys, out_keys, n, comp, env));
  }
}

template void sorted_order_strings<sort_method::STABLE>(column_view const&,
                                                        mutable_column_view&,
                                                        bool,
                                                        cuda::stream_ref);
template void sorted_order_strings<sort_method::UNSTABLE>(column_view const&,
                                                          mutable_column_view&,
                                                          bool,
                                                          cuda::stream_ref);

}  // namespace detail
}  // namespace cudf
