/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
/**
 * @brief Creates a column of `type_id::BOOL8` elements by applying a predicate to every element
 * between
 * [`begin, `end`) `true` indicates the value is satisfies the predicate and `false` indicates it
 * doesn't.
 *
 * @tparam InputIterator Iterator type for `begin` and `end`
 * @tparam Predicate A predicator type which will be evaluated
 * @param begin Beginning of the sequence of elements
 * @param end End of the sequence of elements
 * @param size Size of the output column
 * @param p Predicate to be applied to each element in `[begin,end)`
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column of type `type_id::BOOL8,` with `true` representing predicate is satisfied.
 */

template <typename InputIterator, typename Predicate>
std::unique_ptr<column> true_if(InputIterator begin,
                                InputIterator end,
                                size_type size,
                                Predicate p,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  auto output =
    make_numeric_column(data_type(type_id::BOOL8), size, mask_state::UNALLOCATED, stream, mr);
  auto output_mutable_view = output->mutable_view();
  auto output_data         = output_mutable_view.data<bool>();

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    begin,
                    end,
                    output_data,
                    p);

  return output;
}

}  // namespace detail
}  // namespace cudf
