/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

namespace cudf::detail {

/**
 * @brief Fill label values for segments defined by a given offsets array.
 *
 * Given a pair of iterators accessing to an offset array, generate label values for segments
 * defined by the offset values. The output will be an array containing consecutive groups of
 * identical labels, the number of elements in each group `i` is defined by
 * `offsets[i+1] - offsets[i]`.
 *
 * The labels always start from `0` regardless of the offset values.
 * In case there are empty segments, their corresponding label values will be skipped in the output.
 *
 * Note that the caller is responsible to make sure the output range have the correct size, which is
 * the total segment sizes (i.e., `size = *(offsets_end - 1) - *offsets_begin`). Otherwise, the
 * result is undefined.
 *
 * @code{.pseudo}
 * Examples:
 *
 * offsets = [ 0, 4, 6, 6, 6, 10 ]
 * output  = [ 0, 0, 0, 0, 1, 1, 4, 4, 4, 4 ]
 *
 * offsets = [ 5, 10, 12 ]
 * output  = [ 0, 0, 0, 0, 0, 1, 1 ]
 * @endcode
 *
 * @param offsets_begin The beginning of the offsets that define segments.
 * @param offsets_end The end of the offsets that define segments.
 * @param label_begin The beginning of the output label range.
 * @param label_end The end of the output label range.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename InputIterator, typename OutputIterator>
void label_segments(InputIterator offsets_begin,
                    InputIterator offsets_end,
                    OutputIterator label_begin,
                    OutputIterator label_end,
                    cuda::stream_ref stream)
{
  auto const num_labels = cuda::std::distance(label_begin, label_end);

  // If the output array is empty, that means we have all empty segments.
  // In such cases, we must terminate immediately. Otherwise, the `for_each` loop below may try to
  // access memory of the output array, resulting in "illegal memory access" error.
  if (num_labels == 0) { return; }

  // When the output array is not empty, always fill it with `0` value first.
  using OutputType = cuda::std::iter_value_t<OutputIterator>;
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    label_begin,
    label_end,
    OutputType{0});

  // If the offsets array has no more than 2 offset values, there will be at max 1 segment.
  // In such cases, the output will just be an array of all `0` values (which we already filled).
  // We should terminate from here, otherwise the `inclusive_scan` call below still does its entire
  // computation. That is unnecessary and may be expensive if we have the input offsets defining a
  // very large segment.
  if (cuda::std::distance(offsets_begin, offsets_end) <= 2) { return; }

  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   offsets_begin + 1,  // exclude the first offset value
                   offsets_end - 1,    // exclude the last offset value
                   [num_labels = static_cast<cuda::std::iter_value_t<InputIterator>>(num_labels),
                    offsets    = offsets_begin,
                    output     = label_begin] __device__(auto const idx) {
                     // Zero-normalized offsets.
                     auto const dst_idx = idx - (*offsets);

                     // Scatter value `1` to the index at (idx - offsets[0]).
                     // Note that we need to check for out of bound, since the offset values may be
                     // invalid due to empty segments at the end. In case we have repeated offsets
                     // (i.e., we have empty segments), this `atomicAdd` call will make sure the
                     // label values corresponding to these empty segments will be skipped in the
                     // output.
                     if (dst_idx < num_labels) { atomicAdd(&output[dst_idx], OutputType{1}); }
                   });
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         label_begin,
                         label_end,
                         label_begin);
}

/**
 * @brief Generate segment offsets from groups of identical label values.
 *
 * Given a pair of iterators accessing to an array containing groups of identical label values,
 * generate offsets for segments defined by these label.
 *
 * Empty segments are also taken into account. If the input label values are discontinuous, the
 * segments corresponding to the missing labels will be inferred as empty segments and their offsets
 * will also be generated.
 *
 * Note that the caller is responsible to make sure the output range for offsets have the correct
 * size, which is the maximum label value plus two (i.e., `size = *(labels_end - 1) + 2`).
 * Otherwise, the result is undefined.
 *
 * @code{.pseudo}
 * Examples:
 *
 * labels = [ 0, 0, 0, 0, 1, 1, 4, 4, 4, 4 ]
 * output = [ 0, 4, 6, 6, 6, 10 ]
 *
 * labels = [ 0, 0, 0, 0, 0, 1, 1 ]
 * output = [ 0, 5, 7 ]
 * @endcode
 *
 * @param labels_begin The beginning of the labels that define segments.
 * @param labels_end The end of the labels that define segments.
 * @param offsets_begin The beginning of the output offset range.
 * @param offsets_end The end of the output offset range.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename InputIterator, typename OutputIterator>
void labels_to_offsets(InputIterator labels_begin,
                       InputIterator labels_end,
                       OutputIterator offsets_begin,
                       OutputIterator offsets_end,
                       cuda::stream_ref stream)
{
  // Always fill the entire output array with `0` value regardless of the input.
  using OutputType = cuda::std::iter_value_t<OutputIterator>;
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    offsets_begin,
    offsets_end,
    OutputType{0});

  // If there is not any label value, we will have zero segment or all empty segments. We should
  // terminate from here because:
  //  - If we have zero segment, the output array is empty thus `num_segments` computed below is
  //    wrong and may cascade to undefined behavior if we continue.
  //  - If we have all empty segments, the output offset values will be all `0`, which we already
  //    filled above. If we continue, the `exclusive_scan` call below still does its entire
  //    computation. That is unnecessary and may be expensive if we have the input labels defining
  //    a very large number of segments.
  if (cuda::std::distance(labels_begin, labels_end) == 0) { return; }

  auto const num_segments = cuda::std::distance(offsets_begin, offsets_end) - 1;

  //================================================================================
  // Let's consider an example: Given input labels = [ 0, 0, 0, 0, 1, 1, 4, 4, 4, 4 ].

  // This stores the unique label values.
  // Given the example above, we will have this array containing [0, 1, 4].
  auto list_indices = rmm::device_uvector<OutputType>(num_segments, stream);

  // Stores the non-zero segment sizes.
  // Given the example above, we will have this array containing [4, 2, 4].
  auto list_sizes = rmm::device_uvector<OutputType>(num_segments, stream);

  // Count the numbers of labels in the each segment.
  auto const end = cudf::detail::reduce_by_key(labels_begin,  // keys
                                               labels_end,
                                               cuda::make_constant_iterator<OutputType>(1),
                                               list_indices.begin(),  // output unique label values
                                               list_sizes.begin(),    // count for each label
                                               cuda::std::plus<OutputType>(),
                                               stream);

  auto const num_non_empty_segments = cuda::std::distance(list_indices.begin(), end.first);

  // Scatter segment sizes into the end position of their corresponding segment indices.
  // Given the example above, we scatter [4, 2, 4] by the scatter map [0, 1, 4], resulting
  // output = [4, 2, 0, 0, 4, 0].
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  list_sizes.begin(),
                  list_sizes.begin() + num_non_empty_segments,
                  list_indices.begin(),
                  offsets_begin);

  // Generate offsets from sizes.
  // Given the example above, the final output is [0, 4, 6, 6, 6, 10].
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         offsets_begin,
                         offsets_end,
                         offsets_begin);
}

}  // namespace cudf::detail
