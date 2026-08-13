/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cmath>  // for std::ceil()
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

/**
 * @brief Handles the "degenerate" case num_partitions >= num_rows.
 *
 * Specifically,
 * If num_partitions == nrows:
 * Then, offsets = [0..nrows-1]
 * gather_row_indices = rotate [0..nrows-1] right by start_partition positions;
 *
 * If num_partitions > nrows:
 * Then, let:
 * dbg = generate a directed bipartite graph with num_partitions nodes and nrows edges,
 * so that node j has an edge to node (j+start_partition) % num_partitions, for j = 0,...,nrows-1;
 *
 * transpose_dbg = transpose graph of dbg; (i.e., (i -> j) edge in dbg means (j -> i) edge in
 * transpose);
 *
 * (offsets, indices) = (row_offsets, col_indices) of transpose_dbg;
 * where (row_offsets, col_indices) are the CSR format of the graph;
 *
 * @param[in] input The input table to be round-robin partitioned
 * @param[in] num_partitions Number of partitions for the table
 * @param[in] start_partition Index of the 1st partition
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 * @param[in] mr Device memory resource used to allocate the returned table's device memory
 *
 * @returns A std::pair consisting of a unique_ptr to the partitioned table and the partition
 * offsets for each partition within the table
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> degenerate_partitions(
  cudf::table_view const& input,
  cudf::size_type num_partitions,
  cudf::size_type start_partition,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto nrows = input.num_rows();

  // iterator for partition index rotated right by start_partition positions:
  auto rotated_iter_begin = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::proclaim_return_type<cudf::size_type>(
      [num_partitions, start_partition] __device__(auto index) {
        return (index + num_partitions - start_partition) % num_partitions;
      }));

  if (num_partitions == nrows) {
    rmm::device_uvector<cudf::size_type> partition_offsets(num_partitions + 1, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     partition_offsets.begin(),
                     partition_offsets.end());

    auto uniq_tbl = cudf::detail::gather(input,
                                         rotated_iter_begin,
                                         rotated_iter_begin + nrows,  // map
                                         cudf::out_of_bounds_policy::DONT_CHECK,
                                         stream,
                                         mr);

    return std::pair{std::move(uniq_tbl), cudf::detail::make_std_vector(partition_offsets, stream)};
  } else {  //( num_partitions > nrows )
    rmm::device_uvector<cudf::size_type> d_row_indices(nrows, stream);

    // copy rotated right partition indexes that
    // fall in the interval [0, nrows):
    //(this relies on a _stable_ copy_if())
    cudf::detail::copy_if_async(
      rotated_iter_begin,
      rotated_iter_begin + num_partitions,
      d_row_indices.begin(),
      [nrows] __device__(auto index) -> bool { return (index < nrows); },
      stream);

    //...and then use the result, d_row_indices, as gather map:
    auto uniq_tbl = cudf::detail::gather(input,
                                         d_row_indices.begin(),
                                         d_row_indices.end(),  // map
                                         cudf::out_of_bounds_policy::DONT_CHECK,
                                         stream,
                                         mr);

    // offsets (part 1: compute partition sizes);
    // iterator for number of edges of the transposed bipartite graph;
    // this composes rotated_iter transform (above) iterator with
    // calculating number of edges of transposed bi-graph:
    auto nedges_iter_begin = thrust::make_transform_iterator(
      rotated_iter_begin,
      cuda::proclaim_return_type<cudf::size_type>(
        [nrows] __device__(auto index) { return (index < nrows ? 1 : 0); }));

    // offsets (part 2: compute partition offsets):
    rmm::device_uvector<cudf::size_type> partition_offsets(num_partitions + 1, stream);
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           nedges_iter_begin,
                           nedges_iter_begin + num_partitions,
                           partition_offsets.begin());
    // Add the total row count as the last offset
    thrust::fill_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   partition_offsets.end() - 1,
                   1,
                   nrows);

    return std::pair{std::move(uniq_tbl), cudf::detail::make_std_vector(partition_offsets, stream)};
  }
}
}  // namespace

namespace cudf {
namespace detail {
std::pair<std::unique_ptr<table>, std::vector<cudf::size_type>> round_robin_partition(
  table_view const& input,
  cudf::size_type num_partitions,
  cudf::size_type start_partition,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto nrows = input.num_rows();

  CUDF_EXPECTS(num_partitions > 0, "Incorrect number of partitions. Must be greater than 0.");
  CUDF_EXPECTS(start_partition < num_partitions,
               "Incorrect start_partition index. Must be less than number of partitions.");
  CUDF_EXPECTS(
    start_partition >= 0,
    "Incorrect start_partition index. Must be positive.");  // since cudf::size_type is an alias for
                                                            // int32_t, it _can_ be negative

  if (nrows == 0) {
    return std::pair{empty_like(input), std::vector<size_type>(num_partitions + 1, 0)};
  }

  // handle degenerate case:
  //
  if (num_partitions >= nrows) {
    return degenerate_partitions(input, num_partitions, start_partition, stream, mr);
  }

  auto np_max_size = nrows % num_partitions;  // # partitions of max size

  // handle case when nr `mod` np == 0;
  // fix for bug: https://github.com/rapidsai/cudf/issues/4043
  auto num_partitions_max_size = (np_max_size > 0 ? np_max_size : num_partitions);

  cudf::size_type max_partition_size = std::ceil(
    static_cast<double>(nrows) / static_cast<double>(num_partitions));  // max size of partitions

  auto total_max_partitions_size = num_partitions_max_size * max_partition_size;
  auto num_partitions_min_size   = num_partitions - num_partitions_max_size;

  // delta is the number of positions to rotate right
  // the original range [0,1,...,n-1]
  // and is calculated by accumulating the first
  //`start_partition` partition sizes from the end;
  // i.e.,
  // the partition sizes array (of size p) being:
  //[m,m,...,m,(m-1),...,(m-1)]
  //(with num_partitions_max_size sizes `m` at the beginning;
  // and (p-num_partitions_max_size) sizes `(m-1)` at the end)
  // we accumulate the 1st `start_partition` entries from the end:
  //
  auto delta = (start_partition > num_partitions_min_size
                  ? num_partitions_min_size * (max_partition_size - 1) +
                      (start_partition - num_partitions_min_size) * max_partition_size
                  : start_partition * (max_partition_size - 1));

  auto iter_begin = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::proclaim_return_type<size_type>([nrows,
                                           num_partitions,
                                           max_partition_size,
                                           num_partitions_max_size,
                                           total_max_partitions_size,
                                           delta] __device__(auto index0) {
      // rotate original index right by delta positions;
      // this is the effect of applying start_partition:
      //
      auto rotated_index = (index0 + nrows - delta) % nrows;

      // using rotated_index = given index0, rotated;
      // the algorithm below calculates the src round-robin row,
      // by calculating the partition_index and the index_within_partition:
      //
      auto index_within_partition =
        (rotated_index <= total_max_partitions_size
           ? rotated_index % max_partition_size
           : (rotated_index - total_max_partitions_size) % (max_partition_size - 1));
      auto partition_index =
        (rotated_index <= total_max_partitions_size
           ? rotated_index / max_partition_size
           : num_partitions_max_size +
               (rotated_index - total_max_partitions_size) / (max_partition_size - 1));
      return num_partitions * index_within_partition + partition_index;
    }));

  auto uniq_tbl = cudf::detail::gather(
    input, iter_begin, iter_begin + nrows, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto ret_pair = std::pair{std::move(uniq_tbl), std::vector<cudf::size_type>(num_partitions + 1)};

  // this has the effect of rotating the set of partition sizes
  // right by start_partition positions:
  //
  auto rotated_iter_begin = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0},
    [num_partitions, start_partition, max_partition_size, num_partitions_max_size](auto index) {
      return ((index + num_partitions - start_partition) % num_partitions < num_partitions_max_size
                ? max_partition_size
                : max_partition_size - 1);
    });

  // then exclusive_scan on the resulting
  // rotated partition sizes to get the partition offsets
  // corresponding to start_partition:
  // Since:
  //"num_partitions is usually going to be relatively small
  //(<1,000), as such, it's probably more expensive to do this on the device.
  // Instead, do it on the host directly into the std::vector and avoid the memcpy." - JH
  //
  thrust::exclusive_scan(
    thrust::host, rotated_iter_begin, rotated_iter_begin + num_partitions, ret_pair.second.begin());
  // Add the total row count as the last offset
  ret_pair.second[num_partitions] = nrows;

  return ret_pair;
}

}  // namespace detail

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> round_robin_partition(
  table_view const& input,
  cudf::size_type num_partitions,
  cudf::size_type start_partition,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::round_robin_partition(input, num_partitions, start_partition, stream, mr);
}

}  // namespace cudf
