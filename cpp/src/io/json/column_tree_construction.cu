/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nested_json.hpp"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>

namespace cudf::io::json {

using row_offset_t = size_type;

#ifdef CSR_DEBUG_PRINT
template <typename T>
void print(device_span<T const> d_vec, std::string name, cuda::stream_ref stream)
{
  stream.sync();
  auto h_vec = cudf::detail::make_std_vector(d_vec, stream);
  std::cout << name << " = ";
  for (auto e : h_vec) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
}
#endif

namespace experimental::detail {

struct level_ordering {
  device_span<TreeDepthT const> node_levels;
  device_span<NodeIndexT const> col_ids;
  device_span<NodeIndexT const> parent_node_ids;
  __device__ bool operator()(NodeIndexT lhs_node_id, NodeIndexT rhs_node_id) const
  {
    auto lhs_parent_col_id = parent_node_ids[lhs_node_id] == parent_node_sentinel
                               ? parent_node_sentinel
                               : col_ids[parent_node_ids[lhs_node_id]];
    auto rhs_parent_col_id = parent_node_ids[rhs_node_id] == parent_node_sentinel
                               ? parent_node_sentinel
                               : col_ids[parent_node_ids[rhs_node_id]];

    return (node_levels[lhs_node_id] < node_levels[rhs_node_id]) ||
           (node_levels[lhs_node_id] == node_levels[rhs_node_id] &&
            lhs_parent_col_id < rhs_parent_col_id) ||
           (node_levels[lhs_node_id] == node_levels[rhs_node_id] &&
            lhs_parent_col_id == rhs_parent_col_id && col_ids[lhs_node_id] < col_ids[rhs_node_id]);
  }
};

struct parent_nodeids_to_colids {
  device_span<NodeIndexT const> rev_mapped_col_ids;
  __device__ auto operator()(NodeIndexT parent_node_id) -> NodeIndexT
  {
    return parent_node_id == parent_node_sentinel ? parent_node_sentinel
                                                  : rev_mapped_col_ids[parent_node_id];
  }
};

/**
 * @brief Reduces node tree representation to column tree CSR representation.
 *
 * @param node_tree Node tree representation of JSON string
 * @param original_col_ids Column ids of nodes
 * @param sorted_col_ids Sorted column IDs
 * @param ordered_node_ids Ordered node IDs
 * @param row_offsets Row offsets of nodes
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param row_array_parent_col_id Column id of row array, if is_array_of_arrays is true
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A tuple of column tree representation of JSON string, column ids of columns, and
 * max row offsets of columns
 */
std::tuple<compressed_sparse_row, column_tree_properties> reduce_to_column_tree(
  tree_meta_t& node_tree,
  device_span<NodeIndexT const> original_col_ids,
  device_span<NodeIndexT const> sorted_col_ids,
  device_span<NodeIndexT const> ordered_node_ids,
  device_span<row_offset_t const> row_offsets,
  bool is_array_of_arrays,
  NodeIndexT row_array_parent_col_id,
  cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();

  if (original_col_ids.empty()) {
    rmm::device_uvector<NodeIndexT> empty_row_idx(0, stream);
    rmm::device_uvector<NodeIndexT> empty_col_idx(0, stream);
    rmm::device_uvector<NodeT> empty_column_categories(0, stream);
    rmm::device_uvector<row_offset_t> empty_max_row_offsets(0, stream);
    rmm::device_uvector<NodeIndexT> empty_mapped_col_ids(0, stream);
    return std::tuple{compressed_sparse_row{std::move(empty_row_idx), std::move(empty_col_idx)},
                      column_tree_properties{std::move(empty_column_categories),
                                             std::move(empty_max_row_offsets),
                                             std::move(empty_mapped_col_ids)}};
  }

  auto [unpermuted_tree, unpermuted_col_ids, unpermuted_max_row_offsets] =
    cudf::io::json::detail::reduce_to_column_tree(node_tree,
                                                  original_col_ids,
                                                  sorted_col_ids,
                                                  ordered_node_ids,
                                                  row_offsets,
                                                  is_array_of_arrays,
                                                  row_array_parent_col_id,
                                                  stream);

  NodeIndexT num_columns = unpermuted_col_ids.size();

  auto mapped_col_ids = cudf::detail::make_device_uvector_async(
    unpermuted_col_ids, stream, cudf::get_current_device_resource_ref());
  rmm::device_uvector<NodeIndexT> rev_mapped_col_ids(num_columns, stream);
  rmm::device_uvector<NodeIndexT> reordering_index(unpermuted_col_ids.size(), stream);

  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   reordering_index.begin(),
                   reordering_index.end());
  // Reorder nodes and column ids in level-wise fashion
  thrust::sort_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    reordering_index.begin(),
    reordering_index.end(),
    mapped_col_ids.begin(),
    level_ordering{
      unpermuted_tree.node_levels, unpermuted_col_ids, unpermuted_tree.parent_node_ids});

  {
    auto mapped_col_ids_copy = cudf::detail::make_device_uvector_async(
      mapped_col_ids, stream, cudf::get_current_device_resource_ref());
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     rev_mapped_col_ids.begin(),
                     rev_mapped_col_ids.end());
    thrust::sort_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        mapped_col_ids_copy.begin(),
                        mapped_col_ids_copy.end(),
                        rev_mapped_col_ids.begin());
  }

  rmm::device_uvector<NodeIndexT> parent_col_ids(num_columns, stream);
  cuda::transform_output_iterator parent_col_ids_it(parent_col_ids.begin(),
                                                    parent_nodeids_to_colids{rev_mapped_col_ids});
  rmm::device_uvector<row_offset_t> max_row_offsets(num_columns, stream);
  rmm::device_uvector<NodeT> column_categories(num_columns, stream);
  thrust::copy_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::make_zip_iterator(
      cuda::make_permutation_iterator(unpermuted_tree.parent_node_ids.begin(),
                                      reordering_index.begin()),
      cuda::make_permutation_iterator(unpermuted_max_row_offsets.begin(), reordering_index.begin()),
      cuda::make_permutation_iterator(unpermuted_tree.node_categories.begin(),
                                      reordering_index.begin())),
    num_columns,
    cuda::make_zip_iterator(parent_col_ids_it, max_row_offsets.begin(), column_categories.begin()));

#ifdef CSR_DEBUG_PRINT
  print<NodeIndexT>(reordering_index, "h_reordering_index", stream);
  print<NodeIndexT>(mapped_col_ids, "h_mapped_col_ids", stream);
  print<NodeIndexT>(rev_mapped_col_ids, "h_rev_mapped_col_ids", stream);
  print<NodeIndexT>(parent_col_ids, "h_parent_col_ids", stream);
  print<row_offset_t>(max_row_offsets, "h_max_row_offsets", stream);
#endif

  auto construct_row_idx = [&stream](NodeIndexT num_columns,
                                     device_span<NodeIndexT const> parent_col_ids) {
    auto row_idx = cudf::detail::make_zeroed_device_uvector_async<NodeIndexT>(
      static_cast<std::size_t>(num_columns + 1), stream, cudf::get_current_device_resource_ref());
    // Note that the first element of csr_parent_col_ids is -1 (parent_node_sentinel)
    // children adjacency

    auto num_non_leaf_columns =
      thrust::unique_count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           parent_col_ids.begin() + 1,
                           parent_col_ids.end());
    rmm::device_uvector<NodeIndexT> non_leaf_nodes(num_non_leaf_columns, stream);
    rmm::device_uvector<NodeIndexT> non_leaf_nodes_children(num_non_leaf_columns, stream);
    cudf::detail::reduce_by_key_async(parent_col_ids.begin() + 1,
                                      parent_col_ids.end(),
                                      cuda::make_constant_iterator(1),
                                      non_leaf_nodes.begin(),
                                      non_leaf_nodes_children.begin(),
                                      cuda::std::plus<NodeIndexT>(),
                                      stream);

    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    non_leaf_nodes_children.begin(),
                    non_leaf_nodes_children.end(),
                    non_leaf_nodes.begin(),
                    row_idx.begin() + 1);

    if (num_columns > 1) {
      thrust::transform_inclusive_scan(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        cuda::make_zip_iterator(cuda::counting_iterator<NodeIndexT>{1}, row_idx.begin() + 1),
        cuda::make_zip_iterator(cuda::counting_iterator<NodeIndexT>{1} + num_columns,
                                row_idx.end()),
        row_idx.begin() + 1,
        cuda::proclaim_return_type<NodeIndexT>([] __device__(auto a) {
          auto n   = cuda::std::get<0>(a);
          auto idx = cuda::std::get<1>(a);
          return n == 1 ? idx : idx + 1;
        }),
        cuda::std::plus<NodeIndexT>{});
    } else {
      // Uses thrust::fill instead of device_uvector::set_element_async to prevent the case where
      // single_node goes out of scope before the memcpy-async(stream) completes. This is also
      // allows us to make the copy without incurring a stream synchronize.
      auto single_node = NodeIndexT{1};
      auto exec_policy = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
      thrust::fill(exec_policy, row_idx.begin() + 1, row_idx.end(), single_node);
    }

#ifdef CSR_DEBUG_PRINT
    print<NodeIndexT>(row_idx, "h_row_idx", stream);
#endif
    return row_idx;
  };

  auto construct_col_idx = [&stream](NodeIndexT num_columns,
                                     device_span<NodeIndexT const> parent_col_ids,
                                     device_span<NodeIndexT const> row_idx) {
    rmm::device_uvector<NodeIndexT> col_idx((num_columns - 1) * 2, stream);
    thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                 col_idx.begin(),
                 col_idx.end(),
                 -1);
    // excluding root node, construct scatter map
    rmm::device_uvector<NodeIndexT> map(num_columns - 1, stream);
    thrust::inclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      parent_col_ids.begin() + 1,
      parent_col_ids.end(),
      cuda::make_constant_iterator(1),
      map.begin());
    thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<NodeIndexT>{1},
                       num_columns - 1,
                       [row_idx        = row_idx.begin(),
                        map            = map.begin(),
                        parent_col_ids = parent_col_ids.begin()] __device__(auto i) {
                         auto parent_col_id = parent_col_ids[i];
                         if (parent_col_id == 0)
                           --map[i - 1];
                         else
                           map[i - 1] += row_idx[parent_col_id];
                       });
    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<NodeIndexT>{1},
                    cuda::counting_iterator<NodeIndexT>{1} + num_columns - 1,
                    map.begin(),
                    col_idx.begin());

    // Skip the parent of root node
    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    parent_col_ids.begin() + 1,
                    parent_col_ids.end(),
                    row_idx.begin() + 1,
                    col_idx.begin());

#ifdef CSR_DEBUG_PRINT
    print<NodeIndexT>(col_idx, "h_col_idx", stream);
#endif

    return col_idx;
  };

  /*
    5. CSR construction:
      a. Sort column levels and get their ordering
      b. For each column node coln iterated according to sorted_column_levels; do
          i. Find nodes that have coln as the parent node -> set adj_coln
          ii. row idx[coln] = size of adj_coln + 1
          iii. col idx[coln] = adj_coln U {parent_col_id[coln]}
  */
  auto row_idx = construct_row_idx(num_columns, parent_col_ids);
  auto col_idx = construct_col_idx(num_columns, parent_col_ids, row_idx);

  return std::tuple{
    compressed_sparse_row{std::move(row_idx), std::move(col_idx)},
    column_tree_properties{
      std::move(column_categories), std::move(max_row_offsets), std::move(mapped_col_ids)}};
}

}  // namespace experimental::detail
}  // namespace cudf::io::json
