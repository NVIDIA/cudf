/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/parsing_utils.cuh"
#include "nested_json.hpp"

#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace cudf::io::json {
namespace detail {

// The node that a token represents
struct token_to_node {
  __device__ auto operator()(PdaTokenT const token) -> NodeT
  {
    switch (token) {
      case token_t::StructBegin: return NC_STRUCT;
      case token_t::ListBegin: return NC_LIST;
      case token_t::StringBegin: return NC_STR;
      case token_t::ValueBegin:
        return NC_STR;  // NC_VAL;
      // NV_VAL is removed because type inference and
      // reduce_to_column_tree category collapsing takes care of this.
      case token_t::FieldNameBegin: return NC_FN;
      default: return NC_ERR;
    };
  }
};

// Convert token indices to node range for each valid node.
struct node_ranges {
  device_span<PdaTokenT const> tokens;
  device_span<SymbolOffsetT const> token_indices;
  bool include_quote_char;
  __device__ auto operator()(size_type i) -> cuda::std::tuple<SymbolOffsetT, SymbolOffsetT>
  {
    // Whether a token expects to be followed by its respective end-of-* token partner
    auto const is_begin_of_section = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin:
        case token_t::ValueBegin:
        case token_t::FieldNameBegin: return true;
        default: return false;
      };
    };
    // The end-of-* partner token for a given beginning-of-* token
    auto const end_of_partner = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin: return token_t::StringEnd;
        case token_t::ValueBegin: return token_t::ValueEnd;
        case token_t::FieldNameBegin: return token_t::FieldNameEnd;
        default: return token_t::ErrorBegin;
      };
    };
    // Includes quote char for end-of-string token or Skips the quote char for
    // beginning-of-field-name token
    auto const get_token_index = [include_quote_char = include_quote_char] __device__(
                                   PdaTokenT const token, SymbolOffsetT const token_index) {
      constexpr SymbolOffsetT quote_char_size = 1;
      switch (token) {
        // Strip off quote char included for StringBegin
        case token_t::StringBegin: return token_index + (include_quote_char ? 0 : quote_char_size);
        // Strip off or Include trailing quote char for string values for StringEnd
        case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
        // Strip off quote char included for FieldNameBegin
        case token_t::FieldNameBegin: return token_index + quote_char_size;
        default: return token_index;
      };
    };
    PdaTokenT const token = tokens[i];
    // The section from the original JSON input that this token demarcates
    SymbolOffsetT range_begin = get_token_index(token, token_indices[i]);
    SymbolOffsetT range_end   = range_begin + 1;  // non-leaf, non-field nodes ignore this value.
    if (is_begin_of_section(token)) {
      if ((i + 1) < tokens.size() && end_of_partner(token) == tokens[i + 1]) {
        // Update the range_end for this pair of tokens
        range_end = get_token_index(tokens[i + 1], token_indices[i + 1]);
      }
    }
    return cuda::std::make_tuple(range_begin, range_end);
  }
};

struct is_nested_end {
  PdaTokenT const* tokens;
  __device__ auto operator()(NodeIndexT i) -> bool
  {
    return tokens[i] == token_t::StructEnd or tokens[i] == token_t::ListEnd;
  }
};

struct checked_token_level_output {
  int32_t* depth_out_of_range;

  __device__ TreeDepthT operator()(size_type level) const
  {
    static_assert(sizeof(TreeDepthT) < sizeof(size_type));
    if (level < static_cast<size_type>(cuda::std::numeric_limits<TreeDepthT>::min()) ||
        level > static_cast<size_type>(cuda::std::numeric_limits<TreeDepthT>::max())) {
      cuda::atomic_ref<int32_t, cuda::thread_scope_device> flag{*depth_out_of_range};
      if (!flag.load(cuda::std::memory_order_relaxed)) {
        flag.store(true, cuda::std::memory_order_relaxed);
      }
    }
    return static_cast<TreeDepthT>(level);
  }
};

/**
 * @brief Returns stable sorted keys and its sorted order
 *
 * Uses cub stable radix sort. The order is internally generated, hence it saves a copy and memory.
 * Since the key and order is returned, using double buffer helps to avoid extra copy to user
 * provided output iterator.
 *
 * @tparam IndexType sorted order type
 * @tparam KeyType key type
 * @param keys keys to sort
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Sorted keys and indices producing that sorted order
 */
template <typename IndexType = size_t, typename KeyType>
std::pair<rmm::device_uvector<KeyType>, rmm::device_uvector<IndexType>> stable_sorted_key_order(
  cudf::device_span<KeyType const> keys, rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // Determine temporary device storage requirements
  rmm::device_uvector<KeyType> keys_buffer1(keys.size(), stream);
  rmm::device_uvector<KeyType> keys_buffer2(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer1(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer2(keys.size(), stream);
  cub::DoubleBuffer<IndexType> order_buffer(order_buffer1.data(), order_buffer2.data());
  cub::DoubleBuffer<KeyType> keys_buffer(keys_buffer1.data(), keys_buffer2.data());
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes, keys_buffer, order_buffer, keys.size());
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  thrust::copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               keys.begin(),
               keys.end(),
               keys_buffer1.begin());
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   order_buffer1.begin(),
                   order_buffer1.end());

  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_buffer,
                                  order_buffer,
                                  keys.size(),
                                  0,
                                  sizeof(KeyType) * 8,
                                  stream.value());

  return std::pair{keys_buffer.Current() == keys_buffer1.data() ? std::move(keys_buffer1)
                                                                : std::move(keys_buffer2),
                   order_buffer.Current() == order_buffer1.data() ? std::move(order_buffer1)
                                                                  : std::move(order_buffer2)};
}

/**
 * @brief Propagate parent node from first sibling to other siblings.
 *
 * @param node_levels Node levels of each node
 * @param parent_node_ids parent node ids initialized for first child of each push node,
 *                       and other siblings are initialized to -1.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void propagate_first_sibling_to_other(cudf::device_span<TreeDepthT const> node_levels,
                                      cudf::device_span<NodeIndexT> parent_node_ids,
                                      rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  auto [sorted_node_levels, sorted_order] = stable_sorted_key_order<size_type>(node_levels, stream);
  // instead of gather, using permutation_iterator, which is ~17% faster

  thrust::inclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    sorted_node_levels.begin(),
    sorted_node_levels.end(),
    cuda::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
    cuda::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
    cuda::std::equal_to<TreeDepthT>{},
    cuda::maximum<NodeIndexT>{});
}

// Generates a tree representation of the given tokens, token_indices.
tree_meta_t get_tree_representation(device_span<PdaTokenT const> tokens,
                                    device_span<SymbolOffsetT const> token_indices,
                                    bool is_strict_nested_boundaries,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Whether a token does represent a node in the tree representation
  auto const is_node = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  };

  // Whether the token pops from the parent node stack
  auto const does_pop = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::StructMemberEnd:
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto const does_push = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::FieldNameBegin:
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  // Look for ErrorBegin and report the point of error.
  if (auto const error_count =
        thrust::count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      tokens.begin(),
                      tokens.end(),
                      token_t::ErrorBegin);
      error_count > 0) {
    auto const error_location =
      thrust::find(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   tokens.begin(),
                   tokens.end(),
                   token_t::ErrorBegin);
    auto error_index = cudf::detail::make_host_vector<SymbolOffsetT>(
      device_span<SymbolOffsetT const>{
        token_indices.data() + cuda::std::distance(tokens.begin(), error_location), 1},
      stream);

    CUDF_FAIL("JSON Parser encountered an invalid format at location " +
              std::to_string(error_index[0]));
  }

  auto const num_tokens = tokens.size();
  auto const num_nodes  = cudf::detail::count_if(tokens.begin(), tokens.end(), is_node, stream);

  // Node levels: transform_exclusive_scan, copy_if.
  rmm::device_uvector<TreeDepthT> node_levels(num_nodes, stream, mr);
  {
    rmm::device_uvector<TreeDepthT> token_levels(num_tokens, stream);
    auto const push_pop_it = thrust::make_transform_iterator(
      tokens.begin(),
      cuda::proclaim_return_type<size_type>(
        [does_push, does_pop] __device__(PdaTokenT const token) -> size_type {
          return does_push(token) - does_pop(token);
        }));
    auto depth_out_of_range =
      cudf::detail::device_scalar<int32_t>(0, stream, cudf::get_current_device_resource_ref());
    auto const token_level_output_it = cuda::make_transform_output_iterator(
      token_levels.begin(), checked_token_level_output{depth_out_of_range.data()});
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           push_pop_it,
                           push_pop_it + num_tokens,
                           token_level_output_it,
                           size_type{0});

    auto const node_levels_end = cudf::detail::copy_if(token_levels.begin(),
                                                       token_levels.end(),
                                                       tokens.begin(),
                                                       node_levels.begin(),
                                                       is_node,
                                                       stream);
    CUDF_EXPECTS(
      !depth_out_of_range.value(stream),
      "JSON token nesting depth is outside the supported range for TreeDepthT [" +
        std::to_string(static_cast<size_type>(cuda::std::numeric_limits<TreeDepthT>::min())) +
        ", " +
        std::to_string(static_cast<size_type>(cuda::std::numeric_limits<TreeDepthT>::max())) + "]");
    CUDF_EXPECTS(cuda::std::distance(node_levels.begin(), node_levels_end) ==
                   static_cast<std::ptrdiff_t>(num_nodes),
                 "node level count mismatch");
  }

  // Node parent ids:
  // previous push node_id transform, stable sort by level, segmented scan with Max, reorder.
  rmm::device_uvector<NodeIndexT> parent_node_ids(num_nodes, stream, mr);
  rmm::device_uvector<NodeIndexT> node_token_ids(num_nodes, stream);  // needed for SE, LE later
  // This block of code is generalized logical stack algorithm. TODO: make this a separate function.
  {
    cudf::detail::copy_if_async(cuda::counting_iterator<NodeIndexT>{0},
                                cuda::counting_iterator<NodeIndexT>{0} + num_tokens,
                                tokens.begin(),
                                node_token_ids.begin(),
                                is_node,
                                stream);

    // previous push node_id
    // if previous node is a push, then i-1
    // if previous node is FE, then i-2 (returns FB's index)
    // if previous node is SMB and its previous node is a push, then i-2
    // eg. `{ SMB FB FE VB VE SME` -> `{` index as FB's parent.
    // else -1
    auto const first_childs_parent_token_id = [tokens_gpu =
                                                 tokens.begin()] __device__(auto i) -> NodeIndexT {
      if (i <= 0) { return -1; }
      if (tokens_gpu[i - 1] == token_t::StructBegin or tokens_gpu[i - 1] == token_t::ListBegin) {
        return i - 1;
      } else if (tokens_gpu[i - 1] == token_t::FieldNameEnd) {
        return i - 2;
      } else if (tokens_gpu[i - 1] == token_t::StructMemberBegin and
                 (tokens_gpu[i - 2] == token_t::StructBegin ||
                  tokens_gpu[i - 2] == token_t::ListBegin)) {
        return i - 2;
      } else {
        return -1;
      }
    };

    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      node_token_ids.begin(),
      node_token_ids.end(),
      parent_node_ids.begin(),
      [node_ids_gpu = node_token_ids.begin(), num_nodes, first_childs_parent_token_id] __device__(
        NodeIndexT const tid) -> NodeIndexT {
        auto const pid = first_childs_parent_token_id(tid);
        return pid < 0
                 ? parent_node_sentinel
                 : thrust::lower_bound(thrust::seq, node_ids_gpu, node_ids_gpu + num_nodes, pid) -
                     node_ids_gpu;
        // parent_node_sentinel is -1, useful for segmented max operation below
      });
  }
  // Propagate parent node to siblings from first sibling - inplace.
  propagate_first_sibling_to_other(
    cudf::device_span<TreeDepthT const>{node_levels.data(), node_levels.size()},
    parent_node_ids,
    stream);

  // Node categories: copy_if with transform.
  rmm::device_uvector<NodeT> node_categories(num_nodes, stream, mr);
  auto const node_categories_it =
    cuda::make_transform_output_iterator(node_categories.begin(), token_to_node{});
  auto const node_categories_end =
    cudf::detail::copy_if(tokens.begin(), tokens.end(), node_categories_it, is_node, stream);
  CUDF_EXPECTS(cuda::std::distance(node_categories_it, node_categories_end) ==
                 static_cast<std::ptrdiff_t>(num_nodes),
               "node category count mismatch");

  // Node ranges: copy_if with transform.
  rmm::device_uvector<SymbolOffsetT> node_range_begin(num_nodes, stream, mr);
  rmm::device_uvector<SymbolOffsetT> node_range_end(num_nodes, stream, mr);
  auto const node_range_tuple_it =
    thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  // Whether the tokenizer stage should keep quote characters for string values
  // If the tokenizer keeps the quote characters, they may be stripped during type casting
  constexpr bool include_quote_char = true;
  auto const node_range_out_it      = cuda::make_transform_output_iterator(
    node_range_tuple_it, node_ranges{tokens, token_indices, include_quote_char});

  auto const node_range_out_end = cudf::detail::copy_if(
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{0} + num_tokens,
    node_range_out_it,
    [is_node, tokens_gpu = tokens.begin()] __device__(size_type i) -> bool {
      return is_node(tokens_gpu[i]);
    },
    stream);
  CUDF_EXPECTS(cuda::std::distance(node_range_out_it, node_range_out_end) ==
                 static_cast<std::ptrdiff_t>(num_nodes),
               "node range count mismatch");

  // Extract Struct, List range_end:
  // 1. Extract Struct, List - begin & end separately, their token ids
  // 2. push, pop to get levels
  // 3. copy first child's parent token_id, also translate to node_id
  // 4. propagate to siblings using levels, parent token id. (segmented scan)
  // 5. scatter to node_range_end for only nested end tokens.
  if (is_strict_nested_boundaries) {
    // Whether the token is nested
    auto const is_nested = [] __device__(PdaTokenT const token) -> bool {
      switch (token) {
        case token_t::StructBegin:
        case token_t::StructEnd:
        case token_t::ListBegin:
        case token_t::ListEnd: return true;
        default: return false;
      };
    };
    auto const num_nested = cudf::detail::count_if(tokens.begin(), tokens.end(), is_nested, stream);
    rmm::device_uvector<TreeDepthT> token_levels(num_nested, stream);
    rmm::device_uvector<NodeIndexT> token_id(num_nested, stream);
    rmm::device_uvector<NodeIndexT> parent_node_ids(num_nested, stream);
    auto const push_pop_it = thrust::make_transform_iterator(
      tokens.begin(),
      cuda::proclaim_return_type<cudf::size_type>(
        [] __device__(PdaTokenT const token) -> size_type {
          if (token == token_t::StructBegin or token == token_t::ListBegin) {
            return 1;
          } else if (token == token_t::StructEnd or token == token_t::ListEnd) {
            return -1;
          }
          return 0;
        }));
    // copy_if only struct/list's token levels, token ids, tokens.
    auto zipped_in_it =
      thrust::make_zip_iterator(push_pop_it, cuda::counting_iterator<NodeIndexT>{0});
    auto zipped_out_it = thrust::make_zip_iterator(token_levels.begin(), token_id.begin());
    cudf::detail::copy_if_async(
      zipped_in_it, zipped_in_it + num_tokens, tokens.begin(), zipped_out_it, is_nested, stream);

    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           token_levels.begin(),
                           token_levels.end(),
                           token_levels.begin());

    // Get parent of first child of struct/list begin.
    auto const nested_first_childs_parent_token_id =
      [tokens_gpu = tokens.begin(), token_id = token_id.begin()] __device__(auto i) -> NodeIndexT {
      if (i <= 0) { return -1; }
      auto id = token_id[i - 1];  // current token's predecessor
      if (tokens_gpu[id] == token_t::StructBegin or tokens_gpu[id] == token_t::ListBegin) {
        return id;
      } else {
        return -1;
      }
    };

    // copied L+S tokens, and their token ids, their token levels.
    // initialize first child parent token ids
    // translate token ids to node id using similar binary search.
    thrust::transform(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<NodeIndexT>{0},
      cuda::counting_iterator<NodeIndexT>{0} + num_nested,
      parent_node_ids.begin(),
      [node_ids_gpu = node_token_ids.begin(),
       num_nodes,
       nested_first_childs_parent_token_id] __device__(NodeIndexT const tid) -> NodeIndexT {
        auto const pid = nested_first_childs_parent_token_id(tid);
        // token_ids which are converted to nodes, are stored in node_ids_gpu in order
        // so finding index of token_id in node_ids_gpu will return its node index.
        return pid < 0
                 ? parent_node_sentinel
                 : thrust::lower_bound(thrust::seq, node_ids_gpu, node_ids_gpu + num_nodes, pid) -
                     node_ids_gpu;
        // parent_node_sentinel is -1, useful for segmented max operation below
      });

    // propagate parent node from first sibling to other siblings - inplace.
    propagate_first_sibling_to_other(
      cudf::device_span<TreeDepthT const>{token_levels.data(), token_levels.size()},
      parent_node_ids,
      stream);

    // scatter to node_range_end for only nested end tokens.
    auto token_indices_it =
      thrust::make_permutation_iterator(token_indices.begin(), token_id.begin());
    auto nested_node_range_end_it =
      thrust::make_transform_output_iterator(node_range_end.begin(), [] __device__(auto i) {
        // add +1 to include end symbol.
        return i + 1;
      });
    auto stencil = thrust::make_transform_iterator(token_id.begin(), is_nested_end{tokens.data()});
    thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       token_indices_it,
                       token_indices_it + num_nested,
                       parent_node_ids.begin(),
                       stencil,
                       nested_node_range_end_it);
  }

  return {std::move(node_categories),
          std::move(parent_node_ids),
          std::move(node_levels),
          std::move(node_range_begin),
          std::move(node_range_end)};
}

// Return field node ids after unicode decoding of field names and matching them to same field names
std::pair<size_t, rmm::device_uvector<size_type>> remapped_field_nodes_after_unicode_decode(
  device_span<SymbolT const> d_input,
  tree_meta_t const& d_tree,
  device_span<size_type const> keys,
  rmm::cuda_stream_view stream)
{
  size_t num_keys = keys.size();
  if (num_keys == 0) { return {num_keys, rmm::device_uvector<size_type>(num_keys, stream)}; }
  rmm::device_uvector<size_type> offsets(num_keys, stream);
  rmm::device_uvector<size_type> lengths(num_keys, stream);
  auto offset_length_it = thrust::make_zip_iterator(offsets.begin(), lengths.begin());
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    keys.begin(),
                    keys.end(),
                    offset_length_it,
                    [node_range_begin = d_tree.node_range_begin.data(),
                     node_range_end   = d_tree.node_range_end.data()] __device__(auto key) {
                      return cuda::std::make_tuple(node_range_begin[key],
                                                   node_range_end[key] - node_range_begin[key]);
                    });
  cudf::io::parse_options_view opt{',', '\n', '\0', '.'};
  opt.keepquotes = true;

  auto utf8_decoded_fields = parse_data(d_input.data(),
                                        offset_length_it,
                                        num_keys,
                                        data_type{type_id::STRING},
                                        rmm::device_buffer{},
                                        0,
                                        opt,
                                        stream,
                                        cudf::get_current_device_resource_ref());
  // hash using iter, create a hashmap for 0-num_keys.
  // insert and find. -> array
  // store to static_map with keys as field key[index], and values as key[array[index]]

  auto str_view       = strings_column_view{utf8_decoded_fields->view()};
  auto const char_ptr = str_view.chars_begin(stream);
  // Offsets may be INT32 or INT64 depending on the char buffer size, so they cannot be read as
  // a typed pointer.
  auto const offset_ptr =
    cudf::detail::offsetalator_factory::make_input_iterator(str_view.offsets(), str_view.offset());

  // String hasher
  auto const d_hasher = cuda::proclaim_return_type<
    typename cudf::hashing::detail::default_hash<cudf::string_view>::result_type>(
    [char_ptr, offset_ptr] __device__(auto node_id) {
      auto const field_name =
        cudf::string_view(char_ptr + offset_ptr[node_id],
                          static_cast<size_type>(offset_ptr[node_id + 1] - offset_ptr[node_id]));
      return cudf::hashing::detail::default_hash<cudf::string_view>{}(field_name);
    });
  auto const d_equal = [char_ptr, offset_ptr] __device__(auto node_id1, auto node_id2) {
    auto const field_name1 =
      cudf::string_view(char_ptr + offset_ptr[node_id1],
                        static_cast<size_type>(offset_ptr[node_id1 + 1] - offset_ptr[node_id1]));
    auto const field_name2 =
      cudf::string_view(char_ptr + offset_ptr[node_id2],
                        static_cast<size_type>(offset_ptr[node_id2 + 1] - offset_ptr[node_id2]));
    return field_name1 == field_name2;
  };

  using hasher_type                             = decltype(d_hasher);
  constexpr size_type empty_node_index_sentinel = -1;
  auto key_set                                  = cuco::static_set{cuco::extent{num_keys},
                                  cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                  cuco::empty_key{empty_node_index_sentinel},
                                  d_equal,
                                  cuco::linear_probing<1, hasher_type>{d_hasher},
                                                                   {},
                                                                   {},
                                  rmm::mr::polymorphic_allocator<char>{},
                                  stream.value()};
  auto const counting_iter                      = cuda::counting_iterator<size_type>{0};
  rmm::device_uvector<size_type> found_keys(num_keys, stream);
  key_set.insert_and_find_async(counting_iter,
                                counting_iter + num_keys,
                                found_keys.begin(),
                                cuda::make_discard_iterator(),
                                stream.value());
  // set.size will synchronize the stream before return.
  return {key_set.size(stream), std::move(found_keys)};
}

/**
 * @brief Generates unique node_type id for each node.
 * Field nodes with the same name are assigned the same node_type id.
 * List, Struct, and String nodes are assigned their category values as node_type ids.
 *
 * All inputs and outputs are in node_id order.
 * @param d_input JSON string in device memory
 * @param d_tree Tree representation of the JSON
 * @param is_enabled_experimental Whether to enable experimental features such as
 * utf8 field name support
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Vector of node_type ids
 */
rmm::device_uvector<size_type> hash_node_type_with_field_name(device_span<SymbolT const> d_input,
                                                              tree_meta_t const& d_tree,
                                                              bool is_enabled_experimental,
                                                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto const num_nodes = d_tree.node_categories.size();
  auto const num_fields =
    thrust::count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  d_tree.node_categories.begin(),
                  d_tree.node_categories.end(),
                  node_t::NC_FN);

  auto const d_hasher = cuda::proclaim_return_type<
    typename cudf::hashing::detail::default_hash<cudf::string_view>::result_type>(
    [d_input          = d_input.data(),
     node_range_begin = d_tree.node_range_begin.data(),
     node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id) {
      auto const field_name = cudf::string_view(
        d_input + node_range_begin[node_id], node_range_end[node_id] - node_range_begin[node_id]);
      return cudf::hashing::detail::default_hash<cudf::string_view>{}(field_name);
    });
  auto const d_equal = [d_input          = d_input.data(),
                        node_range_begin = d_tree.node_range_begin.data(),
                        node_range_end   = d_tree.node_range_end.data()] __device__(auto node_id1,
                                                                                  auto node_id2) {
    auto const field_name1 = cudf::string_view(
      d_input + node_range_begin[node_id1], node_range_end[node_id1] - node_range_begin[node_id1]);
    auto const field_name2 = cudf::string_view(
      d_input + node_range_begin[node_id2], node_range_end[node_id2] - node_range_begin[node_id2]);
    return field_name1 == field_name2;
  };
  // key-value pairs: uses node_id itself as node_type. (unique node_id for a field name due to
  // hashing)
  auto const counting_iter = cuda::counting_iterator<size_type>{0};

  auto const is_field_name_node = [node_categories =
                                     d_tree.node_categories.data()] __device__(auto node_id) {
    return node_categories[node_id] == node_t::NC_FN;
  };

  using hasher_type                             = decltype(d_hasher);
  constexpr size_type empty_node_index_sentinel = -1;
  auto key_set                                  = cuco::static_set{cuco::extent{num_fields},
                                  0.4,  // 40% load factor
                                  cuco::empty_key{empty_node_index_sentinel},
                                  d_equal,
                                  cuco::linear_probing<1, hasher_type>{d_hasher},
                                                                   {},
                                                                   {},
                                  rmm::mr::polymorphic_allocator<char>{},
                                  stream.value()};
  key_set.insert_if_async(counting_iter,
                          counting_iter + num_nodes,
                          cuda::counting_iterator<size_type>{0},  // stencil
                          is_field_name_node,
                          stream.value());

  // experimental feature: utf8 field name support
  // parse_data on field names,
  // rehash it using another map,
  // reassign the reverse map values to new matched node indices.
  auto get_utf8_matched_field_nodes = [&]() {
    auto make_map = [&stream](auto num_keys) {
      using hasher_type3 = cudf::hashing::detail::default_hash<size_type>;
      return cuco::static_map{cuco::extent{static_cast<std::size_t>(num_keys)},
                              1.0,  // 100% load factor
                              cuco::empty_key{empty_node_index_sentinel},
                              cuco::empty_value{empty_node_index_sentinel},
                              {},
                              cuco::linear_probing<1, hasher_type3>{hasher_type3{}},
                              {},
                              {},
                              rmm::mr::polymorphic_allocator<char>{},
                              stream.value()};
    };
    if (!is_enabled_experimental) { return std::pair{false, make_map(size_type{0})}; }
    // get all unique field node ids for utf8 decoding
    auto num_keys = static_cast<size_type>(key_set.size(stream));
    rmm::device_uvector<size_type> keys(num_keys, stream);
    key_set.retrieve_all(keys.data(), stream.value());

    auto [num_unique_fields, found_keys] =
      remapped_field_nodes_after_unicode_decode(d_input, d_tree, keys, stream);

    auto is_need_remap = num_unique_fields != static_cast<std::size_t>(num_keys);
    if (!is_need_remap) { return std::pair{false, make_map(size_type{0})}; }

    // store to static_map with keys as field keys[index], and values as keys[found_keys[index]]
    auto reverse_map        = make_map(num_keys);
    auto matching_keys_iter = cuda::make_permutation_iterator(keys.begin(), found_keys.begin());
    auto pair_iter =
      thrust::make_zip_iterator(cuda::std::make_tuple(keys.begin(), matching_keys_iter));
    reverse_map.insert_async(pair_iter, pair_iter + num_keys, stream);
    return std::pair{is_need_remap, std::move(reverse_map)};
  };
  auto [is_need_remap, reverse_map] = get_utf8_matched_field_nodes();

  auto const get_hash_value =
    [key_set       = key_set.ref(cuco::op::find),
     is_need_remap = is_need_remap,
     rm            = reverse_map.ref(cuco::op::find)] __device__(auto node_id) -> size_type {
    auto const it = key_set.find(node_id);
    if (it != key_set.end() and is_need_remap) {
      auto const it2 = rm.find(*it);
      return (it2 == rm.end()) ? size_type{0} : it2->second;
    }
    return (it == key_set.end()) ? size_type{0} : *it;
  };

  // convert field nodes to node indices, and other nodes to enum value.
  rmm::device_uvector<size_type> node_type(num_nodes, stream);
  thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   node_type.begin(),
                   node_type.end(),
                   [node_categories = d_tree.node_categories.data(),
                    is_field_name_node,
                    get_hash_value] __device__(auto node_id) -> size_type {
                     if (is_field_name_node(node_id))
                       return static_cast<size_type>(NUM_NODE_CLASSES) + get_hash_value(node_id);
                     else
                       return static_cast<size_type>(node_categories[node_id]);
                   });
  return node_type;
}

std::pair<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<NodeIndexT>>
get_array_children_indices(TreeDepthT row_array_children_level,
                           device_span<TreeDepthT const> node_levels,
                           device_span<NodeIndexT const> parent_node_ids,
                           rmm::cuda_stream_view stream)
{
  // array children level: (level 2 for values, level 1 for values-JSONLines format)
  // copy nodes id of level 1's children (level 2)
  // exclusive scan by key (on key their parent_node_id, because we need indices in each row.
  // parent_node_id for each row will be same).
  // -> return their indices and their node id
  auto const num_nodes = node_levels.size();
  auto num_level2_nodes =
    thrust::count(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  node_levels.begin(),
                  node_levels.end(),
                  row_array_children_level);
  rmm::device_uvector<NodeIndexT> level2_nodes(num_level2_nodes, stream);
  rmm::device_uvector<NodeIndexT> level2_indices(num_level2_nodes, stream);
  cudf::detail::copy_if_async(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{num_nodes},
    node_levels.begin(),
    level2_nodes.begin(),
    [row_array_children_level] __device__(auto level) -> bool {
      return level == row_array_children_level;
    },
    stream);
  auto level2_parent_nodes =
    cuda::make_permutation_iterator(parent_node_ids.begin(), level2_nodes.cbegin());
  thrust::exclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    level2_parent_nodes,
    level2_parent_nodes + num_level2_nodes,
    cuda::make_constant_iterator(NodeIndexT{1}),
    level2_indices.begin());
  return std::make_pair(std::move(level2_nodes), std::move(level2_indices));
}

// Two level hashing algorithm
// 1. Convert node_category+fieldname to node_type. (passed as argument)
//   a. Create a hashmap to hash field name and assign unique node id as values.
//   b. Convert the node categories to node types.
//      Node type is defined as node category enum value if it is not a field node,
//      otherwise it is the unique node id assigned by the hashmap (value shifted by #NUM_CATEGORY).
// 2. Set operation on entire path of each node
//   a. Create a hash map with hash of {node_level, node_type} of its node and the entire parent
//      until root.
//   b. While creating hashmap, transform node id to unique node ids that are inserted into the
//      hash map. This mimics set operation with hash map. This unique node ids are set ids.
//   c. Return this converted set ids, which are the hash map keys/values, and unique set ids.
std::pair<rmm::device_uvector<size_type>, rmm::device_uvector<size_type>> hash_node_path(
  device_span<TreeDepthT const> node_levels,
  device_span<size_type const> node_type,
  device_span<NodeIndexT const> parent_node_ids,
  bool is_array_of_arrays,
  bool is_enabled_lines,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const num_nodes = parent_node_ids.size();

  // array of arrays
  NodeIndexT const row_array_children_level = is_enabled_lines ? 1 : 2;
  rmm::device_uvector<size_type> list_indices(0, stream);
  if (is_array_of_arrays) {
    // For array of arrays, level 2 nodes do not have column name (field name).
    // So, we need to generate indices for each level 2 node w.r.t to that row, to uniquely
    // identify each level 2 node as separate column.
    // Example:
    // array of structs: [ { a: 1, b: 2}, { a: 3, b: 4} ]
    //           levels: 0 1 2  3  2  3   1 2  3  2  3
    // array of arrays:  [ [    1,    2], [    3,    4] ]
    //           levels: 0 1    2     2   1    2     2
    // For example, in the above example, we need to generate indices for each level 2 node:
    // array of arrays:  [ [    1,    2], [    3,    4] ]
    //          levels:  0 1    2     2   1    2     2
    //   child indices:         0     1        0     1
    // These indices uniquely identify each column in each row. This is used during hashing for
    // level 2 nodes to generate unique column ids, instead of field name for level 2 nodes.
    auto [level2_nodes, level2_indices] =
      get_array_children_indices(row_array_children_level, node_levels, parent_node_ids, stream);
    // memory usage could be reduced by using different data structure (hashmap)
    // or alternate method to hash it at node_type
    list_indices.resize(num_nodes, stream);
    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    level2_indices.cbegin(),
                    level2_indices.cend(),
                    level2_nodes.cbegin(),
                    list_indices.begin());
  }

  // path compression is not used since extra writes make all map operations slow.
  auto const d_hasher = [node_level      = node_levels.begin(),
                         node_type       = node_type.begin(),
                         parent_node_ids = parent_node_ids.begin(),
                         list_indices    = list_indices.begin(),
                         is_array_of_arrays,
                         row_array_children_level] __device__(auto node_id) {
    auto hash = cudf::hashing::detail::hash_combine(
      cudf::hashing::detail::default_hash<TreeDepthT>{}(node_level[node_id]),
      cudf::hashing::detail::default_hash<size_type>{}(node_type[node_id]));
    node_id = parent_node_ids[node_id];
    // Each node computes its hash by walking from its node up to the root.
    while (node_id != parent_node_sentinel) {
      hash = cudf::hashing::detail::hash_combine(
        hash, cudf::hashing::detail::default_hash<TreeDepthT>{}(node_level[node_id]));
      hash = cudf::hashing::detail::hash_combine(
        hash, cudf::hashing::detail::default_hash<size_type>{}(node_type[node_id]));
      if (is_array_of_arrays and node_level[node_id] == row_array_children_level)
        hash = cudf::hashing::detail::hash_combine(hash, list_indices[node_id]);
      node_id = parent_node_ids[node_id];
    }
    return hash;
  };

  rmm::device_uvector<hash_value_type> node_hash(num_nodes, stream);
  thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   node_hash.begin(),
                   node_hash.end(),
                   d_hasher);
  auto const d_hashed_cache = [node_hash = node_hash.begin()] __device__(auto node_id) {
    return node_hash[node_id];
  };

  auto const d_equal = [node_level      = node_levels.begin(),
                        node_type       = node_type.begin(),
                        parent_node_ids = parent_node_ids.begin(),
                        is_array_of_arrays,
                        row_array_children_level,
                        list_indices = list_indices.begin(),
                        d_hashed_cache] __device__(auto node_id1, auto node_id2) {
    if (node_id1 == node_id2) return true;
    if (d_hashed_cache(node_id1) != d_hashed_cache(node_id2)) return false;
    auto const is_equal_level =
      [node_level, node_type, is_array_of_arrays, row_array_children_level, list_indices](
        auto node_id1, auto node_id2) {
        if (node_id1 == node_id2) return true;
        auto const is_level2_equal = [&]() {
          if (!is_array_of_arrays) return true;
          return node_level[node_id1] != row_array_children_level or
                 list_indices[node_id1] == list_indices[node_id2];
        }();
        return node_level[node_id1] == node_level[node_id2] and
               node_type[node_id1] == node_type[node_id2] and is_level2_equal;
      };
    // if both nodes have same node types at all levels, it will check until it has common parent
    // or root.
    while (node_id1 != parent_node_sentinel and node_id2 != parent_node_sentinel and
           node_id1 != node_id2 and is_equal_level(node_id1, node_id2)) {
      node_id1 = parent_node_ids[node_id1];
      node_id2 = parent_node_ids[node_id2];
    }
    return node_id1 == node_id2;
  };

  constexpr size_type empty_node_index_sentinel = -1;
  using hasher_type                             = decltype(d_hashed_cache);

  auto key_set = cuco::static_set{cuco::extent{num_nodes},
                                  cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                  cuco::empty_key<cudf::size_type>{empty_node_index_sentinel},
                                  d_equal,
                                  cuco::linear_probing<1, hasher_type>{d_hashed_cache},
                                  {},
                                  {},
                                  rmm::mr::polymorphic_allocator<char>{},
                                  stream.value()};

  // insert and convert node ids to unique set ids
  auto nodes_itr         = cuda::counting_iterator<size_type>{0};
  auto const num_columns = key_set.insert(nodes_itr, nodes_itr + num_nodes, stream.value());

  rmm::device_uvector<size_type> unique_keys(num_columns, stream);
  rmm::device_uvector<size_type> col_id(num_nodes, stream, mr);
  key_set.find_async(nodes_itr, nodes_itr + num_nodes, col_id.begin(), stream.value());
  std::ignore = key_set.retrieve_all(unique_keys.begin(), stream.value());

  return {std::move(col_id), std::move(unique_keys)};
}

/**
 * @brief Generates column id and parent column id for each node
 *
 * 1. Generate col_id:
 *    a. Set operation on entire path of each node, translate each node id to set id.
 *       (two level hashing)
 *    b. gather unique set ids.
 *    c. sort and use binary search to generate column ids.
 *    d. Translate parent node ids to parent column ids.
 *
 * All inputs and outputs are in node_id order.
 * @param d_input JSON string in device memory
 * @param d_tree Tree representation of the JSON
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param is_enabled_lines Whether the input is a line-delimited JSON
 * @param is_enabled_experimental Whether the experimental feature is enabled such as
 * utf8 field name support
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return column_id, parent_column_id
 */
std::pair<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<NodeIndexT>> generate_column_id(
  device_span<SymbolT const> d_input,
  tree_meta_t const& d_tree,
  bool is_array_of_arrays,
  bool is_enabled_lines,
  bool is_enabled_experimental,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const num_nodes = d_tree.node_categories.size();

  // Two level hashing:
  //   one for field names -> node_type and,
  //   another for {node_level, node_category} + field hash for the entire path
  //    which is {node_level, node_type} recursively using parent_node_id
  auto [col_id, unique_keys] = [&]() {
    // Convert node_category + field_name to node_type.
    rmm::device_uvector<size_type> node_type =
      hash_node_type_with_field_name(d_input, d_tree, is_enabled_experimental, stream);

    // hash entire path from node to root.
    return hash_node_path(d_tree.node_levels,
                          node_type,
                          d_tree.parent_node_ids,
                          is_array_of_arrays,
                          is_enabled_lines,
                          stream,
                          mr);
  }();

  thrust::sort(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               unique_keys.begin(),
               unique_keys.end());
  thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      unique_keys.begin(),
                      unique_keys.end(),
                      col_id.begin(),
                      col_id.end(),
                      col_id.begin());

  rmm::device_uvector<size_type> parent_col_id(num_nodes, stream, mr);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    d_tree.parent_node_ids.begin(),
                    d_tree.parent_node_ids.end(),
                    parent_col_id.begin(),
                    [col_id = col_id.begin()] __device__(auto node_id) {
                      return node_id >= 0 ? col_id[node_id] : parent_node_sentinel;
                    });

  return {std::move(col_id), std::move(parent_col_id)};
}

/**
 * @brief Computes row indices of each node in the hierarchy.
 * 2. Generate row_offset.
 *   a. Extract only list children
 *   b. stable_sort by parent_col_id.
 *   c. scan_by_key {parent_col_id} (done only on nodes who's parent is list)
 *   d. propagate to non-list leaves from parent list node by recursion
 *
 * pre-condition:
 *  d_tree.node_categories, d_tree.parent_node_ids, parent_col_id are in order of node_id.
 * post-condition: row_offsets is in order of node_id.
 *  parent_col_id is moved and reused inside this function.
 * @param parent_col_id parent node's column id
 * @param d_tree Tree representation of the JSON string
 * @param is_array_of_arrays Whether the tree is an array of arrays
 * @param is_enabled_lines Whether the input is a line-delimited JSON
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return row_offsets
 */
rmm::device_uvector<size_type> compute_row_offsets(rmm::device_uvector<NodeIndexT>&& parent_col_id,
                                                   tree_meta_t const& d_tree,
                                                   bool is_array_of_arrays,
                                                   bool is_enabled_lines,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const num_nodes = d_tree.node_categories.size();

  rmm::device_uvector<size_type> scatter_indices(num_nodes, stream);
  thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   scatter_indices.begin(),
                   scatter_indices.end());

  // array of arrays
  NodeIndexT const row_array_parent_level = is_enabled_lines ? 0 : 1;
  // condition is true if parent is not a list, or sentinel/root
  // Special case to return true if parent is a list and is_array_of_arrays is true
  auto is_non_list_parent = [node_categories = d_tree.node_categories.begin(),
                             node_levels     = d_tree.node_levels.begin(),
                             is_array_of_arrays,
                             row_array_parent_level] __device__(auto pnid) {
    return !(pnid == parent_node_sentinel ||
             node_categories[pnid] == NC_LIST &&
               (!is_array_of_arrays || node_levels[pnid] != row_array_parent_level));
  };

  // Extract only list children. (nodes who's parent is a list/root)
  auto const list_parent_end =
    thrust::remove_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      thrust::make_zip_iterator(parent_col_id.begin(), scatter_indices.begin()),
                      thrust::make_zip_iterator(parent_col_id.end(), scatter_indices.end()),
                      d_tree.parent_node_ids.begin(),
                      is_non_list_parent);
  auto const num_list_parent = cuda::std::distance(
    thrust::make_zip_iterator(parent_col_id.begin(), scatter_indices.begin()), list_parent_end);

  thrust::stable_sort_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    parent_col_id.begin(),
    parent_col_id.begin() + num_list_parent,
    scatter_indices.begin());

  rmm::device_uvector<size_type> row_offsets(num_nodes, stream, mr);
  // TODO is it possible to generate list child_offsets too here?
  // write only 1st child offset to parent node id child_offsets?
  thrust::exclusive_scan_by_key(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    parent_col_id.begin(),
    parent_col_id.begin() + num_list_parent,
    cuda::make_constant_iterator<size_type>(1),
    row_offsets.begin());

  // Using scatter instead of sort.
  auto& temp_storage = parent_col_id;  // reuse parent_col_id as temp storage
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  row_offsets.begin(),
                  row_offsets.begin() + num_list_parent,
                  scatter_indices.begin(),
                  temp_storage.begin());
  row_offsets = std::move(temp_storage);

  // Propagate row offsets to non-list leaves from list's immediate children node by recursion
  thrust::transform_if(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{num_nodes},
    row_offsets.begin(),
    [node_categories = d_tree.node_categories.data(),
     parent_node_ids = d_tree.parent_node_ids.begin(),
     row_offsets     = row_offsets.begin(),
     is_non_list_parent] __device__(size_type node_id) {
      auto parent_node_id = parent_node_ids[node_id];
      while (is_non_list_parent(parent_node_id)) {
        node_id        = parent_node_id;
        parent_node_id = parent_node_ids[parent_node_id];
      }
      return row_offsets[node_id];
    },
    [node_categories = d_tree.node_categories.data(),
     parent_node_ids = d_tree.parent_node_ids.begin(),
     is_non_list_parent] __device__(size_type node_id) {
      auto const parent_node_id = parent_node_ids[node_id];
      return is_non_list_parent(parent_node_id);
    });
  return row_offsets;
}

// This algorithm assigns a unique column id to each node in the tree.
// The row offset is the row index of the node in that column id.
// Algorithm:
// 1. Generate col_id:
//   a. Set operation on entire path of each node, translate each node id to set id.
//   b. gather unique set ids.
//   c. sort and use binary search to generate column ids.
//   d. Translate parent node ids to parent column ids.
// 2. Generate row_offset.
//   a. filter only list children
//   a. stable_sort by parent_col_id.
//   b. scan_by_key {parent_col_id} (done only on nodes whose parent is a list)
//   c. propagate to non-list leaves from parent list node by recursion
std::tuple<rmm::device_uvector<NodeIndexT>, rmm::device_uvector<size_type>>
records_orient_tree_traversal(device_span<SymbolT const> d_input,
                              tree_meta_t const& d_tree,
                              bool is_array_of_arrays,
                              bool is_enabled_lines,
                              bool is_enabled_experimental,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto [new_col_id, new_parent_col_id] = generate_column_id(
    d_input, d_tree, is_array_of_arrays, is_enabled_lines, is_enabled_experimental, stream, mr);

  auto row_offsets = compute_row_offsets(
    std::move(new_parent_col_id), d_tree, is_array_of_arrays, is_enabled_lines, stream, mr);
  return std::tuple{std::move(new_col_id), std::move(row_offsets)};
}

}  // namespace detail
}  // namespace cudf::io::json
