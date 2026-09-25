/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort.hpp"
#include "sort_radix.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cudf {
namespace detail {

/**
 * @brief Extracts the first bytes of a string as an unsigned big-endian integer.
 *
 * Zero padding is order preserving. It can create a prefix tie between a short string and a
 * longer string containing zero bytes; string lengths resolve that case.
 */
template <typename PrefixKey, bool has_nulls>
struct string_prefix_extractor {
  static_assert(std::is_unsigned_v<PrefixKey>);
  static constexpr auto prefix_bytes  = static_cast<size_type>(sizeof(PrefixKey));
  static constexpr auto bits_per_byte = std::numeric_limits<uint8_t>::digits;

  __device__ PrefixKey operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (d_column.is_null(row)) { return 0; }
    }

    auto const string = d_column.element<string_view>(row);
    PrefixKey prefix  = 0;
    for (size_type byte = 0; byte < prefix_bytes; ++byte) {
      prefix <<= bits_per_byte;
      if (byte < string.size_bytes()) {
        prefix |= static_cast<PrefixKey>(static_cast<uint8_t>(string.data()[byte]));
      }
    }
    return prefix;
  }

  column_device_view const d_column;
};

/**
 * @brief String comparator accelerated by a contiguous array of cached prefix keys.
 *
 * This optimization was inspired by Eiger (https://arxiv.org/abs/2607.04489), which optionally
 * caches four-byte prefixes based on runtime prefix-distribution statistics. This implementation
 * instead uses a fixed-width prefix (currently eight bytes) for every nontrivial single-column
 * string sort and does not perform Eiger's runtime profiling or algorithm selection.
 *
 * This implementation also right-pads short strings with zero bytes and resolves the resulting
 * prefix collisions using string lengths. Comparisons tied after a complete prefix resume at the
 * first uncached byte, and nullable and non-nullable inputs use separate comparator
 * specializations.
 */
template <typename PrefixKey, bool has_nulls>
struct string_prefix_comparator {
  static_assert(std::is_unsigned_v<PrefixKey>);
  static constexpr auto prefix_bytes = static_cast<size_type>(sizeof(PrefixKey));

  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if constexpr (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs)};
      bool const rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        return null_compare(lhs_null, rhs_null, null_precedence) ==
               (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
      }
    }

    auto const lhs_prefix = prefixes[lhs];
    auto const rhs_prefix = prefixes[rhs];
    if (lhs_prefix != rhs_prefix) {
      return ascending ? lhs_prefix < rhs_prefix : lhs_prefix > rhs_prefix;
    }

    auto const left_element  = d_column.element<string_view>(lhs);
    auto const right_element = d_column.element<string_view>(rhs);
    auto const left_size     = left_element.size_bytes();
    auto const right_size    = right_element.size_bytes();
    if (left_size < prefix_bytes or right_size < prefix_bytes) {
      // Equal zero-padded prefixes prove that all bytes in the shorter value match and that any
      // represented bytes beyond it are zero. The shorter value is therefore lexicographically
      // smaller, while equal lengths prove equality without rereading either string.
      return ascending ? left_size < right_size : right_size < left_size;
    }

    // Both values contain a complete cached prefix, so resume comparison at the first byte not
    // represented by the key instead of rescanning known-equal bytes.
    auto const left_suffix =
      string_view{left_element.data() + prefix_bytes, left_element.size_bytes() - prefix_bytes};
    auto const right_suffix =
      string_view{right_element.data() + prefix_bytes, right_element.size_bytes() - prefix_bytes};
    return ascending ? left_suffix < right_suffix : right_suffix < left_suffix;
  }

  column_device_view const d_column;
  PrefixKey const* prefixes;
  bool ascending;
  null_order null_precedence{};
};

/**
 * @brief Comparator functor needed for single column sort.
 *
 * @tparam Column element type.
 */
template <typename T>
struct simple_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs)
  {
    if (has_nulls) {
      bool const lhs_null{d_column.is_null(lhs)};
      bool const rhs_null{d_column.is_null(rhs)};
      if (lhs_null || rhs_null) {
        return null_compare(lhs_null, rhs_null, null_precedence) ==
               (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
      }
    }

    auto const left_element  = d_column.element<T>(lhs);
    auto const right_element = d_column.element<T>(rhs);
    return relational_compare(left_element, right_element) ==
           (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
  }
  column_device_view const d_column;
  bool has_nulls;
  bool ascending;
  null_order null_precedence{};
};

template <sort_method method>
struct column_sorted_order_fn {
 private:
  template <typename Comparator>
  void merge_sort(mutable_column_view& indices, Comparator comp, cuda::stream_ref stream)
  {
    auto in_keys   = cuda::counting_iterator<cudf::size_type>{0};
    auto out_keys  = indices.begin<size_type>();
    auto tmp_bytes = std::size_t{0};
    auto sort_keys = [&](void* tmp_stg) {
      if constexpr (method == sort_method::STABLE) {
        return cub::DeviceMergeSort::StableSortKeysCopy(
          tmp_stg, tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.get());
      } else {
        return cub::DeviceMergeSort::SortKeysCopy(
          tmp_stg, tmp_bytes, in_keys, out_keys, indices.size(), comp, stream.get());
      }
    };

    sort_keys(nullptr);
    auto tmp_stg = rmm::device_buffer(tmp_bytes, stream);
    sort_keys(tmp_stg.data());
  }

  template <typename PrefixKey, bool has_nulls>
  void prefix_sorted_order_impl(column_view const& input,
                                column_device_view const& keys,
                                mutable_column_view& indices,
                                bool ascending,
                                null_order null_precedence,
                                cuda::stream_ref stream)
  {
    auto prefixes =
      rmm::device_uvector<PrefixKey>(input.size(), stream, cudf::get_current_device_resource_ref());
    auto rows = cuda::counting_iterator<cudf::size_type>{0};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      rows,
                      rows + input.size(),
                      prefixes.begin(),
                      string_prefix_extractor<PrefixKey, has_nulls>{keys});

    auto comp = string_prefix_comparator<PrefixKey, has_nulls>{
      keys, prefixes.data(), ascending, null_precedence};
    merge_sort(indices, comp, stream);
  }

  template <typename PrefixKey>
  void prefix_sorted_order(column_view const& input,
                           mutable_column_view& indices,
                           bool ascending,
                           null_order null_precedence,
                           cuda::stream_ref stream)
  {
    // A non-null strings column with no chars buffer contains only empty strings. Checking the
    // buffer pointer avoids the host synchronization required to read the terminal offset.
    auto const all_values_equal =
      not input.has_nulls() and strings_column_view{input}.chars_begin(stream) == nullptr;
    if (input.size() < 2 or input.null_count() == input.size() or all_values_equal) {
      thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       indices.begin<size_type>(),
                       indices.end<size_type>(),
                       size_type{0});
      return;
    }

    auto keys = column_device_view::create(input, stream);
    if (input.has_nulls()) {
      prefix_sorted_order_impl<PrefixKey, true>(
        input, *keys, indices, ascending, null_precedence, stream);
    } else {
      prefix_sorted_order_impl<PrefixKey, false>(
        input, *keys, indices, ascending, null_precedence, stream);
    }
  }

 public:
  /**
   * @brief Sorts a single column with a relationally comparable type.
   *
   * This is used when a comparator is required.
   *
   * @param input Column to sort
   * @param indices Output sorted indices
   * @param ascending True if sort order is ascending
   * @param null_precedence How null rows are to be ordered
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  template <typename T>
  void sorted_order(column_view const& input,
                    mutable_column_view& indices,
                    bool ascending,
                    null_order null_precedence,
                    cuda::stream_ref stream)
  {
    if constexpr (std::is_same_v<T, string_view>) {
      prefix_sorted_order<uint64_t>(input, indices, ascending, null_precedence, stream);
      return;
    }

    auto keys = column_device_view::create(input, stream);
    auto comp = simple_comparator<T>{*keys, input.has_nulls(), ascending, null_precedence};
    merge_sort(indices, comp, stream);
  }

  template <typename T>
    requires(cudf::is_relationally_comparable<T, T>() and not cudf::is_dictionary<T>())
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream)
  {
    sorted_order<T>(input, indices, ascending, null_precedence, stream);
  }

  template <typename T>
    requires(not cudf::is_relationally_comparable<T, T>())
  void operator()(column_view const&, mutable_column_view&, bool, null_order, cuda::stream_ref)
  {
    CUDF_FAIL("Column type must be relationally comparable");
  }

  template <typename T>
    requires(is_dictionary<T>())
  void operator()(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream)
  {
    auto const keys = dictionary_column_view(input).keys();
    // For the keys we do an arg-sort of arg-sort to get the rank and use that as a map
    // to sort the indices in rank order.
    // First, get sorted-order of just the keys (slow but expect keys.size <<< indices.size)
    auto temp_mr = cudf::get_current_device_resource_ref();
    auto ordered_indices =
      cudf::detail::sorted_order<method>(keys, order::ASCENDING, null_precedence, stream, temp_mr);
    // Now, sort the ordered indices to get their ordered positions (very fast integer sort)
    ordered_indices = cudf::detail::sorted_order<method>(
      ordered_indices->view(), order::ASCENDING, null_precedence, stream, temp_mr);
    // And use the result as a map over the dictionary indices
    auto map = ordered_indices->view().template data<size_type>();
    auto itr = cudf::detail::indexalator_factory::make_input_iterator(
      dictionary_column_view(input).indices());
    auto mapped_indices = rmm::device_uvector<size_type>(input.size(), stream);
    thrust::gather(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   itr,
                   itr + input.size(),
                   map,
                   mapped_indices.begin());

    // Finally, sort-order the dictionary indices using mapped values
    auto mapped_view = column_view(data_type{type_to_id<size_type>()},
                                   input.size(),
                                   mapped_indices.data(),
                                   input.null_mask(),
                                   input.null_count());
    // these should be very fast since they are sorting integers
    if (input.has_nulls()) {
      sorted_order<size_type>(mapped_view, indices, ascending, null_precedence, stream);
    } else {
      sorted_order_radix(mapped_view, indices, ascending, stream);
    }
  }
};

}  // namespace detail
}  // namespace cudf
