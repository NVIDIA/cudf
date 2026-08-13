/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "join/join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/search.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <cuda/iterator>

namespace cudf::detail {

using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

/**
 * @brief An hasher adapter wrapping both haystack hasher and needles hasher
 */
template <typename HaystackHasher, typename NeedleHasher>
struct hasher_adapter {
  hasher_adapter(HaystackHasher const& haystack_hasher, NeedleHasher const& needle_hasher)
    : _haystack_hasher{haystack_hasher}, _needle_hasher{needle_hasher}
  {
  }

  __device__ constexpr auto operator()(lhs_index_type idx) const noexcept
  {
    return _needle_hasher(static_cast<size_type>(idx));
  }

  __device__ constexpr auto operator()(rhs_index_type idx) const noexcept
  {
    return _haystack_hasher(static_cast<size_type>(idx));
  }

 private:
  HaystackHasher const _haystack_hasher;
  NeedleHasher const _needle_hasher;
};

/**
 * @brief An comparator adapter wrapping both self comparator and two table comparator
 */
template <typename SelfEqual, typename TwoTableEqual>
struct comparator_adapter {
  comparator_adapter(SelfEqual const& self_equal, TwoTableEqual const& two_table_equal)
    : _self_equal{self_equal}, _two_table_equal{two_table_equal}
  {
  }

  __device__ constexpr auto operator()(rhs_index_type lhs_index,
                                       rhs_index_type rhs_index) const noexcept
  {
    auto const lhs = static_cast<size_type>(lhs_index);
    auto const rhs = static_cast<size_type>(rhs_index);

    return _self_equal(lhs, rhs);
  }

  __device__ constexpr auto operator()(lhs_index_type lhs_index,
                                       rhs_index_type rhs_index) const noexcept
  {
    return _two_table_equal(lhs_index, rhs_index);
  }

 private:
  SelfEqual const _self_equal;
  TwoTableEqual const _two_table_equal;
};

/**
 * @brief Build a row bitmask for the input table.
 *
 * The output bitmask will have invalid bits corresponding to the input rows having nulls (at
 * any nested level) and vice versa.
 *
 * @param input The input table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of pointer to the output bitmask and the buffer containing the bitmask
 */
std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream);

/**
 * @brief Helper function to perform the contains operation using a hash set
 *
 * @tparam Comparator Type of the equality comparator
 * @tparam ProbingScheme Type of the probing scheme
 *
 * @param haystack The haystack table view
 * @param needles The needles table view
 * @param haystack_has_nulls Flag indicating whether haystack has nulls
 * @param needles_has_nulls Flag indicating whether needles has nulls
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param d_equal The equality comparator
 * @param probing_scheme The probing scheme for the hash set
 * @param contained The output vector to store results
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <typename Comparator, typename ProbingScheme>
void perform_contains(table_view const& haystack,
                      table_view const& needles,
                      bool haystack_has_nulls,
                      bool needles_has_nulls,
                      null_equality compare_nulls,
                      Comparator const& d_equal,
                      ProbingScheme const& probing_scheme,
                      rmm::device_uvector<bool>& contained,
                      rmm::cuda_stream_view stream)
{
  auto const haystack_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, cuda::proclaim_return_type<rhs_index_type>([] __device__(auto idx) {
      return rhs_index_type{idx};
    }));

  auto const needles_iter = cudf::detail::make_counting_transform_iterator(
    size_type{0}, cuda::proclaim_return_type<lhs_index_type>([] __device__(auto idx) {
      return lhs_index_type{idx};
    }));

  auto set = cuco::static_set{cuco::extent{static_cast<std::size_t>(haystack.num_rows())},
                              cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                              cuco::empty_key{rhs_index_type{-1}},
                              d_equal,
                              probing_scheme,
                              {},
                              {},
                              rmm::mr::polymorphic_allocator<char>{},
                              stream.value()};

  if (haystack_has_nulls && compare_nulls == null_equality::UNEQUAL) {
    auto const bitmask_buffer_and_ptr = build_row_bitmask(haystack, stream);
    auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

    // If the haystack table has nulls but they are compared unequal, don't insert them.
    // Otherwise, it was known to cause performance issue:
    // - https://github.com/rapidsai/cudf/pull/6943
    // - https://github.com/rapidsai/cudf/pull/8277
    set.insert_if_async(haystack_iter,
                        haystack_iter + haystack.num_rows(),
                        cuda::counting_iterator<size_type>{0},  // stencil
                        row_is_valid{row_bitmask_ptr},
                        stream.value());
  } else {
    set.insert_async(haystack_iter, haystack_iter + haystack.num_rows(), stream.value());
  }

  if (needles_has_nulls && compare_nulls == null_equality::UNEQUAL) {
    auto const bitmask_buffer_and_ptr = build_row_bitmask(needles, stream);
    auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
    set.contains_if_async(needles_iter,
                          needles_iter + needles.num_rows(),
                          cuda::counting_iterator<size_type>{0},  // stencil
                          row_is_valid{row_bitmask_ptr},
                          contained.begin(),
                          stream.value());
  } else {
    set.contains_async(
      needles_iter, needles_iter + needles.num_rows(), contained.begin(), stream.value());
  }
}

/**
 * @brief Invokes perform_contains with desired comparators based on the specified `compare_nans`
 * parameter
 *
 * @tparam HasNested Flag indicating whether there are nested columns in haystack or needles
 * @tparam Hasher Type of device hash function
 *
 * @param haystack The haystack table view
 * @param needles The needles table view
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param haystack_has_nulls Flag indicating whether haystack has nulls or not
 * @param needles_has_nulls Flag indicating whether needles has nulls or not
 * @param has_any_nulls Flag indicating whether there are nested nulls is either haystack or needles
 * @param self_equal Self table comparator
 * @param two_table_equal Two table comparator
 * @param d_hasher Device hash functor
 * @param contained The output vector to store results
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
template <bool HasNested, typename Hasher>
void dispatch_nan_comparator(table_view const& haystack,
                             table_view const& needles,
                             null_equality compare_nulls,
                             nan_equality compare_nans,
                             bool haystack_has_nulls,
                             bool needles_has_nulls,
                             bool has_any_nulls,
                             cudf::detail::row::equality::self_comparator self_equal,
                             cudf::detail::row::equality::two_table_comparator two_table_equal,
                             Hasher const& d_hasher,
                             rmm::device_uvector<bool>& contained,
                             rmm::cuda_stream_view stream)
{
  // Distinguish probing scheme CG sizes between nested and flat types for better performance
  auto const probing_scheme = [&]() {
    if constexpr (HasNested) {
      return cuco::linear_probing<4, Hasher>{d_hasher};
    } else {
      return cuco::linear_probing<1, Hasher>{d_hasher};
    }
  }();

  if (compare_nans == nan_equality::ALL_EQUAL) {
    using nan_equal_comparator =
      cudf::detail::row::equality::nan_equal_physical_equality_comparator;
    auto const d_self_equal = self_equal.equal_to<HasNested>(
      nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, nan_equal_comparator{});
    auto const d_two_table_equal = two_table_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_any_nulls}, compare_nulls, nan_equal_comparator{});
    auto const d_equal = comparator_adapter{d_self_equal, d_two_table_equal};
    perform_contains(haystack,
                     needles,
                     haystack_has_nulls,
                     needles_has_nulls,
                     compare_nulls,
                     d_equal,
                     probing_scheme,
                     contained,
                     stream);
  } else {
    using nan_unequal_comparator = cudf::detail::row::equality::physical_equality_comparator;
    auto const d_self_equal      = self_equal.equal_to<HasNested>(
      nullate::DYNAMIC{haystack_has_nulls}, compare_nulls, nan_unequal_comparator{});
    auto const d_two_table_equal = two_table_equal.equal_to<HasNested>(
      nullate::DYNAMIC{has_any_nulls}, compare_nulls, nan_unequal_comparator{});
    auto const d_equal = comparator_adapter{d_self_equal, d_two_table_equal};
    perform_contains(haystack,
                     needles,
                     haystack_has_nulls,
                     needles_has_nulls,
                     compare_nulls,
                     d_equal,
                     probing_scheme,
                     contained,
                     stream);
  }
}

}  // namespace cudf::detail
