/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nested_types_extrema_utils.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/extrema.h>

namespace cudf::reduction::simple::detail {

/**
 * @brief An adapter to make a functor's operator() non-inlinable.
 *
 * This is to reduce compile time when compiling heavy binary comparators with thrust algorithms,
 * especially `thrust::reduce` and `thrust::min/max_element`.
 * @tparam Functor The functor to prevent inlining.
 */
template <typename Functor>
struct noinline_adapter_fn {
  Functor f;
  noinline_adapter_fn(Functor&& f_) : f{std::forward<Functor>(f_)} {}
  template <typename... Args>
  [[nodiscard]] __attribute__((noinline)) __device__ auto operator()(Args&&... args) const
    -> decltype(f(std::forward<Args>(args)...))
  {
    return f(std::forward<Args>(args)...);
  }
};

/**
 * @brief Ordering for arithmetic ARGMIN/ARGMAX that selects the first NaN
 *
 * NaN propagates as in the IEEE 754-2019 minimum/maximum operations: NaN is ordered before every
 * other value for ARGMIN and after every other value for ARGMAX. This is a strict weak ordering,
 * so `thrust::min_element`/`thrust::max_element` return the first NaN. `is_nan_bits` avoids FP64
 * instructions (see device_operators.cuh).
 */
template <aggregation::Kind K>
struct arithmetic_extremum_less {
  template <typename T>
  __device__ bool operator()(T const& lhs, T const& rhs) const
  {
    if constexpr (cuda::std::is_floating_point_v<T>) {
      auto const lhs_nan = cudf::detail::is_nan_bits(lhs);
      auto const rhs_nan = cudf::detail::is_nan_bits(rhs);
      if (lhs_nan || rhs_nan) {
        return K == aggregation::ARGMIN ? (lhs_nan && !rhs_nan) : (!lhs_nan && rhs_nan);
      }
    }
    return lhs < rhs;
  }
};

/**
 * @brief Maps a floating-point row to an integer key that orders it for TOTALLY_ORDERED ARGMIN
 *
 * With `nan_semantics::TOTALLY_ORDERED`, NaN is greater than every other value, and nulls are
 * ordered after NaN so that a NaN row is selected only when every valid row is NaN. -0 and +0 map
 * to the same key, so equal values select the smaller row index as with UNORDERED ARGMIN.
 */
template <typename T>
struct totally_ordered_argmin_key_fn {
  using key_type = cudf::detail::float_bits_t<T>;

  __device__ key_type operator()(T value, bool is_valid = true) const
  {
    auto constexpr max_key = cuda::std::numeric_limits<key_type>::max();
    if (!is_valid) { return max_key; }
    if (cudf::detail::is_nan_bits(value)) { return max_key - 1; }
    auto const bits = cuda::std::bit_cast<key_type>(value);
    if ((bits & max_key) == 0) { return key_type{0}; }  // -0 and +0
    return cudf::detail::ordered_bits(value);
  }

  __device__ key_type operator()(cuda::std::pair<T, bool> const& p) const
  {
    return (*this)(p.first, p.second);
  }
};

/**
 * @brief Call reduce and return a scalar of the specified type, specialized for ARGMIN and
 * ARGMAX aggregations.
 *
 * @tparam K The aggregation to execute on the column, must be either ARGMIN or ARGMAX.
 */
template <aggregation::Kind K>
class arg_minmax_dispatcher {
  static_assert(K == aggregation::ARGMIN or K == aggregation::ARGMAX,
                "Aggregation kind must be either ARGMIN or ARGMAX");

  nan_semantics semantics_;  ///< NaN ordering for floating-point ARGMIN

  template <typename ElementType>
  static constexpr bool is_supported()
  {
    return !cudf::is_dictionary<ElementType>() && !std::is_same_v<ElementType, void>;
  }

  template <typename InputIterator, typename... Args>
  size_type find_extremum_idx(InputIterator it,
                              size_type size,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr,
                              Args&&... args) const
  {
    auto const temp_mr = mr.get_temporary_mr();
    auto const pos     = [&] {
      if constexpr (K == aggregation::ARGMIN) {
        return thrust::min_element(
          rmm::exec_policy_nosync(stream, temp_mr), it, it + size, std::forward<Args>(args)...);
      } else {
        return thrust::max_element(
          rmm::exec_policy_nosync(stream, temp_mr), it, it + size, std::forward<Args>(args)...);
      }
    }();
    return static_cast<size_type>(cuda::std::distance(it, pos));
  }

  template <typename ElementType>
  [[nodiscard]] size_type find_totally_ordered_argmin(column_view const& input,
                                                      cuda::stream_ref stream,
                                                      cudf::memory_resources mr) const
  {
    auto const key_fn  = totally_ordered_argmin_key_fn<ElementType>{};
    auto const d_input = column_device_view::create(input, stream, mr.get_temporary_mr());
    if (is_dictionary(input.type())) {
      auto const p = cudf::dictionary::detail::make_dictionary_pair_iterator<ElementType>(
        *d_input, input.has_nulls());
      return find_extremum_idx(cuda::transform_iterator(p, key_fn), input.size(), stream, mr);
    }
    if (input.has_nulls()) {
      auto const it = cuda::transform_iterator(d_input->pair_begin<ElementType, true>(), key_fn);
      return find_extremum_idx(it, input.size(), stream, mr);
    }
    auto const it = cuda::transform_iterator(input.begin<ElementType>(), key_fn);
    return find_extremum_idx(it, input.size(), stream, mr);
  }

  template <cudf::concepts::nested ElementType>
  [[nodiscard]] size_type find_arg_minmax(column_view const& input,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr) const
  {
    using Op = std::conditional_t<K == aggregation::ARGMIN,
                                  reduction::detail::op::min,
                                  reduction::detail::op::max>;
    auto const binop_generator =
      reduction::detail::arg_minmax_binop_generator::create<Op>(input, stream, mr);
    return find_extremum_idx(cuda::counting_iterator<cudf::size_type>{0},
                             input.size(),
                             stream,
                             mr,
                             noinline_adapter_fn{binop_generator.less()});
  }

  // This function is used for types such as string, timestamp, fixed point, etc.
  // `input` may be the dictionary column itself (rather than its keys or a decoded column):
  // the lexicographic self_comparator compares `keys[indices[i]]` for dictionary columns
  // directly, so no decoding is required and no assumption is made that the keys are sorted.
  template <typename ElementType>
  [[nodiscard]] size_type find_arg_minmax(column_view const& input,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr) const
    requires(not cudf::is_nested<ElementType>() and not cudf::is_numeric<ElementType>())
  {
    // Nulls are considered "greater" (ARGMIN), or "less" (ARGMAX) than non-null values.
    auto const null_orders =
      std::vector<null_order>{K == aggregation::ARGMIN ? null_order::AFTER : null_order::BEFORE};
    auto const comparator = cudf::detail::row::lexicographic::self_comparator{
      table_view{{input}}, {}, null_orders, stream, mr};
    auto d_comp =
      comparator.less<false /* has_nested_columns */>(nullate::DYNAMIC{input.has_nulls()});
    return find_extremum_idx(cuda::counting_iterator<cudf::size_type>{0},
                             input.size(),
                             stream,
                             mr,
                             noinline_adapter_fn{std::move(d_comp)});
  }

  template <cudf::concepts::arithmetic ElementType>  // integer + floating point numbers
  [[nodiscard]] size_type find_arg_minmax(column_view const& input,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr) const
  {
    // TOTALLY_ORDERED only changes ARGMIN: both semantics select the first NaN for ARGMAX.
    if constexpr (cuda::std::is_floating_point_v<ElementType> && K == aggregation::ARGMIN) {
      if (semantics_ == nan_semantics::TOTALLY_ORDERED) {
        return find_totally_ordered_argmin<ElementType>(input, stream, mr);
      }
    }
    using Op        = std::conditional_t<K == aggregation::ARGMIN,
                                         reduction::detail::op::min,
                                         reduction::detail::op::max>;
    auto const less = arithmetic_extremum_less<K>{};
    // Dictionary keys are not guaranteed to be sorted, so the index of the min/max index does
    // not identify the min/max key. Read `keys[indices[i]]` directly per row via a lazy
    // iterator instead of decoding (and copying) the whole column.
    if (is_dictionary(input.type())) {
      auto const d_dict = column_device_view::create(input, stream, mr.get_temporary_mr());
      if (input.has_nulls()) {
        auto const transformer =
          Op{}.template get_null_replacing_element_transformer<ElementType>();
        auto const p =
          cudf::dictionary::detail::make_dictionary_pair_iterator<ElementType>(*d_dict, true);
        auto const it = cuda::transform_iterator(p, transformer);
        return find_extremum_idx(it, input.size(), stream, mr, less);
      }
      auto const it = cudf::dictionary::detail::make_dictionary_iterator<ElementType>(*d_dict);
      return find_extremum_idx(it, input.size(), stream, mr, less);
    }
    if (input.has_nulls()) {
      auto const d_input     = column_device_view::create(input, stream, mr.get_temporary_mr());
      auto const transformer = Op{}.template get_null_replacing_element_transformer<ElementType>();
      auto const it =
        cuda::transform_iterator(d_input->pair_begin<ElementType, true>(), transformer);
      return find_extremum_idx(it, input.size(), stream, mr, less);
    } else {
      return find_extremum_idx(input.begin<ElementType>(), input.size(), stream, mr, less);
    }
  }

 public:
  /**
   * @brief Called by the type-dispatcher to reduce the input column.
   *
   * @tparam ElementType The input column type
   * @param input Input column (must be numeric)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Memory resources used for temporary allocations and the returned scalar
   */
  explicit arg_minmax_dispatcher(nan_semantics semantics = nan_semantics::UNORDERED)
    : semantics_{semantics}
  {
  }

  template <typename ElementType>
  [[nodiscard]] std::unique_ptr<scalar> operator()(column_view const& input,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr) const
    requires(is_supported<ElementType>())
  {
    auto const idx = find_arg_minmax<ElementType>(input, stream, mr.as_temporary());
    return make_fixed_width_scalar<size_type>(idx, stream, mr.get_output_mr());
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     cuda::stream_ref,
                                     cudf::memory_resources) const
    requires(not is_supported<ElementType>())
  {
    CUDF_FAIL("ARGMIN/ARGMAX is not supported for this type");
  }
};

}  // namespace cudf::reduction::simple::detail
