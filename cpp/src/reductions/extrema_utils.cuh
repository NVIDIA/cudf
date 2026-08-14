/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nested_types_extrema_utils.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
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
 * @brief Call reduce and return a scalar of the specified type, specialized for ARGMIN and
 * ARGMAX aggregations.
 *
 * @tparam K The aggregation to execute on the column, must be either ARGMIN or ARGMAX.
 */
template <aggregation::Kind K>
class arg_minmax_dispatcher {
  static_assert(K == aggregation::ARGMIN or K == aggregation::ARGMAX,
                "Aggregation kind must be either ARGMIN or ARGMAX");

  template <typename ElementType>
  static constexpr bool is_supported()
  {
    return !cudf::is_dictionary<ElementType>() && !std::is_same_v<ElementType, void>;
  }

  template <typename InputIterator, typename... Args>
  size_type find_extremum_idx(InputIterator it,
                              size_type size,
                              rmm::cuda_stream_view stream,
                              Args&&... args) const
  {
    auto const pos = [&] {
      if constexpr (K == aggregation::ARGMIN) {
        return thrust::min_element(
          rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
          it,
          it + size,
          std::forward<Args>(args)...);
      } else {
        return thrust::max_element(
          rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
          it,
          it + size,
          std::forward<Args>(args)...);
      }
    }();
    return static_cast<size_type>(cuda::std::distance(it, pos));
  }

  template <typename ElementType>
  [[nodiscard]] size_type find_arg_minmax(column_view const& input,
                                          rmm::cuda_stream_view stream) const
    requires(cudf::is_nested<ElementType>())
  {
    using Op = std::conditional_t<K == aggregation::ARGMIN,
                                  reduction::detail::op::min,
                                  reduction::detail::op::max>;
    auto const binop_generator =
      reduction::detail::arg_minmax_binop_generator::create<Op>(input, stream);
    return find_extremum_idx(cuda::counting_iterator<cudf::size_type>{0},
                             input.size(),
                             stream,
                             noinline_adapter_fn{binop_generator.less()});
  }

  // This function is used for types such as string, timestamp, fixed point, etc.
  // `input` may be the dictionary column itself (rather than its keys or a decoded column):
  // the lexicographic self_comparator compares `keys[indices[i]]` for dictionary columns
  // directly, so no decoding is required and no assumption is made that the keys are sorted.
  template <typename ElementType>
  [[nodiscard]] size_type find_arg_minmax(column_view const& input,
                                          rmm::cuda_stream_view stream) const
    requires(not cudf::is_nested<ElementType>() and not cudf::is_numeric<ElementType>())
  {
    // Nulls are considered "greater" (ARGMIN), or "less" (ARGMAX) than non-null values.
    auto const null_orders =
      std::vector<null_order>{K == aggregation::ARGMIN ? null_order::AFTER : null_order::BEFORE};
    auto const comparator = cudf::detail::row::lexicographic::self_comparator{
      table_view{{input}}, {}, null_orders, stream};
    auto d_comp =
      comparator.less<false /* has_nested_columns */>(nullate::DYNAMIC{input.has_nulls()});
    return find_extremum_idx(cuda::counting_iterator<cudf::size_type>{0},
                             input.size(),
                             stream,
                             noinline_adapter_fn{std::move(d_comp)});
  }

  template <typename ElementType>
  [[nodiscard]] size_type find_arg_minmax(column_view const& input,
                                          rmm::cuda_stream_view stream) const
    requires(cudf::is_numeric<ElementType>())  // integer + floating point numbers
  {
    using Op = std::conditional_t<K == aggregation::ARGMIN,
                                  reduction::detail::op::min,
                                  reduction::detail::op::max>;
    // Dictionary keys are not guaranteed to be sorted, so the index of the min/max index does
    // not identify the min/max key. Read `keys[indices[i]]` directly per row via a lazy
    // iterator instead of decoding (and copying) the whole column.
    if (is_dictionary(input.type())) {
      auto const d_dict = column_device_view::create(input, stream);
      if (input.has_nulls()) {
        auto const transformer =
          Op{}.template get_null_replacing_element_transformer<ElementType>();
        auto const p =
          cudf::dictionary::detail::make_dictionary_pair_iterator<ElementType>(*d_dict, true);
        auto const it = thrust::make_transform_iterator(p, transformer);
        return find_extremum_idx(it, input.size(), stream);
      }
      auto const it = cudf::dictionary::detail::make_dictionary_iterator<ElementType>(*d_dict);
      return find_extremum_idx(it, input.size(), stream);
    }
    if (input.has_nulls()) {
      auto const d_input     = column_device_view::create(input, stream);
      auto const transformer = Op{}.template get_null_replacing_element_transformer<ElementType>();
      auto const it =
        thrust::make_transform_iterator(d_input->pair_begin<ElementType, true>(), transformer);
      return find_extremum_idx(it, input.size(), stream);
    } else {
      return find_extremum_idx(input.begin<ElementType>(), input.size(), stream);
    }
  }

 public:
  /**
   * @brief Called by the type-dispatcher to reduce the input column.
   *
   * @tparam ElementType The input column type
   * @param input Input column (must be numeric)
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned scalar's device memory
   */
  template <typename ElementType>
  [[nodiscard]] std::unique_ptr<scalar> operator()(column_view const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
    requires(is_supported<ElementType>())
  {
    auto const idx = find_arg_minmax<ElementType>(input, stream);
    return make_fixed_width_scalar<size_type>(idx, stream, mr);
  }

  template <typename ElementType>
  std::unique_ptr<scalar> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
    requires(not is_supported<ElementType>())
  {
    CUDF_FAIL("ARGMIN/ARGMAX is not supported for this type");
  }
};

}  // namespace cudf::reduction::simple::detail
