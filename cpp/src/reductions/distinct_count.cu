/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/reduction/detail/distinct_count.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {
namespace {

class row_validity {
 public:
  row_validity(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ inline bool operator()(size_type const& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  bitmask_type const* _row_bitmask;
};

/**
 * @brief Functor to check for `NaN` at an index in a `column_device_view`.
 *
 * @tparam T The type of `column_device_view`
 */
template <typename T>
struct check_for_nan {
  /*
   * @brief Construct from a column_device_view.
   *
   * @param[in] input The `column_device_view`
   */
  check_for_nan(cudf::column_device_view input) : _input{input} {}

  /**
   * @brief Operator to be called to check for `NaN` at `index` in `_input`
   *
   * @param[in] index The index at which the `NaN` needs to be checked in `input`
   *
   * @returns bool true if value at `index` is `NaN` and not null, else false
   */
  __device__ bool operator()(size_type index) const noexcept
  {
    return std::isnan(_input.data<T>()[index]) and _input.is_valid(index);
  }

  cudf::column_device_view _input;
};

/**
 * @brief A structure to be used along with type_dispatcher to check if a
 * `column_view` has `NaN`.
 */
struct has_nans {
  /**
   * @brief Checks if `input` has `NaN`
   *
   * @note This will be applicable only for floating point type columns.
   *
   * @param[in] input The `column_view` which will be checked for `NaN`
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @returns bool true if `input` has `NaN` else false
   */
  template <typename T>
  bool operator()(column_view const& input, rmm::cuda_stream_view stream)
    requires(std::is_floating_point_v<T>)
  {
    auto input_device_view = cudf::column_device_view::create(input, stream);
    auto device_view       = *input_device_view;
    return thrust::any_of(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          cuda::counting_iterator<cudf::size_type>{0},
                          cuda::counting_iterator<cudf::size_type>{input.size()},
                          check_for_nan<T>(device_view));
  }

  /**
   * @brief Checks if `input` has `NaN`
   *
   * @note This will be applicable only for non-floating point type columns. And
   * non-floating point columns can never have `NaN`, so it will always return
   * false
   *
   * @param[in] input The `column_view` which will be checked for `NaN`
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @returns bool Always false as non-floating point columns can't have `NaN`
   */
  template <typename T>
  bool operator()(column_view const&, rmm::cuda_stream_view)
    requires(not std::is_floating_point_v<T>)
  {
    return false;
  }
};
}  // namespace

cudf::size_type distinct_count(table_view const& keys,
                               null_equality nulls_equal,
                               rmm::cuda_stream_view stream)
{
  auto const num_rows = keys.num_rows();
  if (num_rows == 0) { return 0; }  // early exit for empty input
  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(keys)};

  auto const preprocessed_input = cudf::detail::row::hash::preprocessed_table::create(keys, stream);
  auto const row_hasher         = cudf::detail::row::hash::row_hasher(preprocessed_input);
  auto const hash_key           = row_hasher.device_hasher(has_nulls);
  auto const row_comp           = cudf::detail::row::equality::self_comparator(preprocessed_input);

  auto const comparator_helper = [&](auto const row_equal) {
    using hasher_type = decltype(hash_key);
    auto key_set      = cuco::static_set{cuco::extent{static_cast<std::size_t>(num_rows)},
                                    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                    cuco::empty_key<cudf::size_type>{-1},
                                    row_equal,
                                    cuco::linear_probing<1, hasher_type>{hash_key},
                                         {},
                                         {},
                                    rmm::mr::polymorphic_allocator<char>{},
                                    stream.value()};

    auto const iter = cuda::counting_iterator<cudf::size_type>{0};
    // when nulls are equal, we skip hashing any row that has a null
    // in every column to improve efficiency.
    if (nulls_equal == null_equality::EQUAL and has_nulls) {
      cuda::counting_iterator<size_type> stencil(0);
      // We must consider a row if any of its column entries is valid,
      // hence OR together the validities of the columns.
      auto const [row_bitmask, null_count] =
        cudf::detail::bitmask_or(keys, stream, cudf::get_current_device_resource_ref());

      // Unless all columns have a null mask, row_bitmask will be
      // null, and null_count will be zero. Equally, unless there is
      // some row which is null in all columns, null_count will be
      // zero. So, it is only when null_count is not zero that we need
      // to do a filtered insertion.
      if (null_count > 0) {
        row_validity pred{static_cast<bitmask_type const*>(row_bitmask.data())};
        return key_set.insert_if(iter, iter + num_rows, stencil, pred, stream.value()) + 1;
      }
    }
    // otherwise, insert all
    return key_set.insert(iter, iter + num_rows, stream.value());
  };

  if (cudf::detail::has_nested_columns(keys)) {
    auto const row_equal = row_comp.equal_to<true>(has_nulls, nulls_equal);
    return comparator_helper(row_equal);
  } else {
    auto const row_equal = row_comp.equal_to<false>(has_nulls, nulls_equal);
    return comparator_helper(row_equal);
  }
}

cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream)
{
  if (0 == input.size()) { return 0; }

  if (input.null_count() == input.size()) {
    return static_cast<size_type>(null_handling == null_policy::INCLUDE);
  }

  auto count = detail::distinct_count(table_view{{input}}, null_equality::EQUAL, stream);

  // Check for nulls. If the null policy is EXCLUDE and null values were found,
  // we decrement the count.
  auto const has_null = input.has_nulls();
  if (null_handling == null_policy::EXCLUDE and has_null) { --count; }

  // Check for NaNs. There are two cases that can lead to decrementing the
  // count. The first case is when the input has no nulls, but has NaN values
  // handled as a null via NAN_IS_NULL and has a policy to EXCLUDE null values
  // from the count. The second case is when the input has null values and NaN
  // values handled as nulls via NAN_IS_NULL. Regardless of whether the null
  // policy is set to EXCLUDE, we decrement the count to avoid double-counting
  // null and NaN as distinct entities.
  auto const has_nan_as_null = (nan_handling == nan_policy::NAN_IS_NULL) and
                               cudf::type_dispatcher(input.type(), has_nans{}, input, stream);
  if (has_nan_as_null and (has_null or null_handling == null_policy::EXCLUDE)) { --count; }
  return count;
}
}  // namespace detail

cudf::size_type distinct_count(column_view const& input,
                               null_policy null_handling,
                               nan_policy nan_handling,
                               rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::distinct_count(input, null_handling, nan_handling, stream);
}

cudf::size_type distinct_count(table_view const& input,
                               null_equality nulls_equal,
                               rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::distinct_count(input, nulls_equal, stream);
}
}  // namespace cudf
