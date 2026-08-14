/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/operator.hpp>
#include <cuco/static_set.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/uninitialized_fill.h>

#include <optional>

namespace cudf::reduction::detail {

namespace {

// A CUDA Cooperative Group of 1 thread for the hash set for histogram
auto constexpr DEFAULT_HISTOGRAM_CG_SIZE = 1;

// Always use 64-bit signed integer for storing count.
using histogram_count_type = int64_t;

/**
 * @brief Specialized functor to check for not-zero of the second component of the input.
 */
struct is_not_zero {
  template <typename Pair>
  __device__ bool operator()(Pair const input) const
  {
    return cuda::std::get<1>(input) != 0;
  }
};

/**
 * @brief Building a histogram by gathering distinct rows from the input table and their
 * corresponding distinct counts.
 *
 * @param input The input table
 * @param distinct_indices Indices of the distinct rows
 * @param distinct_counts Distinct counts corresponding to the distinct rows
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned object's device memory
 * @return A list_scalar storing the output histogram
 */
auto gather_histogram(table_view const& input,
                      device_span<size_type const> distinct_indices,
                      std::unique_ptr<column>&& distinct_counts,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  auto distinct_rows = cudf::detail::gather(input,
                                            distinct_indices,
                                            out_of_bounds_policy::DONT_CHECK,
                                            cudf::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr);

  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(std::move(distinct_rows->release().front()));
  struct_children.emplace_back(std::move(distinct_counts));
  auto output_structs = make_structs_column(
    static_cast<size_type>(distinct_indices.size()), std::move(struct_children), 0, {}, stream, mr);

  return std::make_unique<cudf::list_scalar>(
    std::move(*output_structs.release()), true, stream, mr);
}

}  // namespace

std::unique_ptr<column> make_empty_histogram_like(column_view const& values)
{
  std::vector<std::unique_ptr<column>> struct_children;
  struct_children.emplace_back(empty_like(values));
  struct_children.emplace_back(make_empty_column(type_id::INT64));
  return std::make_unique<column>(data_type{type_id::STRUCT},
                                  0,
                                  rmm::device_buffer{},
                                  rmm::device_buffer{},
                                  0,
                                  std::move(struct_children));
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>, std::unique_ptr<column>>
compute_row_frequencies(table_view const& input,
                        std::optional<column_view> const& partial_counts,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  auto const has_nested_columns = cudf::detail::has_nested_columns(input);

  // Nested types are not tested, thus we just throw exception if we see such input for now.
  // We should remove this check after having enough tests.
  CUDF_EXPECTS(!has_nested_columns,
               "Nested types are not yet supported in histogram aggregation.",
               std::invalid_argument);

  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input, stream);
  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};

  auto const row_hasher = cudf::detail::row::hash::row_hasher(preprocessed_input);
  auto const key_hasher = row_hasher.device_hasher(has_nulls);
  auto const row_comp   = cudf::detail::row::equality::self_comparator(preprocessed_input);

  // Always compare NaNs as equal.
  using nan_equal_comparator = cudf::detail::row::equality::nan_equal_physical_equality_comparator;
  auto const value_comp      = nan_equal_comparator{};
  // Hard set the tparam `has_nested_columns` = false for now as we don't yet support nested columns
  auto const key_equal = row_comp.equal_to<false>(has_nulls, null_equality::EQUAL, value_comp);

  using row_hash = cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                              cudf::nullate::DYNAMIC>;

  std::size_t const num_rows = input.num_rows();

  // Construct a vector to store reduced counts and init to zero
  rmm::device_uvector<histogram_count_type> reduction_results(num_rows, stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    reduction_results.begin(),
    reduction_results.end(),
    histogram_count_type{0});

  // Construct a hash set
  auto row_set =
    cuco::static_set{cuco::extent{num_rows},
                     cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                     cuco::empty_key<size_type>{-1},
                     key_equal,
                     cuco::linear_probing<DEFAULT_HISTOGRAM_CG_SIZE, row_hash>{key_hasher},
                     {},  // thread scope
                     {},  // storage
                     rmm::mr::polymorphic_allocator<char>{},
                     stream.value()};

  // Device-accessible reference to the hash set with `insert_and_find` operator
  auto row_set_ref = row_set.ref(cuco::op::insert_and_find);

  // Compute frequencies (aka distinct counts) for the input rows.
  // Note that we consider null and NaNs as always equal.
  thrust::for_each(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator<std::size_t>{num_rows},
    [set_ref = row_set_ref,
     increments =
       partial_counts.has_value() ? partial_counts.value().begin<histogram_count_type>() : nullptr,
     counts = reduction_results.begin()] __device__(auto const idx) mutable {
      auto const [inserted_idx_ptr, _] = set_ref.insert_and_find(idx);
      cuda::atomic_ref<histogram_count_type, cuda::thread_scope_device> count_ref{
        counts[*inserted_idx_ptr]};
      auto const increment = increments ? increments[idx] : histogram_count_type{1};
      count_ref.fetch_add(increment, cuda::std::memory_order_relaxed);
    });

  // Set-size is the number of distinct (inserted) rows
  auto const set_size = row_set.size(stream);

  // Vector of distinct indices
  auto distinct_indices = std::make_unique<rmm::device_uvector<size_type>>(set_size, stream, mr);
  // Column of distinct counts
  auto distinct_counts = make_numeric_column(
    data_type{type_to_id<histogram_count_type>()}, set_size, mask_state::UNALLOCATED, stream, mr);

  // Copy row indices and counts to the output if counts are non-zero
  auto const input_it = cuda::make_zip_iterator(
    cuda::std::make_tuple(cuda::counting_iterator<cudf::size_type>{0}, reduction_results.begin()));
  auto const output_it = cuda::make_zip_iterator(cuda::std::make_tuple(
    distinct_indices->begin(), distinct_counts->mutable_view().begin<histogram_count_type>()));

  // Reduction results above are either group sizes of equal rows, or `0`.
  // The final output is non-zero group sizes only.
  cudf::detail::copy_if_async(input_it, input_it + num_rows, output_it, is_not_zero{}, stream);

  return {std::move(distinct_indices), std::move(distinct_counts)};
}

std::unique_ptr<cudf::scalar> histogram(column_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  // Empty group should be handled before reaching here.
  CUDF_EXPECTS(input.size() > 0, "Input should not be empty.", std::invalid_argument);

  auto const input_tv = table_view{{input}};
  auto [distinct_indices, distinct_counts] =
    compute_row_frequencies(input_tv, std::nullopt, stream, mr);
  return gather_histogram(input_tv, *distinct_indices, std::move(distinct_counts), stream, mr);
}

std::unique_ptr<cudf::scalar> merge_histogram(column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  // Empty group should be handled before reaching here.
  CUDF_EXPECTS(input.size() > 0, "Input should not be empty.", std::invalid_argument);
  CUDF_EXPECTS(!input.has_nulls(), "The input column must not have nulls.", std::invalid_argument);
  CUDF_EXPECTS(input.type().id() == type_id::STRUCT && input.num_children() == 2,
               "The input must be a structs column having two children.",
               std::invalid_argument);
  CUDF_EXPECTS(cudf::is_integral(input.child(1).type()) && !input.child(1).has_nulls(),
               "The second child of the input column must be of integral type and without nulls.",
               std::invalid_argument);

  auto const structs_cv   = structs_column_view{input};
  auto const input_values = structs_cv.get_sliced_child(0, stream);
  auto const input_counts = structs_cv.get_sliced_child(1, stream);

  auto const values_tv = table_view{{input_values}};
  auto [distinct_indices, distinct_counts] =
    compute_row_frequencies(values_tv, input_counts, stream, mr);
  return gather_histogram(values_tv, *distinct_indices, std::move(distinct_counts), stream, mr);
}

}  // namespace cudf::reduction::detail
