/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_pass_reductions.cuh"

namespace cudf::groupby::detail::hash {
namespace {

struct m2_sum_count {
  double sum;
  size_type count;
};

struct m2_sum_count_op {
  __device__ m2_sum_count operator()(m2_sum_count const& lhs, m2_sum_count const& rhs) const
  {
    return {lhs.sum + rhs.sum, lhs.count + rhs.count};
  }
};

template <typename Source, bool HasNulls>
struct grouped_m2_sum_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;

  using result_type = cuda::std::conditional_t<HasNulls, m2_sum_count, double>;

  __device__ result_type operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if constexpr (HasNulls) {
      if (has_nulls && value.col.is_null_nocheck(row)) { return {}; }
      return {static_cast<double>(value(row)), 1};
    } else {
      return static_cast<double>(value(row));
    }
  }
};

struct m2_mean_count_fn {
  __device__ cuda::std::tuple<double, size_type> operator()(m2_sum_count const& state) const
  {
    return {state.count == 0 ? 0.0 : state.sum / state.count, state.count};
  }
};

struct m2_nonnullable_mean_writer {
  double* means;
  size_type* counts;
  size_type const* offsets;

  __device__ void operator()(size_type group, double sum) const
  {
    auto const count = offsets[group + 1] - offsets[group];
    means[group]     = count == 0 ? 0.0 : sum / count;
    if (counts != nullptr) { counts[group] = count; }
  }
};

template <typename Source>
struct grouped_squared_deviation_fn {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;
  double mean;

  __device__ double operator()(size_type position) const
  {
    auto const row = grouped_rows[position];
    if (has_nulls && value.col.is_null_nocheck(row)) { return 0.0; }
    auto const deviation = static_cast<double>(value(row)) - mean;
    return deviation * deviation;
  }
};

/// The shared CSR engine supplies the original group ID even when reducing a
/// long-group chunk. Binding the mean once avoids allocating or reading a label
/// for every input row.
template <typename Source>
struct centered_m2_values {
  size_type const* grouped_rows;
  value_accessor<Source> value;
  bool has_nulls;
  double const* means;

  __device__ auto for_group(size_type group) const
  {
    return cudf::detail::make_counting_transform_iterator(
      0, grouped_squared_deviation_fn<Source>{grouped_rows, value, has_nulls, means[group]});
  }
};

/// Finish bounded nonnullable groups in one collective so both centered passes access the same
/// group consecutively. Long groups continue to use means from the separate chunked pass.
template <typename Source>
struct local_centered_m2_values {
  centered_m2_values<Source> values;
  size_type const* offsets;
  size_type* counts;

  __device__ auto for_group(size_type group) const
  {
    auto const begin = static_cast<cuda::std::int64_t>(offsets[group]);
    auto const end   = static_cast<cuda::std::int64_t>(offsets[group + 1]);
    auto const count = static_cast<size_type>(end - begin);
    if (count > rows_per_chunk) { return values.for_group(group); }

    double sum  = 0.0;
    double mean = 0.0;
    if (count <= cudf::detail::warp_size) {
      // The small-group kernel assigns one complete group to each thread.
      for (auto position = begin; position < end; ++position) {
        sum += static_cast<double>(values.value(values.grouped_rows[position]));
      }
      mean = sum / count;
      if (counts != nullptr) { counts[group] = count; }
    } else {
      // Every lane participates: this branch is used only by the direct-group warp kernel.
      using warp_reduce = cub::WarpReduce<double, cudf::detail::warp_size>;
      __shared__
        typename warp_reduce::TempStorage storage[reduction_block_size / cudf::detail::warp_size];
      auto const lane = threadIdx.x % cudf::detail::warp_size;
      for (auto position = begin + lane; position < end; position += cudf::detail::warp_size) {
        sum += static_cast<double>(values.value(values.grouped_rows[position]));
      }
      sum = warp_reduce{storage[threadIdx.x / cudf::detail::warp_size]}.Reduce(
        sum, cuda::std::plus<double>{});
      if (lane == 0) {
        mean = sum / count;
        if (counts != nullptr) { counts[group] = count; }
      }
      mean = __shfl_sync(0xffffffff, mean, 0);
    }
    return cudf::detail::make_counting_transform_iterator(
      0, grouped_squared_deviation_fn<Source>{values.grouped_rows, values.value, false, mean});
  }
};

struct m2_reductions_fn {
  template <typename Source, bool HasNulls>
  std::vector<std::unique_ptr<column>> compute(host_span<reduction_context const> contexts,
                                               std::span<int8_t const> is_intermediate,
                                               bool include_counts,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr) const
  {
    auto const& first      = contexts.front();
    auto const num_columns = static_cast<size_type>(contexts.size());
    auto const stride      = include_counts ? 2 : 1;
    auto const temp_mr     = mr.get_temporary_mr();
    std::vector<std::unique_ptr<column>> results;
    results.reserve(static_cast<std::size_t>(num_columns) * stride);
    for (size_type i = 0; i < num_columns; ++i) {
      for (size_type j = 0; j < stride; ++j) {
        auto const index = static_cast<std::size_t>(i) * stride + j;
        results.push_back(
          make_numeric_column(j == 0 ? data_type{type_id::FLOAT64} : data_type{type_id::INT32},
                              first.num_groups,
                              mask_state::UNALLOCATED,
                              stream,
                              is_intermediate[index] ? temp_mr : mr.get_output_mr()));
      }
    }
    if (first.num_groups == 0) { return results; }

    auto const storage_size = static_cast<std::size_t>(num_columns) * first.num_groups;
    rmm::device_uvector<double> means(storage_size, stream, temp_mr);
    rmm::device_uvector<size_type> private_counts(
      HasNulls && !include_counts ? storage_size : 0, stream, temp_mr);
    using SumIterator           = decltype(cudf::detail::make_counting_transform_iterator(
      0, std::declval<grouped_m2_sum_fn<Source, HasNulls>>()));
    auto const make_mean_output = [&](double* mean, size_type* count) {
      if constexpr (HasNulls) {
        return cuda::transform_output_iterator{cuda::make_zip_iterator(mean, count),
                                               m2_mean_count_fn{}};
      } else {
        return cuda::tabulate_output_iterator{
          m2_nonnullable_mean_writer{mean, count, first.grouped.offsets.data()}, size_type{0}};
      }
    };
    using MeanOutput = decltype(make_mean_output(nullptr, nullptr));
    using MeanColumn = column_reduction<SumIterator, MeanOutput>;
    using M2Values   = cuda::std::
      conditional_t<HasNulls, centered_m2_values<Source>, local_centered_m2_values<Source>>;
    using M2Column    = column_reduction<M2Values, double*>;
    auto mean_columns = cudf::detail::make_empty_host_vector<MeanColumn>(num_columns, stream);
    auto m2_columns   = cudf::detail::make_empty_host_vector<M2Column>(num_columns, stream);
    for (size_type i = 0; i < num_columns; ++i) {
      auto const& ctx          = contexts[i];
      auto const column_offset = static_cast<std::size_t>(i) * first.num_groups;
      auto const result_offset = static_cast<std::size_t>(i) * stride;
      auto const mean          = means.data() + column_offset;
      auto const count         = include_counts
                                   ? results[result_offset + 1]->mutable_view().template begin<size_type>()
                                   : (HasNulls ? private_counts.data() + column_offset : nullptr);
      mean_columns.push_back(
        {cudf::detail::make_counting_transform_iterator(
           0,
           grouped_m2_sum_fn<Source, HasNulls>{
             ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls()}),
         make_mean_output(mean, count)});
      auto const m2_values = [&] {
        auto const centered = centered_m2_values<Source>{
          ctx.grouped.rows.data(), ctx.accessor<Source>(), ctx.values.has_nulls(), mean};
        if constexpr (HasNulls) {
          return centered;
        } else {
          return local_centered_m2_values<Source>{centered, ctx.grouped.offsets.data(), count};
        }
      }();
      m2_columns.push_back(
        {m2_values, results[result_offset]->mutable_view().template begin<double>()});
    }

    // Nonnullable bounded groups compute their means inside the centered reduction. Only long
    // groups and nullable batches need the separate mean pass over the same CSR.
    auto const reduce = [&](auto sums, auto squares) {
      using Sum = cuda::std::conditional_t<HasNulls, m2_sum_count, double>;
      using Op  = cuda::std::conditional_t<HasNulls, m2_sum_count_op, cuda::std::plus<double>>;
      reduce_group_columns(first.grouped, sums, Op{}, Sum{}, stream, mr, !HasNulls);
      reduce_group_columns(first.grouped, squares, cuda::std::plus<double>{}, 0.0, stream, mr);
    };
    if (num_columns == 1) {
      reduce(mean_columns.front(), m2_columns.front());
    } else {
      auto device_means = cudf::detail::make_device_uvector(mean_columns, stream, temp_mr);
      auto device_m2s   = cudf::detail::make_device_uvector(m2_columns, stream, temp_mr);
      reduce(reduction_columns{device_means.begin(), num_columns},
             reduction_columns{device_m2s.begin(), num_columns});
    }
    return results;
  }

  template <typename Source>
    requires(is_reduction_supported<Source>(aggregation::M2))
  std::vector<std::unique_ptr<column>> operator()(host_span<reduction_context const> contexts,
                                                  std::span<int8_t const> is_intermediate,
                                                  bool include_counts,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr) const
  {
    // Without nulls the CSR offsets already provide counts, leaving a plain double mean pass.
    auto const has_nulls = std::any_of(
      contexts.begin(), contexts.end(), [](auto const& ctx) { return ctx.values.has_nulls(); });
    if (has_nulls) {
      return compute<Source, true>(contexts, is_intermediate, include_counts, stream, mr);
    }
    return compute<Source, false>(contexts, is_intermediate, include_counts, stream, mr);
  }

  template <typename Source>
    requires(!is_reduction_supported<Source>(aggregation::M2))
  std::vector<std::unique_ptr<column>> operator()(host_span<reduction_context const>,
                                                  std::span<int8_t const>,
                                                  bool,
                                                  cuda::stream_ref,
                                                  cudf::memory_resources) const
  {
    CUDF_FAIL("Invalid source type for M2 aggregation.");
  }
};

}  // namespace

std::vector<std::unique_ptr<column>> compute_m2_reductions(
  host_span<reduction_context const> contexts,
  std::span<int8_t const> is_intermediate,
  bool include_counts,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  return type_dispatcher(contexts.front().values_type,
                         m2_reductions_fn{},
                         contexts,
                         is_intermediate,
                         include_counts,
                         stream,
                         mr);
}

}  // namespace cudf::groupby::detail::hash
