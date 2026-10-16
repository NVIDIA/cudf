/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/utils.hpp"
#include "groupby/hash/compute_groupby.hpp"
#include "groupby/hash/compute_single_pass_aggs.hpp"
#include "groupby/hash/functors.hpp"
#include "groupby/hash/group_reductions.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/groupby/groupby_helper.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/stream>

#include <algorithm>
#include <memory>
#include <utility>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

/// Whether an aggregation consumes group labels directly or while sorting its values.
bool uses_group_labels(aggregation::Kind kind)
{
  switch (kind) {
    case aggregation::HISTOGRAM:
    case aggregation::QUANTILE:
    case aggregation::MEDIAN:
    case aggregation::NUNIQUE:
    case aggregation::NTH_ELEMENT:
    case aggregation::COVARIANCE:
    case aggregation::CORRELATION:
    case aggregation::TDIGEST:
    case aggregation::MERGE_TDIGEST:
    case aggregation::BITWISE_AGG:
    case aggregation::HOST_UDF: return true;
    default: return false;
  }
}

bool needs_stable_groups(aggregation::Kind kind)
{
  if (is_hash_aggregation(kind)) { return false; }
  switch (kind) {
    case aggregation::HISTOGRAM:
    case aggregation::MERGE_HISTOGRAM:
    case aggregation::COLLECT_SET:
    case aggregation::MERGE_SETS:
    case aggregation::MERGE_TDIGEST:
    case aggregation::BITWISE_AGG:
    case aggregation::TOP_K:
    case aggregation::QUANTILE:
    case aggregation::MEDIAN:
    case aggregation::NUNIQUE:
    case aggregation::TDIGEST:
    case aggregation::COVARIANCE:
    case aggregation::CORRELATION: return false;
    default: return true;
  }
}

/**
 * @brief Creates column views with only valid elements in both input column views
 *
 * @param column_0 The first column
 * @param column_1 The second column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources whose temporary resource stores the shared null mask
 * @return tuple with new null mask (if null masks of input differ) and new column views
 */
auto column_view_with_common_nulls(column_view const& column_0,
                                   column_view const& column_1,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr)
{
  auto [new_nullmask, null_count] =
    cudf::bitmask_and(table_view{{column_0, column_1}},
                      stream,
                      cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()});
  if (null_count == 0) { return std::make_tuple(std::move(new_nullmask), column_0, column_1); }
  auto column_view_with_new_nullmask = [](auto const& col, void* nullmask, auto null_count) {
    return column_view(col.type(),
                       col.size(),
                       col.head(),
                       static_cast<cudf::bitmask_type const*>(nullmask),
                       null_count,
                       col.offset(),
                       std::vector(col.child_begin(), col.child_end()));
  };
  auto new_column_0 = null_count == column_0.null_count()
                        ? column_0
                        : column_view_with_new_nullmask(column_0, new_nullmask.data(), null_count);
  auto new_column_1 = null_count == column_1.null_count()
                        ? column_1
                        : column_view_with_new_nullmask(column_1, new_nullmask.data(), null_count);
  return std::make_tuple(std::move(new_nullmask), new_column_0, new_column_1);
}

}  // namespace

/**
 * @brief Functor to dispatch aggregation with
 *
 * This functor is to be used with `aggregation_dispatcher` to compute the
 * appropriate aggregation. If the values on which to run the aggregation are
 * unchanged, then this functor should be re-used. This is because it stores
 * memoised sorted and/or grouped values and re-using will save on computation
 * of these values.
 */
struct aggregate_result_functor final : store_result_functor {
  aggregate_result_functor(column_view const& values,
                           groupby_helper& helper,
                           cudf::detail::result_cache& cache,
                           bool expose_intermediates,
                           cuda::stream_ref stream,
                           cudf::memory_resources mr)
    : store_result_functor(values, helper, cache, stream, mr),
      expose_intermediates(expose_intermediates)
  {
  }

  bool const expose_intermediates;

  template <aggregation::Kind k>
  void operator()(aggregation const& agg)
  {
    if (cache.has_result(values, agg)) { return; }
    if constexpr (is_hash_aggregation(k)) {
      auto const values_type = cudf::is_dictionary(values.type())
                                 ? dictionary_column_view(values).keys().type()
                                 : values.type();
      auto const supported   = [values_type] {
        if constexpr (k == aggregation::M2 || k == aggregation::VARIANCE || k == aggregation::STD) {
          return hash::is_single_pass_agg_supported(values_type, aggregation::M2);
        }
        return cudf::detail::is_valid_aggregation(values_type, k);
      }();
      CUDF_EXPECTS(supported, "Unsupported groupby reduction type-agg combination");
      aggregation_request request;
      request.values = values;
      request.aggregations.emplace_back(dynamic_cast<groupby_aggregation*>(agg.clone().release()));
      hash::compute_aggregations(
        std::span{&request, 1}, helper, cache, expose_intermediates, stream, mr);
    } else {
      CUDF_FAIL("Unsupported aggregation.");
    }
  }
};

template <>
void aggregate_result_functor::operator()<aggregation::HISTOGRAM>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  auto const num_groups = helper.num_groups(stream, mr);
  if (num_groups == helper.num_keys(stream, mr)) {
    cache.add_result(
      values, agg, detail::make_singleton_histograms(take_grouped_values(), stream, mr));
    return;
  }

  cache.add_result(values,
                   agg,
                   detail::group_histogram(get_unordered_grouped_values(),
                                           helper.group_labels(stream, mr),
                                           num_groups,
                                           stream,
                                           mr.get_output_mr()));
}

template <>
void aggregate_result_functor::operator()<aggregation::QUANTILE>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  auto count_agg = make_count_aggregation();
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view const group_sizes = cache.get_result(values, *count_agg);
  auto& quantile_agg            = dynamic_cast<cudf::detail::quantile_aggregation const&>(agg);

  auto result = detail::group_quantiles(get_sorted_values(),
                                        group_sizes,
                                        helper.group_offsets(stream, mr),
                                        helper.num_groups(stream, mr),
                                        quantile_agg._quantiles,
                                        quantile_agg._interpolation,
                                        stream,
                                        mr.get_output_mr());
  cache.add_result(values, agg, std::move(result));
}

template <>
void aggregate_result_functor::operator()<aggregation::MEDIAN>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  auto count_agg = make_count_aggregation();
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view const group_sizes = cache.get_result(values, *count_agg);

  auto result = detail::group_quantiles(get_sorted_values(),
                                        group_sizes,
                                        helper.group_offsets(stream, mr),
                                        helper.num_groups(stream, mr),
                                        {0.5},
                                        interpolation::LINEAR,
                                        stream,
                                        mr.get_output_mr());
  cache.add_result(values, agg, std::move(result));
}

template <>
void aggregate_result_functor::operator()<aggregation::NUNIQUE>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  auto& nunique_agg = dynamic_cast<cudf::detail::nunique_aggregation const&>(agg);

  auto result = detail::group_nunique(get_sorted_values(),
                                      helper.group_labels(stream, mr),
                                      helper.num_groups(stream, mr),
                                      helper.group_offsets(stream, mr),
                                      nunique_agg._null_handling,
                                      stream,
                                      mr.get_output_mr());
  cache.add_result(values, agg, std::move(result));
}

template <>
void aggregate_result_functor::operator()<aggregation::NTH_ELEMENT>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) return;

  auto& nth_element_agg = dynamic_cast<cudf::detail::nth_element_aggregation const&>(agg);

  auto count_agg = make_count_aggregation(nth_element_agg._null_handling);
  if (count_agg->kind == aggregation::COUNT_VALID) {
    operator()<aggregation::COUNT_VALID>(*count_agg);
  } else if (count_agg->kind == aggregation::COUNT_ALL) {
    operator()<aggregation::COUNT_ALL>(*count_agg);
  } else {
    CUDF_FAIL("Wrong count aggregation kind");
  }
  column_view const group_sizes = cache.get_result(values, *count_agg);

  cache.add_result(values,
                   agg,
                   detail::group_nth_element(get_grouped_values(),
                                             group_sizes,
                                             helper.group_labels(stream, mr),
                                             helper.group_offsets(stream, mr),
                                             helper.num_groups(stream, mr),
                                             nth_element_agg._n,
                                             nth_element_agg._null_handling,
                                             stream,
                                             mr.get_output_mr()));
}

template <>
void aggregate_result_functor::operator()<aggregation::COLLECT_LIST>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const null_handling =
    dynamic_cast<cudf::detail::collect_list_aggregation const&>(agg)._null_handling;
  auto const grouped = get_grouped_values();
  auto result        = [&] {
    // Null filtering creates a new child, so keep the original owner for later aggregations.
    if (null_handling == null_policy::EXCLUDE && grouped.has_nulls()) {
      return detail::group_collect(grouped,
                                   helper.group_offsets(stream, mr),
                                   helper.num_groups(stream, mr),
                                   null_handling,
                                   stream,
                                   mr);
    }
    // Otherwise the gathered column becomes the list child, retaining the cached view's owner.
    return detail::group_collect(take_grouped_values(),
                                 helper.group_offsets(stream, mr),
                                 helper.num_groups(stream, mr),
                                 null_handling,
                                 stream,
                                 mr);
  }();
  cache.add_result(values, agg, std::move(result));
}

template <>
void aggregate_result_functor::operator()<aggregation::COLLECT_SET>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const null_handling =
    dynamic_cast<cudf::detail::collect_set_aggregation const&>(agg)._null_handling;
  auto const collect_result =
    detail::group_collect(get_unordered_grouped_values(),
                          helper.group_offsets(stream, mr),
                          helper.num_groups(stream, mr),
                          null_handling,
                          stream,
                          cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()});
  auto const nulls_equal =
    dynamic_cast<cudf::detail::collect_set_aggregation const&>(agg)._nulls_equal;
  auto const nans_equal =
    dynamic_cast<cudf::detail::collect_set_aggregation const&>(agg)._nans_equal;
  cache.add_result(values,
                   agg,
                   lists::detail::distinct(lists_column_view{collect_result->view()},
                                           nulls_equal,
                                           nans_equal,
                                           duplicate_keep_option::KEEP_ANY,
                                           stream,
                                           mr.get_output_mr()));
}

/**
 * @brief Perform merging for the lists that correspond to the same key value.
 *
 * This aggregation is similar to `COLLECT_LIST` with the following differences:
 *  - It requires the input values to be a non-nullable lists column, and
 *  - The values (lists) corresponding to the same key will not result in a list of lists as output
 *    from `COLLECT_LIST`. Instead, those lists will result in a list generated by merging them
 *    together.
 *
 * In practice, this aggregation is used to merge the partial results of multiple (distributed)
 * groupby `COLLECT_LIST` aggregations into a final `COLLECT_LIST` result. Those distributed
 * aggregations were executed on different values columns partitioned from the original values
 * column, then their results were (vertically) concatenated before given as the values column for
 * this aggregation.
 */
template <>
void aggregate_result_functor::operator()<aggregation::MERGE_LISTS>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  cache.add_result(values,
                   agg,
                   detail::group_merge_lists(get_grouped_values(),
                                             helper.group_offsets(stream, mr),
                                             helper.num_groups(stream, mr),
                                             stream,
                                             mr.get_output_mr()));
}

/**
 * @brief Perform merging for the lists corresponding to the same key value, then dropping duplicate
 * list entries.
 *
 * This aggregation is similar to `COLLECT_SET` with the following differences:
 *  - It requires the input values to be a non-nullable lists column, and
 *  - The values (lists) corresponding to the same key will result in a list generated by merging
 *    them together then dropping duplicate entries.
 *
 * In practice, this aggregation is used to merge the partial results of multiple (distributed)
 * groupby `COLLECT_LIST` or `COLLECT_SET` aggregations into a final `COLLECT_SET` result. Those
 * distributed aggregations were executed on different values columns partitioned from the original
 * values column, then their results were (vertically) concatenated before given as the values
 * column for this aggregation.
 *
 * Firstly, this aggregation performs `MERGE_LISTS` to concatenate the input lists (corresponding to
 * the same key) into intermediate lists, then it calls `lists::distinct` on them to
 * remove duplicate list entries. As such, the input (partial results) to this aggregation should be
 * generated by (distributed) `COLLECT_LIST` aggregations, not `COLLECT_SET`, to avoid unnecessarily
 * removing duplicate entries for the partial results.
 *
 * Since duplicate list entries will be removed, the parameters `null_equality` and `nan_equality`
 * are needed for calling `lists::distinct`.
 */
template <>
void aggregate_result_functor::operator()<aggregation::MERGE_SETS>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const merged_result   = detail::group_merge_lists(get_unordered_grouped_values(),
                                                       helper.group_offsets(stream, mr),
                                                       helper.num_groups(stream, mr),
                                                       stream,
                                                       mr.get_temporary_mr());
  auto const& merge_sets_agg = dynamic_cast<cudf::detail::merge_sets_aggregation const&>(agg);
  cache.add_result(values,
                   agg,
                   lists::detail::distinct(lists_column_view{merged_result->view()},
                                           merge_sets_agg._nulls_equal,
                                           merge_sets_agg._nans_equal,
                                           duplicate_keep_option::KEEP_ANY,
                                           stream,
                                           mr.get_output_mr()));
}

/**
 * @brief Perform merging for the M2 values that correspond to the same key value.
 *
 * The partial results input to this aggregation is a structs column with children are columns
 * generated by three other groupby aggregations: `COUNT_VALID`, `MEAN`, and `M2` that were
 * performed on partitioned datasets. After distributedly computed, the results output from these
 * aggregations are (vertically) concatenated before assembling into a structs column given as the
 * values column for this aggregation.
 *
 * For recursive merging of `M2` values, the aggregations values of all input (`COUNT_VALID`,
 * `MEAN`, and `M2`) are all merged and stored in the output of this aggregation. As such, the
 * output will be a structs column containing children columns of merged `COUNT_VALID`, `MEAN`, and
 * `M2` values.
 *
 * The values of M2 are merged following the parallel algorithm described here:
 * https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Parallel_algorithm
 */
template <>
void aggregate_result_functor::operator()<aggregation::MERGE_M2>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  cache.add_result(values,
                   agg,
                   detail::group_merge_m2(get_grouped_values(),
                                          helper.group_offsets(stream, mr),
                                          helper.num_groups(stream, mr),
                                          stream,
                                          mr.get_output_mr()));
}

/**
 * @brief Perform merging for multiple histograms that correspond to the same key value.
 *
 * The partial results input to this aggregation is a structs column that is concatenated from
 * multiple outputs of HISTOGRAM aggregations.
 */
template <>
void aggregate_result_functor::operator()<aggregation::MERGE_HISTOGRAM>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  cache.add_result(values,
                   agg,
                   detail::group_merge_histogram(get_unordered_grouped_values(),
                                                 helper.group_offsets(stream, mr),
                                                 helper.num_groups(stream, mr),
                                                 stream,
                                                 mr.get_output_mr()));
}

/**
 * @brief Perform covariance between two child columns of non-nullable struct column.
 *
 */
template <>
void aggregate_result_functor::operator()<aggregation::COVARIANCE>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `groupby covariance` must be a structs column.");
  CUDF_EXPECTS(values.num_children() == 2,
               "Input to `groupby covariance` must be a structs column having 2 children columns.");

  auto const& cov_agg = dynamic_cast<cudf::detail::covariance_aggregation const&>(agg);
  // Covariance only for valid values in both columns.
  // in non-identical null mask cases, this prevents caching of the results - STD, MEAN, COUNT.
  auto [_, values_child0, values_child1] =
    column_view_with_common_nulls(values.child(0), values.child(1), stream, mr);

  auto mean_agg = make_mean_aggregation();
  aggregate_result_functor(values_child0, helper, cache, expose_intermediates, stream, mr)
    .operator()<aggregation::MEAN>(*mean_agg);
  aggregate_result_functor(values_child1, helper, cache, expose_intermediates, stream, mr)
    .operator()<aggregation::MEAN>(*mean_agg);

  auto const mean0 = cache.get_result(values_child0, *mean_agg);
  auto const mean1 = cache.get_result(values_child1, *mean_agg);
  auto count_agg   = make_count_aggregation();
  auto const count = cache.get_result(values_child0, *count_agg);

  cache.add_result(values,
                   agg,
                   detail::group_covariance(values_child0,
                                            values_child1,
                                            helper.reduction_groups(stream, mr),
                                            helper.group_labels(stream, mr),
                                            helper.num_groups(stream, mr),
                                            count,
                                            mean0,
                                            mean1,
                                            cov_agg._min_periods,
                                            cov_agg._ddof,
                                            stream,
                                            mr.get_output_mr()));
}

/**
 * @brief Perform correlation between two child columns of non-nullable struct column.
 *
 */
template <>
void aggregate_result_functor::operator()<aggregation::CORRELATION>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `groupby correlation` must be a structs column.");
  CUDF_EXPECTS(
    values.num_children() == 2,
    "Input to `groupby correlation` must be a structs column having 2 children columns.");
  CUDF_EXPECTS(not values.nullable(),
               "Input to `groupby correlation` must be a non-nullable structs column.");

  auto const& corr_agg = dynamic_cast<cudf::detail::correlation_aggregation const&>(agg);
  CUDF_EXPECTS(corr_agg._type == correlation_type::PEARSON,
               "Only Pearson correlation is supported.");

  // Correlation only for valid values in both columns.
  // in non-identical null mask cases, this prevents caching of the results - STD, MEAN, COUNT
  auto [_, values_child0, values_child1] =
    column_view_with_common_nulls(values.child(0), values.child(1), stream, mr);

  auto std_agg = make_std_aggregation();
  aggregate_result_functor(values_child0, helper, cache, expose_intermediates, stream, mr)
    .operator()<aggregation::STD>(*std_agg);
  aggregate_result_functor(values_child1, helper, cache, expose_intermediates, stream, mr)
    .operator()<aggregation::STD>(*std_agg);

  auto mean_agg = make_mean_aggregation();
  aggregate_result_functor(values_child0, helper, cache, expose_intermediates, stream, mr)
    .operator()<aggregation::MEAN>(*mean_agg);
  aggregate_result_functor(values_child1, helper, cache, expose_intermediates, stream, mr)
    .operator()<aggregation::MEAN>(*mean_agg);

  // Compute covariance here to avoid repeated computation of mean & count
  auto cov_agg = make_covariance_aggregation(corr_agg._min_periods);
  if (not cache.has_result(values, *cov_agg)) {
    auto const mean0 = cache.get_result(values_child0, *mean_agg);
    auto const mean1 = cache.get_result(values_child1, *mean_agg);
    auto count_agg   = make_count_aggregation();
    auto const count = cache.get_result(values_child0, *count_agg);

    auto const& cov_agg_obj = dynamic_cast<cudf::detail::covariance_aggregation const&>(*cov_agg);
    cache.add_result(values,
                     *cov_agg,
                     detail::group_covariance(values_child0,
                                              values_child1,
                                              helper.reduction_groups(stream, mr),
                                              helper.group_labels(stream, mr),
                                              helper.num_groups(stream, mr),
                                              count,
                                              mean0,
                                              mean1,
                                              cov_agg_obj._min_periods,
                                              cov_agg_obj._ddof,
                                              stream,
                                              mr.get_output_mr()));
  }

  auto const stddev0    = cache.get_result(values_child0, *std_agg);
  auto const stddev1    = cache.get_result(values_child1, *std_agg);
  auto const covariance = cache.get_result(values, *cov_agg);
  cache.add_result(
    values,
    agg,
    detail::group_correlation(covariance, stddev0, stddev1, stream, mr.get_output_mr()));
}

/**
 * @brief Generate a tdigest column from a grouped set of numeric input values.
 *
 * The tdigest column produced is of the following structure:
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    },
 *    ...
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 *
 * Each output row is a single tdigest.  The length of the row is the "size" of the
 * tdigest, each element of which represents a weighted centroid (mean, weight).
 */
template <>
void aggregate_result_functor::operator()<aggregation::TDIGEST>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const max_centroids =
    dynamic_cast<cudf::detail::tdigest_aggregation const&>(agg).max_centroids;

  auto count_agg = make_count_aggregation();
  operator()<aggregation::COUNT_VALID>(*count_agg);
  column_view const valid_counts = cache.get_result(values, *count_agg);

  cache.add_result(values,
                   agg,
                   cudf::tdigest::detail::group_tdigest(
                     get_sorted_values(),
                     helper.group_offsets(stream, mr),
                     helper.group_labels(stream, mr),
                     {valid_counts.begin<size_type>(), static_cast<size_t>(valid_counts.size())},
                     helper.num_groups(stream, mr),
                     max_centroids,
                     stream,
                     mr.get_output_mr()));
}

/**
 * @brief Generate a merged tdigest column from a grouped set of input tdigest columns.
 *
 * The tdigest column produced is of the following structure:
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    },
 *    ...
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 *
 * Each output row is a single tdigest.  The length of the row is the "size" of the
 * tdigest, each element of which represents a weighted centroid (mean, weight).
 */
template <>
void aggregate_result_functor::operator()<aggregation::MERGE_TDIGEST>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const max_centroids =
    dynamic_cast<cudf::detail::merge_tdigest_aggregation const&>(agg).max_centroids;
  cache.add_result(values,
                   agg,
                   cudf::tdigest::detail::group_merge_tdigest(get_unordered_grouped_values(),
                                                              helper.group_offsets(stream, mr),
                                                              helper.group_labels(stream, mr),
                                                              helper.num_groups(stream, mr),
                                                              max_centroids,
                                                              stream,
                                                              mr.get_output_mr()));
}

template <>
void aggregate_result_functor::operator()<aggregation::BITWISE_AGG>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const bit_op = dynamic_cast<cudf::detail::bitwise_aggregation const&>(agg).bit_op;
  auto result       = detail::group_bitwise(bit_op,
                                      get_unordered_grouped_values(),
                                      helper.group_labels(stream, mr),
                                      helper.num_groups(stream, mr),
                                      stream,
                                      mr.get_output_mr());
  cache.add_result(values, agg, std::move(result));
}

template <>
void aggregate_result_functor::operator()<aggregation::TOP_K>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const k          = dynamic_cast<cudf::detail::top_k_aggregation const&>(agg).k;
  auto const topk_order = dynamic_cast<cudf::detail::top_k_aggregation const&>(agg).topk_order;

  auto result = detail::group_top_k(k,
                                    topk_order,
                                    get_unordered_grouped_values(),
                                    helper.group_offsets(stream, mr),
                                    stream,
                                    mr.get_output_mr());
  cache.add_result(values, agg, std::move(result));
}

// Note: the definition for HOST_UDF specialization needs to be at last.
// This is because it calls to `aggregation_dispatcher` which requires to see all other function
// specializations defined before this.
template <>
void aggregate_result_functor::operator()<aggregation::HOST_UDF>(aggregation const& agg)
{
  if (cache.has_result(values, agg)) { return; }

  auto const& udf_base_ptr = dynamic_cast<cudf::detail::host_udf_aggregation const&>(agg).udf_ptr;
  auto const udf_ptr       = dynamic_cast<groupby_host_udf*>(udf_base_ptr.get());
  CUDF_EXPECTS(udf_ptr != nullptr, "Invalid HOST_UDF instance for groupby aggregation.");

  if (!udf_ptr->callback_input_values) {
    udf_ptr->callback_input_values = [&]() -> column_view { return values; };
  }
  if (!udf_ptr->callback_grouped_values) {
    udf_ptr->callback_grouped_values = [&]() -> column_view { return get_grouped_values(); };
  }
  if (!udf_ptr->callback_sorted_grouped_values) {
    udf_ptr->callback_sorted_grouped_values = [&]() -> column_view { return get_sorted_values(); };
  }
  if (!udf_ptr->callback_num_groups) {
    udf_ptr->callback_num_groups = [&]() -> size_type { return helper.num_groups(stream, mr); };
  }
  if (!udf_ptr->callback_group_offsets) {
    udf_ptr->callback_group_offsets = [&]() -> device_span<size_type const> {
      return helper.group_offsets(stream, mr);
    };
  }
  if (!udf_ptr->callback_group_labels) {
    udf_ptr->callback_group_labels = [&]() -> device_span<size_type const> {
      return helper.group_labels(stream, mr);
    };
  }
  if (!udf_ptr->callback_compute_aggregation) {
    udf_ptr->callback_compute_aggregation =
      [&](std::unique_ptr<aggregation> other_agg) -> column_view {
      cudf::detail::aggregation_dispatcher(other_agg->kind, *this, *other_agg);
      return cache.get_result(values, *other_agg);
    };
  }

  cache.add_result(values, agg, (*udf_ptr)(stream, mr.get_output_mr()));
}

}  // namespace detail

// All aggregations share one grouping and one result cache.
std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> detail::hash::groupby(
  std::span<aggregation_request const> requests,
  detail::groupby_helper& helper,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  // Build ordered rows directly when any request needs them, before metadata or unordered
  // requests can materialize an intermediate permutation. Existing cached groups are reused.
  auto const any_agg = [&](auto predicate) {
    return std::ranges::any_of(requests, [&](auto const& request) {
      return std::ranges::any_of(request.aggregations,
                                 [&](auto const& agg) { return predicate(agg->kind); });
    });
  };
  auto const has_specialized =
    any_agg([](auto kind) { return !detail::is_hash_aggregation(kind); });
  if (any_agg(detail::needs_stable_groups)) {
    helper.grouped_order(any_agg(detail::uses_group_labels), stream, mr);
  } else if (has_specialized) {
    // Specialized aggregations need grouped rows too. Establish their final group order before
    // a preceding counts-only reduction can cache results using the same group indices.
    helper.unordered_grouped_order(stream, mr);
  }

  // Share primitive results and compound dependencies across all requests.
  cudf::detail::result_cache cache(requests.size());

  // A host UDF can request any dependency from the shared cache. Preserve its public result
  // semantics and output resource ownership even when it is initially only an intermediate.
  auto const expose_intermediates =
    any_agg([](auto kind) { return kind == aggregation::HOST_UDF; });

  if (!has_specialized) {
    detail::hash::compute_aggregations(requests, helper, cache, expose_intermediates, stream, mr);
  } else {
    // Batch reducible requests together even when other requests need specialized algorithms.
    std::vector<aggregation_request> reductions;
    reductions.reserve(requests.size());
    for (auto const& request : requests) {
      aggregation_request direct;
      direct.values = request.values;
      for (auto const& agg : request.aggregations) {
        if (detail::is_hash_aggregation(agg->kind)) {
          direct.aggregations.emplace_back(
            dynamic_cast<groupby_aggregation*>(agg->clone().release()));
        }
      }
      if (!direct.aggregations.empty()) { reductions.push_back(std::move(direct)); }
    }
    if (!reductions.empty()) {
      detail::hash::compute_aggregations(
        reductions, helper, cache, expose_intermediates, stream, mr);
    }
  }

  for (auto const& request : requests) {
    auto store_functor = detail::aggregate_result_functor(
      request.values, helper, cache, expose_intermediates, stream, mr);
    for (auto const& agg : request.aggregations) {
      cudf::detail::aggregation_dispatcher(agg->kind, store_functor, *agg);
    }
  }

  auto results = detail::extract_results(requests, cache, stream, mr.get_output_mr());

  return std::pair(helper.distinct_keys(stream, mr), std::move(results));
}
}  // namespace groupby
}  // namespace cudf
