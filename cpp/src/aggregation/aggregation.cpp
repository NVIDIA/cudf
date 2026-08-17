/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>

namespace cudf {

namespace detail {

}

/// Factory to create a SUM aggregation
template <typename Base>
std::unique_ptr<Base> make_sum_aggregation()
{
  return std::make_unique<detail::sum_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_sum_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_sum_aggregation<rolling_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_sum_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_scan_aggregation>
make_sum_aggregation<groupby_scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation> make_sum_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<scan_aggregation> make_sum_aggregation<scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_sum_aggregation<segmented_reduce_aggregation>();

/// Factory to create a SUM_OVERFLOW aggregation
template <typename Base>
std::unique_ptr<Base> make_sum_overflow_aggregation()
{
  return std::make_unique<detail::sum_overflow_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_sum_overflow_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_sum_overflow_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_scan_aggregation>
make_sum_overflow_aggregation<groupby_scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_sum_overflow_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_sum_overflow_aggregation<segmented_reduce_aggregation>();

/// Factory to create a PRODUCT aggregation
template <typename Base>
std::unique_ptr<Base> make_product_aggregation()
{
  return std::make_unique<detail::product_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_product_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_product_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_scan_aggregation>
make_product_aggregation<groupby_scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_product_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<scan_aggregation> make_product_aggregation<scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_product_aggregation<segmented_reduce_aggregation>();

/// Factory to create a MIN aggregation
template <typename Base>
std::unique_ptr<Base> make_min_aggregation()
{
  return std::make_unique<detail::min_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_min_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_min_aggregation<rolling_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_min_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_scan_aggregation>
make_min_aggregation<groupby_scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation> make_min_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<scan_aggregation> make_min_aggregation<scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_min_aggregation<segmented_reduce_aggregation>();

/// Factory to create a MAX aggregation
template <typename Base>
std::unique_ptr<Base> make_max_aggregation()
{
  return std::make_unique<detail::max_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_max_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_max_aggregation<rolling_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_max_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_scan_aggregation>
make_max_aggregation<groupby_scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation> make_max_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<scan_aggregation> make_max_aggregation<scan_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_max_aggregation<segmented_reduce_aggregation>();

/// Factory to create a COUNT aggregation
template <typename Base>
std::unique_ptr<Base> make_count_aggregation(null_policy null_handling)
{
  auto kind =
    (null_handling == null_policy::INCLUDE) ? aggregation::COUNT_ALL : aggregation::COUNT_VALID;
  return std::make_unique<detail::count_aggregation>(kind);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_count_aggregation<aggregation>(
  null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_count_aggregation<rolling_aggregation>(null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_count_aggregation<groupby_aggregation>(null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<groupby_scan_aggregation>
make_count_aggregation<groupby_scan_aggregation>(null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation> make_count_aggregation<reduce_aggregation>(
  null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<scan_aggregation> make_count_aggregation<scan_aggregation>(
  null_policy null_handling);

/// Factory to create a HISTOGRAM aggregation
template <typename Base>
std::unique_ptr<Base> make_histogram_aggregation()
{
  return std::make_unique<detail::histogram_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_histogram_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_histogram_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_histogram_aggregation<reduce_aggregation>();

/// Factory to create a ANY aggregation
template <typename Base>
std::unique_ptr<Base> make_any_aggregation()
{
  return std::make_unique<detail::any_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_any_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation> make_any_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_any_aggregation<segmented_reduce_aggregation>();

/// Factory to create a ALL aggregation
template <typename Base>
std::unique_ptr<Base> make_all_aggregation()
{
  return std::make_unique<detail::all_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_all_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation> make_all_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_all_aggregation<segmented_reduce_aggregation>();

/// Factory to create a SUM_OF_SQUARES aggregation
template <typename Base>
std::unique_ptr<Base> make_sum_of_squares_aggregation()
{
  return std::make_unique<detail::sum_of_squares_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_sum_of_squares_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_sum_of_squares_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_sum_of_squares_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_sum_of_squares_aggregation<segmented_reduce_aggregation>();

/// Factory to create a MEAN aggregation
template <typename Base>
std::unique_ptr<Base> make_mean_aggregation()
{
  return std::make_unique<detail::mean_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_mean_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_mean_aggregation<rolling_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_mean_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_mean_aggregation<reduce_aggregation>();
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_mean_aggregation<segmented_reduce_aggregation>();

/// Factory to create a M2 aggregation
template <typename Base>
std::unique_ptr<Base> make_m2_aggregation()
{
  return std::make_unique<detail::m2_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_m2_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_m2_aggregation<groupby_aggregation>();

/// Factory to create a VARIANCE aggregation
template <typename Base>
std::unique_ptr<Base> make_variance_aggregation(size_type ddof)
{
  return std::make_unique<detail::var_aggregation>(ddof);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_variance_aggregation<aggregation>(
  size_type ddof);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_variance_aggregation<rolling_aggregation>(size_type ddof);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_variance_aggregation<groupby_aggregation>(size_type ddof);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_variance_aggregation<reduce_aggregation>(size_type ddof);
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_variance_aggregation<segmented_reduce_aggregation>(size_type ddof);

/// Factory to create a STD aggregation
template <typename Base>
std::unique_ptr<Base> make_std_aggregation(size_type ddof)
{
  return std::make_unique<detail::std_aggregation>(ddof);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_std_aggregation<aggregation>(size_type ddof);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation> make_std_aggregation<rolling_aggregation>(
  size_type ddof);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation> make_std_aggregation<groupby_aggregation>(
  size_type ddof);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation> make_std_aggregation<reduce_aggregation>(
  size_type ddof);
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_std_aggregation<segmented_reduce_aggregation>(size_type ddof);

/// Factory to create a MEDIAN aggregation
template <typename Base>
std::unique_ptr<Base> make_median_aggregation()
{
  return std::make_unique<detail::median_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_median_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_median_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_median_aggregation<reduce_aggregation>();

/// Factory to create a QUANTILE aggregation
template <typename Base>
std::unique_ptr<Base> make_quantile_aggregation(std::vector<double> const& quantiles,
                                                interpolation interp)
{
  return std::make_unique<detail::quantile_aggregation>(quantiles, interp);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_quantile_aggregation<aggregation>(
  std::vector<double> const& quantiles, interpolation interp);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_quantile_aggregation<groupby_aggregation>(std::vector<double> const& quantiles,
                                               interpolation interp);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_quantile_aggregation<reduce_aggregation>(std::vector<double> const& quantiles,
                                              interpolation interp);

/// Factory to create an ARGMAX aggregation
template <typename Base>
std::unique_ptr<Base> make_argmax_aggregation()
{
  return std::make_unique<detail::argmax_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_argmax_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_argmax_aggregation<rolling_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_argmax_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_argmax_aggregation<reduce_aggregation>();

/// Factory to create an ARGMIN aggregation
template <typename Base>
std::unique_ptr<Base> make_argmin_aggregation()
{
  return std::make_unique<detail::argmin_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_argmin_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_argmin_aggregation<rolling_aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_argmin_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_argmin_aggregation<reduce_aggregation>();

/// Factory to create an NUNIQUE aggregation
template <typename Base>
std::unique_ptr<Base> make_nunique_aggregation(null_policy null_handling)
{
  return std::make_unique<detail::nunique_aggregation>(null_handling);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_nunique_aggregation<aggregation>(
  null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_nunique_aggregation<groupby_aggregation>(null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_nunique_aggregation<reduce_aggregation>(null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<segmented_reduce_aggregation>
make_nunique_aggregation<segmented_reduce_aggregation>(null_policy null_handling);

/// Factory to create an NTH_ELEMENT aggregation
template <typename Base>
std::unique_ptr<Base> make_nth_element_aggregation(size_type n, null_policy null_handling)
{
  return std::make_unique<detail::nth_element_aggregation>(n, null_handling);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_nth_element_aggregation<aggregation>(
  size_type n, null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_nth_element_aggregation<groupby_aggregation>(size_type n, null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_nth_element_aggregation<reduce_aggregation>(size_type n, null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_nth_element_aggregation<rolling_aggregation>(size_type n, null_policy null_handling);

/// Factory to create a ROW_NUMBER aggregation
template <typename Base>
std::unique_ptr<Base> make_row_number_aggregation()
{
  return std::make_unique<detail::row_number_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_row_number_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_row_number_aggregation<rolling_aggregation>();

/// Factory to create an EWMA aggregation
template <typename Base>
std::unique_ptr<Base> make_ewma_aggregation(double const com, cudf::ewm_history history)
{
  return std::make_unique<detail::ewma_aggregation>(com, history);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_ewma_aggregation<aggregation>(
  double const com, cudf::ewm_history history);
template CUDF_EXPORT std::unique_ptr<scan_aggregation> make_ewma_aggregation<scan_aggregation>(
  double const com, cudf::ewm_history history);

/// Factory to create a RANK aggregation
template <typename Base>
std::unique_ptr<Base> make_rank_aggregation(rank_method method,
                                            order column_order,
                                            null_policy null_handling,
                                            null_order null_precedence,
                                            rank_percentage percentage)
{
  return std::make_unique<detail::rank_aggregation>(
    method, column_order, null_handling, null_precedence, percentage);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_rank_aggregation<aggregation>(
  rank_method method,
  order column_order,
  null_policy null_handling,
  null_order null_precedence,
  rank_percentage percentage);
template CUDF_EXPORT std::unique_ptr<groupby_scan_aggregation>
make_rank_aggregation<groupby_scan_aggregation>(rank_method method,
                                                order column_order,
                                                null_policy null_handling,
                                                null_order null_precedence,
                                                rank_percentage percentage);
template CUDF_EXPORT std::unique_ptr<scan_aggregation> make_rank_aggregation<scan_aggregation>(
  rank_method method,
  order column_order,
  null_policy null_handling,
  null_order null_precedence,
  rank_percentage percentage);

/// Factory to create a COLLECT_LIST aggregation
template <typename Base>
std::unique_ptr<Base> make_collect_list_aggregation(null_policy null_handling)
{
  return std::make_unique<detail::collect_list_aggregation>(null_handling);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_collect_list_aggregation<aggregation>(
  null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_collect_list_aggregation<rolling_aggregation>(null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_collect_list_aggregation<groupby_aggregation>(null_policy null_handling);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_collect_list_aggregation<reduce_aggregation>(null_policy null_handling);

/// Factory to create a COLLECT_SET aggregation
template <typename Base>
std::unique_ptr<Base> make_collect_set_aggregation(null_policy null_handling,
                                                   null_equality nulls_equal,
                                                   nan_equality nans_equal)
{
  return std::make_unique<detail::collect_set_aggregation>(null_handling, nulls_equal, nans_equal);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_collect_set_aggregation<aggregation>(
  null_policy null_handling, null_equality nulls_equal, nan_equality nans_equal);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_collect_set_aggregation<rolling_aggregation>(null_policy null_handling,
                                                  null_equality nulls_equal,
                                                  nan_equality nans_equal);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_collect_set_aggregation<groupby_aggregation>(null_policy null_handling,
                                                  null_equality nulls_equal,
                                                  nan_equality nans_equal);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_collect_set_aggregation<reduce_aggregation>(null_policy null_handling,
                                                 null_equality nulls_equal,
                                                 nan_equality nans_equal);

/// Factory to create a LAG aggregation
template <typename Base>
std::unique_ptr<Base> make_lag_aggregation(size_type offset)
{
  return std::make_unique<detail::lead_lag_aggregation>(aggregation::LAG, offset);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_lag_aggregation<aggregation>(
  size_type offset);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation> make_lag_aggregation<rolling_aggregation>(
  size_type offset);

/// Factory to create a LEAD aggregation
template <typename Base>
std::unique_ptr<Base> make_lead_aggregation(size_type offset)
{
  return std::make_unique<detail::lead_lag_aggregation>(aggregation::LEAD, offset);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_lead_aggregation<aggregation>(
  size_type offset);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation>
make_lead_aggregation<rolling_aggregation>(size_type offset);

/// Factory to create a UDF aggregation
template <typename Base>
std::unique_ptr<Base> make_udf_aggregation(udf_source_type type,
                                           std::string const& user_defined_aggregator,
                                           data_type output_type)
{
  auto* a =
    new detail::udf_aggregation{type == udf_source_type::PTX ? aggregation::PTX : aggregation::CUDA,
                                user_defined_aggregator,
                                output_type};
  return std::unique_ptr<detail::udf_aggregation>(a);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_udf_aggregation<aggregation>(
  udf_source_type type, std::string const& user_defined_aggregator, data_type output_type);
template CUDF_EXPORT std::unique_ptr<rolling_aggregation> make_udf_aggregation<rolling_aggregation>(
  udf_source_type type, std::string const& user_defined_aggregator, data_type output_type);

/// Factory to create a MERGE_LISTS aggregation
template <typename Base>
std::unique_ptr<Base> make_merge_lists_aggregation()
{
  return std::make_unique<detail::merge_lists_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_merge_lists_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_merge_lists_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_merge_lists_aggregation<reduce_aggregation>();

/// Factory to create a MERGE_SETS aggregation
template <typename Base>
std::unique_ptr<Base> make_merge_sets_aggregation(null_equality nulls_equal,
                                                  nan_equality nans_equal)
{
  return std::make_unique<detail::merge_sets_aggregation>(nulls_equal, nans_equal);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_merge_sets_aggregation<aggregation>(
  null_equality, nan_equality);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
  make_merge_sets_aggregation<groupby_aggregation>(null_equality, nan_equality);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
  make_merge_sets_aggregation<reduce_aggregation>(null_equality, nan_equality);

/// Factory to create a MERGE_M2 aggregation
template <typename Base>
std::unique_ptr<Base> make_merge_m2_aggregation()
{
  return std::make_unique<detail::merge_m2_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_merge_m2_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_merge_m2_aggregation<groupby_aggregation>();

/// Factory to create a MERGE_HISTOGRAM aggregation
template <typename Base>
std::unique_ptr<Base> make_merge_histogram_aggregation()
{
  return std::make_unique<detail::merge_histogram_aggregation>();
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_merge_histogram_aggregation<aggregation>();
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_merge_histogram_aggregation<groupby_aggregation>();
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_merge_histogram_aggregation<reduce_aggregation>();

/// Factory to create a COVARIANCE aggregation
template <typename Base>
std::unique_ptr<Base> make_covariance_aggregation(size_type min_periods, size_type ddof)
{
  return std::make_unique<detail::covariance_aggregation>(min_periods, ddof);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_covariance_aggregation<aggregation>(
  size_type min_periods, size_type ddof);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_covariance_aggregation<groupby_aggregation>(size_type min_periods, size_type ddof);

/// Factory to create a CORRELATION aggregation
template <typename Base>
std::unique_ptr<Base> make_correlation_aggregation(correlation_type type, size_type min_periods)
{
  return std::make_unique<detail::correlation_aggregation>(type, min_periods);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_correlation_aggregation<aggregation>(
  correlation_type type, size_type min_periods);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_correlation_aggregation<groupby_aggregation>(correlation_type type, size_type min_periods);

template <typename Base>
std::unique_ptr<Base> make_tdigest_aggregation(int max_centroids)
{
  return std::make_unique<detail::tdigest_aggregation>(max_centroids);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_tdigest_aggregation<aggregation>(
  int max_centroids);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_tdigest_aggregation<groupby_aggregation>(int max_centroids);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_tdigest_aggregation<reduce_aggregation>(int max_centroids);

template <typename Base>
std::unique_ptr<Base> make_merge_tdigest_aggregation(int max_centroids)
{
  return std::make_unique<detail::merge_tdigest_aggregation>(max_centroids);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_merge_tdigest_aggregation<aggregation>(
  int max_centroids);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_merge_tdigest_aggregation<groupby_aggregation>(int max_centroids);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_merge_tdigest_aggregation<reduce_aggregation>(int max_centroids);

template <typename Base>
std::unique_ptr<Base> make_bitwise_aggregation(bitwise_op bit_op)
{
  return std::make_unique<detail::bitwise_aggregation>(bit_op);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_bitwise_aggregation<aggregation>(
  bitwise_op bit_op);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_bitwise_aggregation<groupby_aggregation>(bitwise_op bit_op);
template CUDF_EXPORT std::unique_ptr<reduce_aggregation>
make_bitwise_aggregation<reduce_aggregation>(bitwise_op bit_op);

template <typename Base>
std::unique_ptr<Base> make_top_k_aggregation(size_type k, order topk_order)
{
  return std::make_unique<detail::top_k_aggregation>(k, topk_order);
}
template CUDF_EXPORT std::unique_ptr<aggregation> make_top_k_aggregation<aggregation>(
  size_type k, order topk_order);
template CUDF_EXPORT std::unique_ptr<groupby_aggregation>
make_top_k_aggregation<groupby_aggregation>(size_type k, order topk_order);

namespace detail {
namespace {
struct target_type_functor {
  data_type type;
  template <typename Source, aggregation::Kind k>
  constexpr data_type operator()() const noexcept
  {
    using Type    = target_type_t<Source, k>;
    auto const id = type_to_id<Type>();
    return cudf::is_fixed_point<Type>() ? data_type{id, type.scale()} : data_type{id};
  }
};

struct is_valid_aggregation_impl {
  template <typename Source, aggregation::Kind k>
  constexpr bool operator()() const noexcept
  {
    return is_valid_aggregation<Source, k>();
  }
};
}  // namespace

// Return target data_type for the given source_type and aggregation
data_type target_type(data_type source, aggregation::Kind k)
{
  return dispatch_type_and_aggregation(source, k, target_type_functor{source});
}

// Verifies the aggregation `k` is valid on the type `source`
bool is_valid_aggregation(data_type source, aggregation::Kind k)
{
  return dispatch_type_and_aggregation(source, k, is_valid_aggregation_impl{});
}
}  // namespace detail
bool is_valid_aggregation(data_type source, aggregation::Kind k)
{
  return detail::is_valid_aggregation(source, k);
}
}  // namespace cudf
