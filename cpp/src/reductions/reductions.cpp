/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/detail/histogram.hpp>
#include <cudf/reduction/detail/reduction_functions.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <utility>

namespace cudf {
namespace reduction {
namespace detail {
namespace {

/**
 * @brief Parameters for reduction functions
 *
 * Helps reduce code and allows updating the parameters without changing the function signatures.
 */
struct reduction_parameters {
  reduce_aggregation const& agg;
  column_view const& col;
  data_type const output_dtype;
  std::optional<std::reference_wrapper<scalar const>> init;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  reduction_parameters(reduce_aggregation const& agg,
                       column_view const& col,
                       data_type const output_dtype,
                       std::optional<std::reference_wrapper<scalar const>> init,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
    : agg(agg), col(col), output_dtype(output_dtype), init(init), stream(stream), mr(std::move(mr))
  {
  }
};

/**
 * @brief Base reduction function
 *
 * The member functions provide either default or the most common results.
 */
struct base_reduction_function {
  /// most common result is overridden by catch-all function template below
  [[nodiscard]] bool is_valid() const noexcept { return true; }

  /// overridden by derived classes to provide aggregation/type specific behavior
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const&) const
  {
    CUDF_FAIL("Unsupported reduction operator", std::invalid_argument);
  }

  /// default behavior for most aggregation/types when no input data is available to reduce
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return cudf::is_nested(params.output_dtype)
             ? make_empty_scalar_like(params.col, params.stream, params.mr)
             : make_default_constructed_scalar(params.output_dtype, params.stream, params.mr);
  }
};

/**
 * @brief Reduction function template
 *
 * This is the catch-all for all invalid reductions.
 */
template <typename Source, aggregation::Kind k>
struct reduction_function : public base_reduction_function {
  [[nodiscard]] bool is_valid() const noexcept { return false; }
};

template <typename Source>
  requires(cudf::is_numeric<Source>() or cudf::is_fixed_point<Source>())
struct reduction_function<Source, cudf::aggregation::SUM> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return sum(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
};

template <cudf::detail::sum_overflow_supported Source>
struct reduction_function<Source, cudf::aggregation::SUM_OVERFLOW>
  : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return sum_overflow(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return sum_overflow(params.col, params.output_dtype, std::nullopt, params.stream, params.mr);
  }
};

template <typename Source>
  requires(cudf::is_numeric<Source>() or cudf::is_fixed_point<Source>())
struct reduction_function<Source, cudf::aggregation::PRODUCT> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return product(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::MIN> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return min(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::MAX> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return max(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::ARGMIN> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    CUDF_EXPECTS(params.output_dtype.id() == type_to_id<size_type>(),
                 "ARGMIN aggregation expects output type to be cudf::size_type",
                 cudf::data_type_error);
    return argmin(params.col, data_type{type_to_id<Source>()}, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::ARGMAX> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    CUDF_EXPECTS(params.output_dtype.id() == type_to_id<size_type>(),
                 "ARGMAX aggregation expects output type to be cudf::size_type",
                 cudf::data_type_error);
    return argmax(params.col, data_type{type_to_id<Source>()}, params.stream, params.mr);
  }
};

template <typename Source>
  requires(std::is_arithmetic_v<Source>)
struct reduction_function<Source, cudf::aggregation::ANY> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return any(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return std::make_unique<numeric_scalar<bool>>(false, true, params.stream, params.mr);
  }
};

template <typename Source>
  requires(std::is_arithmetic_v<Source>)
struct reduction_function<Source, cudf::aggregation::ALL> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return all(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return std::make_unique<numeric_scalar<bool>>(true, true, params.stream, params.mr);
  }
};

template <typename Source>
  requires(cudf::is_numeric<Source>() or cudf::is_fixed_point<Source>())
struct reduction_function<Source, cudf::aggregation::SUM_OF_SQUARES>
  : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return sum_of_squares(params.col, params.output_dtype, params.stream, params.mr);
  }
};

template <typename Source>
  requires(std::is_arithmetic_v<Source>)
struct reduction_function<Source, cudf::aggregation::MEAN> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return mean(params.col, params.output_dtype, params.stream, params.mr);
  }
};

template <typename Source>
  requires(std::is_arithmetic_v<Source>)
struct reduction_function<Source, cudf::aggregation::STD> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto std_agg = static_cast<cudf::detail::std_aggregation const&>(params.agg);
    return standard_deviation(
      params.col, params.output_dtype, std_agg._ddof, params.stream, params.mr);
  }
};

template <typename Source>
  requires(std::is_arithmetic_v<Source>)
struct reduction_function<Source, cudf::aggregation::VARIANCE> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto var_agg = static_cast<cudf::detail::var_aggregation const&>(params.agg);
    return variance(params.col, params.output_dtype, var_agg._ddof, params.stream, params.mr);
  }
};

template <typename Source>
  requires(std::is_arithmetic_v<Source> or cudf::is_fixed_point<Source>())
struct reduction_function<Source, cudf::aggregation::MEDIAN> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return quantile(
      params.col, 0.5, interpolation::LINEAR, params.output_dtype, params.stream, params.mr);
  }
};

template <typename Source>
  requires(std::is_arithmetic_v<Source> or cudf::is_fixed_point<Source>())
struct reduction_function<Source, cudf::aggregation::QUANTILE> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto qagg = static_cast<cudf::detail::quantile_aggregation const&>(params.agg);
    CUDF_EXPECTS(qagg._quantiles.size() == 1,
                 "Reduction quantile accepts only one quantile value",
                 std::invalid_argument);
    return quantile(params.col,
                    qagg._quantiles.front(),
                    qagg._interpolation,
                    params.output_dtype,
                    params.stream,
                    params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::COUNT_ALL> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return count(params.col, null_policy::INCLUDE, params.output_dtype, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return reduce(params);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::COUNT_VALID> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return count(params.col, null_policy::EXCLUDE, params.output_dtype, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return reduce(params);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::NUNIQUE> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto nunique_agg = static_cast<cudf::detail::nunique_aggregation const&>(params.agg);
    return nunique(
      params.col, nunique_agg._null_handling, params.output_dtype, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    auto const nunique_agg = static_cast<cudf::detail::nunique_aggregation const&>(params.agg);
    auto const is_empty    = params.col.is_empty();
    auto const valid = !is_empty && (nunique_agg._null_handling == cudf::null_policy::INCLUDE);
    return std::make_unique<numeric_scalar<size_type>>(!is_empty, valid, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::NTH_ELEMENT> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto nth_agg = static_cast<cudf::detail::nth_element_aggregation const&>(params.agg);
    return nth_element(params.col, nth_agg._n, nth_agg._null_handling, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::HISTOGRAM> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return histogram(params.col, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return std::make_unique<list_scalar>(
      std::move(*reduction::detail::make_empty_histogram_like(params.col)),
      true,
      params.stream,
      params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::MERGE_HISTOGRAM>
  : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return merge_histogram(params.col, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return std::make_unique<list_scalar>(
      std::move(*reduction::detail::make_empty_histogram_like(params.col.child(0))),
      true,
      params.stream,
      params.mr);
  }
};

template <typename Source>
  requires(std::is_integral_v<Source>)
struct reduction_function<Source, cudf::aggregation::BITWISE_AGG> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto bitwise_agg = static_cast<cudf::detail::bitwise_aggregation const&>(params.agg);
    return bitwise_reduction(bitwise_agg.bit_op, params.col, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::COLLECT_LIST>
  : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto col_agg = static_cast<cudf::detail::collect_list_aggregation const&>(params.agg);
    return collect_list(params.col, col_agg._null_handling, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    auto scalar = make_list_scalar(empty_like(params.col)->view(), params.stream, params.mr);
    scalar->set_valid_async(false, params.stream);
    return scalar;
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::COLLECT_SET> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto col_agg = static_cast<cudf::detail::collect_set_aggregation const&>(params.agg);
    return collect_set(params.col,
                       col_agg._null_handling,
                       col_agg._nulls_equal,
                       col_agg._nans_equal,
                       params.stream,
                       params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    auto scalar = make_list_scalar(empty_like(params.col)->view(), params.stream, params.mr);
    scalar->set_valid_async(false, params.stream);
    return scalar;
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::MERGE_LISTS> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    return merge_lists(params.col, params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::MERGE_SETS> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto col_agg = static_cast<cudf::detail::merge_sets_aggregation const&>(params.agg);
    return merge_sets(
      params.col, col_agg._nulls_equal, col_agg._nans_equal, params.stream, params.mr);
  }
};

template <typename Source>
  requires(cudf::is_numeric<Source>() or cudf::is_fixed_point<Source>())
struct reduction_function<Source, cudf::aggregation::TDIGEST> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    CUDF_EXPECTS(params.output_dtype.id() == type_id::STRUCT,
                 "Tdigest aggregations expect output type to be STRUCT",
                 std::invalid_argument);
    auto td_agg = static_cast<cudf::detail::tdigest_aggregation const&>(params.agg);
    return tdigest::detail::reduce_tdigest(
      params.col, td_agg.max_centroids, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return tdigest::detail::make_empty_tdigest_scalar(params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::MERGE_TDIGEST>
  : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    CUDF_EXPECTS(params.output_dtype.id() == type_id::STRUCT,
                 "Tdigest aggregations expect output type to be STRUCT",
                 std::invalid_argument);
    auto td_agg = static_cast<cudf::detail::merge_tdigest_aggregation const&>(params.agg);
    return tdigest::detail::reduce_merge_tdigest(
      params.col, td_agg.max_centroids, params.stream, params.mr);
  }
  [[nodiscard]] std::unique_ptr<scalar> reduce_no_data(reduction_parameters const& params) const
  {
    return tdigest::detail::make_empty_tdigest_scalar(params.stream, params.mr);
  }
};

template <typename Source>
struct reduction_function<Source, cudf::aggregation::HOST_UDF> : public base_reduction_function {
  [[nodiscard]] std::unique_ptr<scalar> reduce(reduction_parameters const& params) const
  {
    auto const& udf_base_ptr =
      dynamic_cast<cudf::detail::host_udf_aggregation const&>(params.agg).udf_ptr;
    auto const udf_ptr = dynamic_cast<reduce_host_udf const*>(udf_base_ptr.get());
    CUDF_EXPECTS(
      udf_ptr != nullptr, "Invalid HOST_UDF instance for reduction.", std::invalid_argument);
    return (*udf_ptr)(params.col, params.output_dtype, params.init, params.stream, params.mr);
  }
};

struct reduction_functions_aggregator {
  template <typename Source, cudf::aggregation::Kind k>
  [[nodiscard]] std::unique_ptr<scalar> operator()(reduction_parameters const& params) const
  {
    return reduction_function<Source, k>{}.reduce(params);
  }
};

struct reduction_functions_no_data {
  template <typename Source, cudf::aggregation::Kind k>
  [[nodiscard]] std::unique_ptr<scalar> operator()(reduction_parameters const& params) const
  {
    return reduction_function<Source, k>{}.reduce_no_data(params);
  }
};

struct reduction_functions_is_valid {
  template <typename Source, cudf::aggregation::Kind k>
  [[nodiscard]] bool operator()() const
  {
    return reduction_function<Source, k>{}.is_valid();
  }
};

}  // namespace

std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!init.has_value() || cudf::have_same_types(col, init.value().get()),
               "column and initial value must be the same type",
               cudf::data_type_error);
  if (init.has_value() && !(agg.kind == aggregation::SUM || agg.kind == aggregation::SUM_OVERFLOW ||
                            agg.kind == aggregation::PRODUCT || agg.kind == aggregation::MIN ||
                            agg.kind == aggregation::MAX || agg.kind == aggregation::ANY ||
                            agg.kind == aggregation::ALL || agg.kind == aggregation::HOST_UDF)) {
    CUDF_FAIL(
      "Initial value is only supported for SUM, SUM_OVERFLOW, PRODUCT, MIN, MAX, ANY, ALL, "
      "and HOST_UDF aggregation types",
      std::invalid_argument);
  }

  auto const dt =
    cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type() : col.type();
  auto const params = reduction_parameters(agg, col, output_dtype, init, stream, mr);

  return (col.size() == col.null_count())
           ? cudf::detail::dispatch_type_and_aggregation(
               dt, agg.kind, reduction_functions_no_data{}, params)
           : cudf::detail::dispatch_type_and_aggregation(
               dt, agg.kind, reduction_functions_aggregator{}, params);
}

bool is_valid_aggregation(data_type source, aggregation::Kind kind)
{
  return cudf::detail::dispatch_type_and_aggregation(source, kind, reduction_functions_is_valid{});
}

}  // namespace detail
bool is_valid_aggregation(data_type source, aggregation::Kind kind)
{
  return detail::is_valid_aggregation(source, kind);
}
}  // namespace reduction

std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::reduce(col, agg, output_dtype, std::nullopt, stream, mr);
}

std::unique_ptr<scalar> reduce(column_view const& col,
                               reduce_aggregation const& agg,
                               data_type output_dtype,
                               std::optional<std::reference_wrapper<scalar const>> init,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return reduction::detail::reduce(col, agg, output_dtype, init, stream, mr);
}

}  // namespace cudf
