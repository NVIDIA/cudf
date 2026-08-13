/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "extract_single_pass_aggs.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace cudf::groupby::detail::hash {

// Groupby-specific functor for collecting simple aggregations
struct simple_aggregation_collector {
  // Default case: return clone of the aggregation
  template <aggregation::Kind k>
  std::vector<std::unique_ptr<aggregation>> operator()(data_type col_type,
                                                       aggregation const& agg) const
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(agg.clone());
    return aggs;
  }
};

// Specialization for MIN aggregation
template <>
std::vector<std::unique_ptr<aggregation>>
simple_aggregation_collector::operator()<aggregation::MIN>(data_type col_type,
                                                           aggregation const&) const
{
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(col_type.id() == type_id::STRING ? make_argmin_aggregation()
                                                  : make_min_aggregation());
  return aggs;
}

// Specialization for MAX aggregation
template <>
std::vector<std::unique_ptr<aggregation>>
simple_aggregation_collector::operator()<aggregation::MAX>(data_type col_type,
                                                           aggregation const&) const
{
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(col_type.id() == type_id::STRING ? make_argmax_aggregation()
                                                  : make_max_aggregation());
  return aggs;
}

// Specialization for MEAN aggregation
template <>
std::vector<std::unique_ptr<aggregation>>
simple_aggregation_collector::operator()<aggregation::MEAN>(data_type col_type,
                                                            aggregation const&) const
{
  CUDF_EXPECTS(is_fixed_width(col_type), "MEAN aggregation expects fixed width type");
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(make_sum_aggregation());
  // COUNT_VALID
  aggs.push_back(make_count_aggregation());
  return aggs;
}

// Helper for M2/VARIANCE/STD - they all need the same simple aggregations
std::vector<std::unique_ptr<aggregation>> collect_m2_simple_aggs()
{
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(make_sum_of_squares_aggregation());
  aggs.push_back(make_sum_aggregation());
  // COUNT_VALID
  aggs.push_back(make_count_aggregation());
  return aggs;
}

// Specialization for M2 aggregation
template <>
std::vector<std::unique_ptr<aggregation>> simple_aggregation_collector::operator()<aggregation::M2>(
  data_type, aggregation const&) const
{
  return collect_m2_simple_aggs();
}

// Specialization for VARIANCE aggregation
template <>
std::vector<std::unique_ptr<aggregation>>
simple_aggregation_collector::operator()<aggregation::VARIANCE>(data_type, aggregation const&) const
{
  return collect_m2_simple_aggs();
}

// Specialization for STD aggregation
template <>
std::vector<std::unique_ptr<aggregation>>
simple_aggregation_collector::operator()<aggregation::STD>(data_type, aggregation const&) const
{
  return collect_m2_simple_aggs();
}

std::tuple<table_view,
           cudf::detail::host_vector<aggregation::Kind>,
           std::vector<std::unique_ptr<aggregation>>,
           std::vector<int8_t>,
           bool>
extract_single_pass_aggs(std::span<aggregation_request const> requests,
                         rmm::cuda_stream_view stream)
{
  auto agg_kinds = cudf::detail::make_empty_host_vector<aggregation::Kind>(requests.size(), stream);
  std::vector<column_view> columns;
  std::vector<std::unique_ptr<aggregation>> aggs;
  std::vector<int8_t> is_agg_intermediate;
  columns.reserve(requests.size());
  aggs.reserve(requests.size());
  is_agg_intermediate.reserve(requests.size());

  bool has_compound_aggs = false;
  for (auto const& request : requests) {
    auto const& input_aggs = request.aggregations;

    // Map aggregation kind to:
    // - INPUT_NOT_EXTRACTED: aggregation requested by the users, not yet extracted
    // - INPUT_EXTRACTED: aggregation requested by the users and has already been extracted
    // - INTERMEDIATE: extracted intermediate aggregation
    enum class aggregation_group : int8_t { INPUT_NOT_EXTRACTED, INPUT_EXTRACTED, INTERMEDIATE };
    std::unordered_map<aggregation::Kind, aggregation_group> agg_kinds_map;
    for (auto const& agg : input_aggs) {
      agg_kinds_map[agg->kind] = aggregation_group::INPUT_NOT_EXTRACTED;
    }

    auto insert_agg = [&](column_view const& request_values, std::unique_ptr<aggregation>&& agg) {
      auto const it = agg_kinds_map.find(agg->kind);
      auto const need_update =
        it == agg_kinds_map.end() || it->second == aggregation_group::INPUT_NOT_EXTRACTED;
      if (!need_update) { return; }

      auto const is_intermediate = it == agg_kinds_map.end();
      agg_kinds_map[agg->kind] =
        is_intermediate ? aggregation_group::INTERMEDIATE : aggregation_group::INPUT_EXTRACTED;
      is_agg_intermediate.push_back(static_cast<int8_t>(is_intermediate));

      agg_kinds.push_back(agg->kind);
      aggs.push_back(std::move(agg));
      columns.push_back(request_values);
    };

    auto const values_type = cudf::is_dictionary(request.values.type())
                               ? cudf::dictionary_column_view(request.values).keys().type()
                               : request.values.type();
    for (auto const& agg : input_aggs) {
      auto spass_aggs = cudf::detail::aggregation_dispatcher(
        agg->kind, simple_aggregation_collector{}, values_type, *agg);
      if (spass_aggs.size() > 1 || !spass_aggs.front()->is_equal(*agg)) {
        has_compound_aggs = true;
      }

      for (auto& agg_s : spass_aggs) {
        insert_agg(request.values, std::move(agg_s));
      }
    }
  }

  return {table_view(columns),
          std::move(agg_kinds),
          std::move(aggs),
          std::move(is_agg_intermediate),
          has_compound_aggs};
}

std::vector<aggregation::Kind> get_simple_aggregations(groupby_aggregation const& agg,
                                                       data_type values_type)
{
  auto aggs = cudf::detail::aggregation_dispatcher(
    agg.kind, simple_aggregation_collector{}, values_type, agg);
  std::vector<aggregation::Kind> agg_kinds;
  std::transform(
    aggs.begin(), aggs.end(), std::back_inserter(agg_kinds), [](auto const& a) { return a->kind; });
  return agg_kinds;
}

}  // namespace cudf::groupby::detail::hash
