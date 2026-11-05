/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lto_benchmark.hpp"

#include <benchmarks/common/generate_input.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_reduce_lto_benchmark_fragments.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using cudf::data_type;
using cudf::size_type;
using cudf::type_id;
using columns = std::vector<std::unique_ptr<cudf::column>>;
using scalars = std::vector<std::unique_ptr<cudf::scalar>>;

cudf::lto_fragments precompiled_fragments()
{
  auto const range = cudf_reduce_lto_benchmark_fragments::file_ranges
    [cudf_reduce_lto_benchmark_fragments::tpcds_q13];
  static cudf::lto_fragment const result[]{
    {cudf::lto_binary_type::FATBIN,
     cudf_reduce_lto_benchmark_fragments::files.subspan(range[0], range[1])}};
  return result;
}

cudf::reduce_udf make_udf()
{
  return {precompiled_fragments(),
          "tpcds_q13_init",
          "tpcds_q13_row_update",
          "tpcds_q13_merge",
          "tpcds_q13_finalize",
          "tpcds_q13_filter"};
}

columns make_inputs(size_type rows, int32_t null_percent)
{
  columns input;

  auto add = [&](type_id type, double lower, double upper, bool nullable) {
    auto profile = data_profile_builder();
    profile.distribution(type, distribution_id::UNIFORM, lower, upper);
    if (nullable && null_percent != 0) {
      profile.null_probability(null_percent / 100.0);
    } else {
      profile.no_validity();
    }
    auto const seed = static_cast<unsigned>(input.size() + 1);
    input.push_back(create_random_column(type, row_count{rows}, profile, seed));
  };

  add(type_id::INT32, 1, 50, true);        // ss_quantity
  add(type_id::FLOAT64, 0, 1000, true);    // ss_ext_sales_price
  add(type_id::FLOAT64, 0, 800, true);     // ss_ext_wholesale_cost
  add(type_id::FLOAT64, 0, 200, false);    // ss_sales_price
  add(type_id::FLOAT64, 0, 300, false);    // ss_net_profit
  add(type_id::INT32, 0, 3, false);        // resolved demographic branch
  add(type_id::INT32, 0, 3, false);        // resolved address branch
  add(type_id::INT32, 1998, 2002, false);  // d_year

  return input;
}

cudf::ast::tree make_predicate(scalars& literals, cuda::stream_ref stream)
{
  namespace ast = cudf::ast;
  using op      = ast::ast_operator;
  ast::tree tree;

  auto literal = [&]<typename T>(T value) -> ast::literal const& {
    auto scalar        = std::make_unique<cudf::numeric_scalar<T>>(value, true, stream);
    auto const& result = tree.emplace<ast::literal>(*scalar);
    literals.push_back(std::move(scalar));
    return result;
  };

  auto branch = [&](ast::expression const& code,
                    int32_t expected,
                    ast::expression const& value,
                    double lower,
                    double upper) -> ast::operation const& {
    auto const& matches = tree.emplace<ast::operation>(op::EQUAL, code, literal(expected));
    auto const& minimum = tree.emplace<ast::operation>(op::GREATER_EQUAL, value, literal(lower));
    auto const& maximum = tree.emplace<ast::operation>(op::LESS_EQUAL, value, literal(upper));
    auto const& range   = tree.emplace<ast::operation>(op::LOGICAL_AND, minimum, maximum);
    return tree.emplace<ast::operation>(op::LOGICAL_AND, matches, range);
  };

  auto const& sales_price  = tree.emplace<ast::column_reference>(3);
  auto const& net_profit   = tree.emplace<ast::column_reference>(4);
  auto const& demographic  = tree.emplace<ast::column_reference>(5);
  auto const& address      = tree.emplace<ast::column_reference>(6);
  auto const& demographics = tree.emplace<ast::operation>(
    op::LOGICAL_OR,
    tree.emplace<ast::operation>(op::LOGICAL_OR,
                                 branch(demographic, 1, sales_price, 100, 150),
                                 branch(demographic, 2, sales_price, 50, 100)),
    branch(demographic, 3, sales_price, 150, 200));

  auto const& addresses = tree.emplace<ast::operation>(
    op::LOGICAL_OR,
    tree.emplace<ast::operation>(op::LOGICAL_OR,
                                 branch(address, 1, net_profit, 100, 200),
                                 branch(address, 2, net_profit, 150, 300)),
    branch(address, 3, net_profit, 50, 250));

  auto const& year = tree.emplace<ast::operation>(
    op::EQUAL, tree.emplace<ast::column_reference>(7), literal(int32_t{2001}));

  auto const& branches = tree.emplace<ast::operation>(op::LOGICAL_AND, demographics, addresses);
  tree.emplace<ast::operation>(op::LOGICAL_AND, branches, year);

  return tree;
}

scalars reduce_cudf(cudf::table_view values,
                    cudf::table_view all_inputs,
                    cudf::ast::expression const& predicate,
                    cudf::reduce_aggregation const& sum,
                    cudf::reduce_aggregation const& mean,
                    cuda::stream_ref stream,
                    bool jit)
{
  auto selection             = jit ? cudf::compute_column_jit(all_inputs, predicate, stream)
                                   : cudf::compute_column(all_inputs, predicate, stream);
  auto selected              = cudf::apply_retention_mask(values, selection->view(), stream);
  auto const selected_values = selected->view();
  data_type const result_type{type_id::FLOAT64};
  scalars result;

  for (size_type column = 0; column < selected_values.num_columns(); ++column) {
    result.push_back(cudf::reduce(selected_values.column(column), mean, result_type, stream));
  }
  result.push_back(cudf::reduce(selected_values.column(2), sum, result_type, stream));

  return result;
}

std::vector<std::optional<double>> host_scalars(scalars const& values, cuda::stream_ref stream)
{
  std::vector<std::optional<double>> result;
  for (auto const& value : values) {
    result.push_back(
      value->is_valid(stream)
        ? std::optional{static_cast<cudf::numeric_scalar<double> const&>(*value).value(stream)}
        : std::nullopt);
  }
  return result;
}
}  // namespace

namespace reduction_benchmark {
void run_q13(options const& settings, std::ostream& output)
{
  auto const row_counts = settings.rows ? std::vector<size_type>{*settings.rows}
                          : settings.smoke
                            ? std::vector<size_type>{0, 1000}
                            : std::vector<size_type>{1000, 100'000, 1'000'000, 10'000'000};
  auto const null_counts =
    settings.smoke ? std::vector<int32_t>{0, 10, 100} : std::vector<int32_t>{0, 10};
  for (auto rows : row_counts) {
    for (auto nulls : null_counts) {
      auto const stream = cudf::get_default_stream();
      auto input        = make_inputs(rows, nulls);
      std::vector<cudf::column_view> views;
      std::vector<cudf::reduce_input> arguments;
      double logical_bytes = 0;
      for (auto const& column : input) {
        views.push_back(column->view());
        arguments.emplace_back(column->view());
        logical_bytes +=
          rows * (cudf::size_of(column->type()) + (column->nullable() ? 1.0 / CHAR_BIT : 0.0));
      }
      cudf::table_view const all_inputs{views};
      constexpr int32_t value_inputs = 3;
      cudf::table_view const values{
        std::vector<cudf::column_view>{views.begin(), views.begin() + value_inputs}};
      scalars literals;
      auto predicate    = make_predicate(literals, stream);
      auto sum          = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto mean         = cudf::make_mean_aggregation<cudf::reduce_aggregation>();
      auto cudf_execute = [&](bool jit) {
        return reduce_cudf(values, all_inputs, predicate.back(), *sum, *mean, stream, jit);
      };
      // The public cuDF pipeline supplies the untimed reference.
      auto const expected = host_scalars(cudf_execute(false), stream);
      auto validate_cudf  = [&](scalars const& result) {
        expect_equal(host_scalars(result, stream), expected);
      };
      workload const work{"tpcds_q13_post_join", "scalar", rows, 1, nulls, "none", logical_bytes};
      measure(
        output,
        work,
        "cudf_ast",
        "scalars",
        stream,
        [&] { return cudf_execute(false); },
        validate_cudf);
      measure(
        output,
        work,
        "cudf_ast_jit",
        "scalars",
        stream,
        [&] { return cudf_execute(true); },
        validate_cudf);

      data_type const state_types[]{data_type{type_id::FLOAT64},
                                    data_type{type_id::FLOAT64},
                                    data_type{type_id::FLOAT64},
                                    data_type{type_id::INT64},
                                    data_type{type_id::INT64},
                                    data_type{type_id::INT64}};
      data_type const output_types[]{data_type{type_id::FLOAT64},
                                     data_type{type_id::FLOAT64},
                                     data_type{type_id::FLOAT64},
                                     data_type{type_id::FLOAT64}};
      auto const udf = make_udf();
      auto execute   = [&] {
        return cudf::reduce_lto(
          udf, cudf::null_aware::YES, arguments, state_types, output_types, stream);
      };
      auto validate = [&](columns const& result) {
        std::vector<std::optional<double>> actual;
        for (auto const& column : result) {
          auto values = host_values<double>(column->view(), stream);
          if (values.size() != 1) { throw std::runtime_error{"Expected one reduction output row"}; }
          actual.push_back(values.front());
        }
        expect_equal(actual, expected);
      };
      measure(output, work, "lto", "single_row_columns", stream, execute, validate);
    }
  }
}
}  // namespace reduction_benchmark
