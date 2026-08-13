/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <jit/helpers.hpp>

#include <variant>
#include <vector>

namespace cudf {

namespace detail {
std::unique_ptr<table> filter(std::string const& predicate_udf,
                              cudf::udf_source_type source_type,
                              null_aware is_null_aware,
                              std::optional<void*> user_data,
                              std::span<transform_input const> predicate_inputs,
                              table_view const& filter_table,
                              output_nullability predicate_nullability,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(filter_table.num_columns() > 0,
               "At least one column must be provided to filter.",
               std::invalid_argument);
  auto row_size = filter_table.num_rows();
  CUDF_EXPECTS(std::all_of(filter_table.begin(),
                           filter_table.end(),
                           [&](auto const& col) { return col.size() == row_size; }),
               "All columns to filter must have the same number of rows.",
               std::invalid_argument);
  CUDF_EXPECTS(std::all_of(predicate_inputs.begin(),
                           predicate_inputs.end(),
                           [&](auto& input) {
                             if (auto* col = std::get_if<column_view>(&input)) {
                               return col->size() == row_size;
                             }
                             return true;
                           }),
               "All predicate input columns must have the same number of rows as the filter table.",
               std::invalid_argument);

  transform_output outputs[] = {transform_output{data_type{type_id::BOOL8}, predicate_nullability}};

  auto result = cudf::transform(predicate_udf,
                                source_type,
                                is_null_aware,
                                user_data,
                                predicate_inputs,
                                outputs,
                                {},
                                filter_table.num_rows(),
                                stream,
                                mr);

  return apply_mask(filter_table, result->get_column(0), mask_type::RETENTION, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> filter(table_view const& predicate_table,
                              ast::expression const& predicate_expr,
                              table_view const& filter_table,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  auto args = cudf::detail::row_ir::ast_converter::filter(cudf::detail::row_ir::target::CUDA,
                                                          predicate_expr,
                                                          predicate_table,
                                                          {},
                                                          "filter_operation",
                                                          stream,
                                                          mr);

  return detail::filter(args.udf,
                        args.source_type,
                        args.is_null_aware,
                        args.user_data,
                        args.inputs,
                        filter_table,
                        args.outputs[0].nullability,
                        stream,
                        mr);
}

std::vector<std::unique_ptr<column>> filter_extended(
  std::span<std::variant<column_view, scalar_column_view> const> predicate_inputs,
  std::string const& predicate_udf,
  std::vector<column_view> const& filter_columns,
  cudf::udf_source_type source_type,
  std::optional<void*> user_data,
  null_aware is_null_aware,
  output_nullability predicate_nullability,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto table = detail::filter(predicate_udf,
                              source_type,
                              is_null_aware,
                              user_data,
                              predicate_inputs,
                              table_view{filter_columns},
                              predicate_nullability,
                              stream,
                              mr);
  return table->release();
}

std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& predicate_columns,
                                            std::string const& predicate_udf,
                                            std::vector<column_view> const& filter_columns,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            null_aware is_null_aware,
                                            output_nullability predicate_nullability,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  // legacy behavior was to detect which column were scalars based on their sizes
  std::vector<std::variant<column_view, scalar_column_view>> inputs;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto base_column = jit::get_transform_base_column(predicate_columns);
  for (auto const& col : predicate_columns) {
    if (jit::is_scalar(base_column->size(), col.size())) {
#pragma GCC diagnostic pop
      inputs.emplace_back(scalar_column_view{col});
    } else {
      inputs.emplace_back(col);
    }
  }

  auto table = detail::filter(predicate_udf,
                              is_ptx ? cudf::udf_source_type::PTX : cudf::udf_source_type::CUDA,
                              is_null_aware,
                              user_data,
                              inputs,
                              table_view{filter_columns},
                              predicate_nullability,
                              stream,
                              mr);
  return table->release();
}

}  // namespace cudf
