/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace {

enum class transform_api { TRANSFORM, TRANSFORM_PROGRAM, AST, AST_TRANSFORM_PROGRAM };

transform_api api_from_string(std::string_view api)
{
  if (api == "transform") { return transform_api::TRANSFORM; }
  if (api == "transform_program") { return transform_api::TRANSFORM_PROGRAM; }
  if (api == "ast") { return transform_api::AST; }
  if (api == "ast_transform_program") { return transform_api::AST_TRANSFORM_PROGRAM; }
  CUDF_FAIL("Unrecognized transform API: " + std::string{api});
}

std::string make_udf(cudf::size_type expression_depth)
{
  std::string expression = "in + in";
  for (cudf::size_type level = 1; level < expression_depth; ++level) {
    expression = "(" + expression + ") + in";
  }
  return "__device__ void transform(int32_t* out, int32_t in) { *out = " + expression + "; }";
}

// Measures warm dispatch overhead for equivalent CUDA UDF and AST expressions through both the
// one-shot APIs and a reusable transform_program. Program construction and initial JIT compilation
// happen before the timed region; varying row count separates host-side dispatch costs from kernel
// execution, while expression depth exposes repeated AST lowering and kernel-lookup costs.
void BM_transform_dispatch(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const expression_depth = static_cast<cudf::size_type>(state.get_int64("expression_depth"));
  auto const api              = api_from_string(state.get_string("api"));
  auto source_table =
    create_sequence_table({cudf::type_id::INT32}, row_count{num_rows}, std::nullopt);
  auto const table = source_table->view();

  std::array<cudf::transform_input, 1> inputs{source_table->get_column(0).view()};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::INT32},
                                            cudf::output_nullability::ALL_VALID}};
  auto const udf = make_udf(expression_depth);

  cudf::ast::tree tree;
  tree.push(cudf::ast::column_reference{0});
  tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, tree.at(0), tree.at(0)});
  for (cudf::size_type level = 1; level < expression_depth; ++level) {
    tree.push(cudf::ast::operation{cudf::ast::ast_operator::ADD, tree.back(), tree.at(0)});
  }
  auto const& expression = tree.back();

  std::unique_ptr<cudf::transform_program> program;
  switch (api) {
    case transform_api::TRANSFORM:
      cudf::transform(udf,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      inputs,
                      outputs,
                      {},
                      std::nullopt);
      break;
    case transform_api::TRANSFORM_PROGRAM:
      program =
        std::make_unique<cudf::transform_program>(udf,
                                                  cudf::udf_source_type::CUDA,
                                                  cudf::null_aware::NO,
                                                  std::nullopt,
                                                  inputs,
                                                  outputs,
                                                  std::span<std::unique_ptr<cudf::column> const>{});
      break;
    case transform_api::AST: cudf::compute_column_jit(table, expression); break;
    case transform_api::AST_TRANSFORM_PROGRAM: {
      std::reference_wrapper<cudf::ast::expression const> expressions[] = {expression};
      program = std::make_unique<cudf::transform_program>(table, expressions);
      break;
    }
  }

  state.add_global_memory_reads<int32_t>(static_cast<std::size_t>(num_rows));
  state.add_global_memory_writes<int32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    switch (api) {
      case transform_api::TRANSFORM:
        cudf::transform(udf,
                        cudf::udf_source_type::CUDA,
                        cudf::null_aware::NO,
                        std::nullopt,
                        inputs,
                        outputs,
                        {},
                        std::nullopt,
                        launch.get_stream().get_stream());
        break;
      case transform_api::TRANSFORM_PROGRAM:
        program->run(inputs, outputs, {}, std::nullopt, launch.get_stream().get_stream());
        break;
      case transform_api::AST:
        cudf::compute_column_jit(table, expression, launch.get_stream().get_stream());
        break;
      case transform_api::AST_TRANSFORM_PROGRAM:
        program->run(table, launch.get_stream().get_stream());
        break;
    }
  });
}

}  // namespace

NVBENCH_BENCH(BM_transform_dispatch)
  .set_name("transform_dispatch")
  .add_string_axis("api", {"transform", "transform_program", "ast", "ast_transform_program"})
  .add_int64_axis("expression_depth", {4, 16, 64})
  .add_int64_axis("num_rows", {1});
