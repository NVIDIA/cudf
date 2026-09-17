/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_jni_apis.hpp"
#include "jni_compiled_expr.hpp"

#include <cudf/transform.hpp>

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_ast_AstJitProgram_create(JNIEnv* env,
                                                                     jclass,
                                                                     jlongArray j_asts,
                                                                     jlong j_table)
{
  JNI_NULL_CHECK(env, j_asts, "Compiled AST pointer array is null", 0);
  JNI_NULL_CHECK(env, j_table, "Table view pointer is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray ast_handles(env, j_asts);
    if (ast_handles.size() == 0) { throw std::invalid_argument("At least one AST is required"); }

    std::vector<std::reference_wrapper<cudf::ast::expression const>> expressions;
    expressions.reserve(ast_handles.size());
    auto has_literals = false;
    for (auto const handle : ast_handles) {
      if (handle == 0) { throw std::invalid_argument("Compiled AST pointer is null"); }
      auto const* compiled_expr_ptr =
        reinterpret_cast<cudf::jni::ast::compiled_expr const*>(handle);
      expressions.emplace_back(compiled_expr_ptr->get_jit_top_expression());
      has_literals |= compiled_expr_ptr->has_jit_literals();
    }
    ast_handles.cancel();

    auto const* table = reinterpret_cast<cudf::table_view const*>(j_table);
    auto const stream = cudf::get_default_stream();
    auto program      = std::make_unique<cudf::transform_program>(*table, expressions, stream);
    // Construction inputs may be released by a thread with a different default stream.
    if (has_literals) { stream.sync(); }
    return reinterpret_cast<jlong>(program.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_ai_rapids_cudf_ast_AstJitProgram_computeTableNative(
  JNIEnv* env, jclass, jlong j_program, jlong j_table)
{
  JNI_NULL_CHECK(env, j_program, "AST JIT program pointer is null", nullptr);
  JNI_NULL_CHECK(env, j_table, "Table view pointer is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* program     = reinterpret_cast<cudf::transform_program*>(j_program);
    auto const* table = reinterpret_cast<cudf::table_view const*>(j_table);
    return cudf::jni::convert_table_for_return(env, program->run(*table));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_ast_AstJitProgram_destroy(JNIEnv* env,
                                                                     jclass,
                                                                     jlong j_program)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::transform_program*>(j_program);
  }
  JNI_CATCH(env, );
}

}  // extern "C"
