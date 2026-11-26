/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/testing_main.hpp>

namespace cudf::test::detail {

void initialize_test_compiled(int argc, char** argv, config const& config_override)
{
  initialize_test_impl(argc, argv, config_override);
}

int run_test_main_compiled(int argc, char** argv) { return run_test_main_impl(argc, argv); }

}  // namespace cudf::test::detail
