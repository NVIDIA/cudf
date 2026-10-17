/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/fixture/nvbench_fixture.hpp>

#include <nvbench/main.cuh>

#include <string>
#include <vector>

namespace cudf {

// Strip cudf-specific parameters before passing the remaining arguments to nvbench::option_parser.
void benchmark_arg_handler(std::vector<std::string>& args)
{
  std::vector<std::string> _cudf_tmp_args;

  for (std::size_t i = 0; i < args.size(); ++i) {
    std::string arg = args[i];
    if (arg == cudf::detail::rmm_mode_param) {
      i++;  // skip the next argument
    } else if (arg == cudf::detail::cuio_host_mem_param) {
      i++;  // skip the next argument
    } else if (arg == cudf::detail::output_directory_param) {
      i++;  // skip the next argument
    } else {
      _cudf_tmp_args.push_back(arg);
    }
  }

  args = _cudf_tmp_args;
}

}  // namespace cudf

// Install arg handler
#undef NVBENCH_MAIN_CUSTOM_ARGS_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args) cudf::benchmark_arg_handler(args)

// Global fixture setup:
#undef NVBENCH_MAIN_INITIALIZE_CUSTOM_POST
#define NVBENCH_MAIN_INITIALIZE_CUSTOM_POST(argc, argv) \
  [[maybe_unused]] auto env_state = cudf::nvbench_base_fixture(argc, argv);

// this declares/defines the main() function using the definitions above
NVBENCH_MAIN
