/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "glushkov.cuh"
#include "glushkov_regcomp.hpp"
#include "regcomp.h"
#include "regex.cuh"

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace strings {

/**
 * @brief Implementation object for regex_program
 *
 * It encapsulates internal reprog object used for building its device equivalent.
 */
struct regex_program::regex_program_impl {
  detail::reprog prog;
  std::unique_ptr<detail::gkprog> glushkov_prog;

  regex_program_impl(detail::reprog const& p) : prog(p) {}
  regex_program_impl(detail::reprog&& p) : prog(std::move(p)) {}

  // TODO: There will be other options added here in the future to handle issues
  // 10852 and possibly others like 11979
};

struct regex_device_builder {
  static bool glushkov_fast_path_supported(regex_program const& p)
  {
    return p._impl->glushkov_prog.get() != nullptr;
  }

  static auto create_prog_device(regex_program const& p, rmm::cuda_stream_view stream)
  {
    return detail::reprog_device::create(p._impl->prog, stream);
  }

  static auto create_gkprog_device(regex_program const& p, rmm::cuda_stream_view stream)
  {
    CUDF_EXPECTS(glushkov_fast_path_supported(p), "fast-path not supported");
    return detail::gkprog_device::create(*p._impl->glushkov_prog, stream);
  }
};

}  // namespace strings
}  // namespace cudf
