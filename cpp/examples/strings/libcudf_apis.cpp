/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.hpp"

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

/**
 * @brief Redacts each name per the corresponding visibility entry
 *
 * This implementation uses libcudf APIs to create the output result.
 *
 * @param names Column of names
 * @param visibilities Column of visibilities
 * @return Redacted column of names
 */
std::unique_ptr<cudf::column> redact_strings(cudf::column_view const& names,
                                             cudf::column_view const& visibilities)
{
  auto const visible   = cudf::string_scalar(std::string("public"));
  auto const redaction = cudf::string_scalar(std::string("X X"));

  nvtxRangePushA("redact_strings");

  auto const allowed      = cudf::strings::contains(visibilities, visible);
  auto const redacted     = cudf::copy_if_else(names, redaction, allowed->view());
  auto const first_last   = cudf::strings::split(redacted->view());
  auto const first        = first_last->view().column(0);
  auto const last         = first_last->view().column(1);
  auto const last_initial = cudf::strings::slice_strings(
    last, std::optional<cudf::size_type>(0), std::optional<cudf::size_type>(1));

  auto const last_initial_first = cudf::table_view({last_initial->view(), first});

  auto result = cudf::strings::concatenate(last_initial_first, std::string_view(" "));

  cudaStreamSynchronize(0);

  nvtxRangePop();
  return result;
}
