/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace cudf_streaming::benchmarks {

[[nodiscard]] inline bool is_communicator_available(std::string_view name)
{
#ifdef CUDF_STREAMING_HAVE_MPI
  if (name == "mpi") { return true; }
#endif
#ifdef CUDF_STREAMING_HAVE_UCXX
  if (name == "ucxx") { return true; }
#endif
  return false;
}

[[nodiscard]] inline std::string available_communicators()
{
  std::string result;
#ifdef CUDF_STREAMING_HAVE_MPI
  result += "mpi";
#endif
#ifdef CUDF_STREAMING_HAVE_UCXX
  if (!result.empty()) { result += ", "; }
  result += "ucxx";
#endif
  return result.empty() ? "none" : result;
}

[[nodiscard]] inline std::string default_communicator()
{
#ifdef CUDF_STREAMING_HAVE_MPI
  return "mpi";
#elif defined(CUDF_STREAMING_HAVE_UCXX)
  return "ucxx";
#else
  return {};
#endif
}

}  // namespace cudf_streaming::benchmarks
