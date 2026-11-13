/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "helpers.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <jit/cache.hpp>
#include <rtcx/rtcx.hpp>

#include <algorithm>
#include <format>
#include <stdexcept>

namespace cudf {
namespace jit {

size_type get_projection_size(column_view const& col) { return col.size(); }

/// @brief Scalar columns don't contribute to the row-size of a transform.
size_type get_projection_size(scalar_column_view const& col) { return 0; }

/// @brief Opaque pointers do not contribute to the row count.
size_type get_projection_size(user_data const&) { return 0; }

size_type get_projection_size(std::span<transform_input const> inputs)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);

  auto get_size = [](auto const& var) {
    return std::visit([](auto& a) { return get_projection_size(a); }, var);
  };

  return *std::max_element(cuda::transform_iterator(inputs.begin(), get_size),
                           cuda::transform_iterator(inputs.end(), get_size));
}

rtcx::blob get_udf_kernel_fragment(std::string const& source_file,
                                   std::string const& kernel_name,
                                   std::string const& udf_type,
                                   bool indexed)
{
  auto kernel_instance_source =
    std::format(R"***(#define CUDF_KERNEL_INSTANCE {})***", kernel_name);
  auto kernel_udf_source =
    std::format("#define CUDF_UDF_TYPE {}\n#define CUDF_UDF_INDEXED {}", udf_type, indexed ? 1 : 0);
  char const* include_names[] =  // NOLINT(modernize-avoid-c-arrays)
    {"cudf/detail/kernel_instance.cuh", "cudf/detail/operation_udf.cuh"};
  char const* include_headers[] =  // NOLINT(modernize-avoid-c-arrays)
    {kernel_instance_source.c_str(), kernel_udf_source.c_str()};

  return get_kernel_fragment(source_file, source_file, include_names, include_headers, kernel_name);
}

rtcx::binary_type as_rtcx_binary_type(lto_binary_type type)
{
  switch (type) {
    case lto_binary_type::LTO_IR: return rtcx::binary_type::LTO_IR;
    case lto_binary_type::FATBIN: return rtcx::binary_type::FATBIN;
    case lto_binary_type::PTX: return rtcx::binary_type::PTX;
    default:
      CUDF_FAIL(std::format("Unrecognized LTO binary type {}", static_cast<int>(type)),
                std::invalid_argument);
  }
}

}  // namespace jit
}  // namespace cudf
