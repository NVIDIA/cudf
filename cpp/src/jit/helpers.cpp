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

bool is_scalar(cudf::size_type base_column_size, cudf::size_type column_size)
{
  return column_size == 1 && column_size != base_column_size;
}

typename std::vector<column_view>::const_iterator get_transform_base_column(
  std::vector<column_view> const& inputs)
{
  if (inputs.empty()) { return inputs.end(); }

  auto [smallest, largest] = std::ranges::minmax_element(
    inputs, [](auto const& a, auto const& b) { return a.size() < b.size(); });

  /// when the largest size is 1, the size-1 column could be a scalar or an actual column, it would
  /// be a scalar if it has columns that are zero-sized
  if (largest->size() != 1) { return largest; }

  if (smallest->size() == 0) { return smallest; }

  return largest;
}

size_type get_projection_size(column_view const& col) { return col.size(); }

/// @brief Scalar columns don't contribute to the row-size of a transform.
size_type get_projection_size(scalar_column_view const& col) { return 0; }

size_type get_projection_size(std::span<std::variant<column_view, scalar_column_view> const> inputs)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);

  auto get_size = [](auto const& var) {
    return std::visit([](auto& a) { return get_projection_size(a); }, var);
  };

  return *std::max_element(cuda::transform_iterator(inputs.begin(), get_size),
                           cuda::transform_iterator(inputs.end(), get_size));
}

kernel get_udf_kernel(std::string const& source_file,
                      std::string const& kernel_name,
                      std::string const& cuda_source,
                      cudf::lto_fragments link_fragments)
{
  CUDF_FUNC_RANGE();

  auto kernel_instance_source = std::format(R"***(
 #define CUDF_KERNEL_INSTANCE {}
 )***",
                                            kernel_name);
  char const* include_names[] =  // NOLINT(modernize-avoid-c-arrays)
    {"cudf/detail/operation_udf.cuh", "cudf/detail/kernel_instance.cuh"};
  char const* include_headers[] =  // NOLINT(modernize-avoid-c-arrays)
    {cuda_source.c_str(), kernel_instance_source.c_str()};

  if (link_fragments.empty()) {
    return get_kernel(source_file, source_file, include_names, include_headers, kernel_name);
  }
  auto kernel_fragment =
    get_kernel_fragment(source_file, source_file, include_names, include_headers, kernel_name);
  std::vector<rtcx::memory_fragment> fragments{
    {.data = kernel_fragment->view(), .type = rtcx::binary_type::LTO_IR, .name = nullptr}};
  for (auto const& fragment : link_fragments) {
    fragments.push_back(
      {.data = fragment.data, .type = as_rtcx_binary_type(fragment.type), .name = nullptr});
  }
  return get_lto_linked_kernel(source_file, {}, fragments);
}

std::string define_operation(std::string const& cuda_source,
                             std::string_view operation_macro,
                             std::string_view function_name)
{
  // The call stays unqualified, as the call of a renamed function was: a qualified call is checked
  // when the kernel template is defined, even in a branch the kernel never takes.
  return std::format(
    "{}\n#define {}(...) {}(__VA_ARGS__)\n", cuda_source, operation_macro, function_name);
}

rtcx::blob get_udf_kernel_fragment(std::string const& source_file,
                                   std::string const& kernel_name,
                                   std::string const& udf_type)
{
  auto kernel_instance_source = std::format(R"***(#define CUDF_KERNEL_INSTANCE {}
 #define CUDF_LTO_MODE)***",
                                            kernel_name);
  auto kernel_udf_source      = std::format(R"***(#define CUDF_UDF_TYPE {})***", udf_type);
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
