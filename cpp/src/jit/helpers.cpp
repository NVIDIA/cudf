/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "helpers.hpp"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <jit/cache.hpp>
#include <rtcx.hpp>
#include <runtime/context.hpp>

#include <format>

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

  auto [smallest, largest] = std::minmax_element(
    inputs.begin(), inputs.end(), [](auto const& a, auto const& b) { return a.size() < b.size(); });

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

  return *std::max_element(thrust::make_transform_iterator(inputs.begin(), get_size),
                           thrust::make_transform_iterator(inputs.end(), get_size));
}

std::map<uint32_t, std::string> build_ptx_params(std::span<std::string const> output_typenames,
                                                 std::span<std::string const> input_typenames,
                                                 bool has_user_data)
{
  std::map<uint32_t, std::string> params;
  uint32_t index = 0;

  if (has_user_data) {
    params.emplace(index++, "void *");
    params.emplace(index++, "cudf::size_type");
  }

  for (auto& name : output_typenames) {
    params.emplace(index++, name + "*");
  }

  for (auto& name : input_typenames) {
    params.emplace(index++, name);
  }

  return params;
}

kernel get_udf_kernel(std::string const& source_file,
                      std::string const& kernel_name,
                      std::string const& cuda_source)
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

  return get_kernel(source_file, source_file, include_names, include_headers, kernel_name);
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

}  // namespace jit
}  // namespace cudf
