/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Keep the example in sync with kernel_authoring.rst.

#include "jit/helpers.hpp"
#include "jit/reflect.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_device_view_base.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/errc.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/stream>

#include <cudf_test_fragments.hpp>

#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

namespace {

std::string reflect_ternary_op_kernel(cudf::data_type t0,
                                      cudf::data_type t1,
                                      cudf::data_type t2,
                                      cudf::data_type output_type)
{
  return std::format("cudf::detail::ternary_op_kernel<{}, {}, {}, {}>",
                     cudf::type_to_name(t0),
                     cudf::type_to_name(t1),
                     cudf::type_to_name(t2),
                     cudf::type_to_name(output_type));
}

cudf::kernel get_kernel(cudf::transform_udf const& udf,
                        cudf::data_type t0,
                        cudf::data_type t1,
                        cudf::data_type t2,
                        cudf::data_type output_type)
{
  CUDF_EXPECTS(udf.is_null_aware == cudf::null_aware::NO,
               "This example kernel requires a non-null-aware UDF");

  std::array inputs{cudf::transform_input_spec{.type = t0.id()},
                    cudf::transform_input_spec{.type = t1.id()},
                    cudf::transform_input_spec{.type = t2.id()}};
  std::array outputs{cudf::transform_output_spec{.type = output_type.id()}};
  auto const index_count = udf.indexed ? 1 : 0;
  auto const instance    = reflect_ternary_op_kernel(t0, t1, t2, output_type);
  auto const signature =
    cudf::jit::reflect_udf_signature(false, inputs, outputs, false, index_count);
  auto kernel_lto_ir = cudf::jit::get_udf_kernel_fragment(
    "cudf/cpp/src/sample_ternary_op_kernel/jit/ternary_op_kernel.cu",
    instance,
    signature,
    udf.indexed);

  std::vector<rtcx::memory_fragment> fragments{
    {.data = kernel_lto_ir->view(), .type = rtcx::binary_type::LTO_IR}};
  auto adapter = cudf::get_udf_adapter(
    udf.symbol, "cudf_transform_udf", false, inputs, outputs, false, index_count);
  if (adapter) {
    fragments.push_back({.data = adapter->view().data, .type = rtcx::binary_type::LTO_IR});
  }
  for (auto const& fragment : udf.fragments) {
    fragments.push_back(
      {.data = fragment.data, .type = cudf::jit::as_rtcx_binary_type(fragment.type)});
  }

  return cudf::get_lto_linked_kernel(
    "cudf/cpp/src/sample_ternary_op_kernel/jit/ternary_op_kernel.cu", {}, fragments);
}

cudf::transform_udf asymmetric_squared_loss_udf()
{
  static std::array const fragments{cudf::lto_fragment{
    .type = cudf::lto_binary_type::FATBIN,
    .data = cudf_test_fragments::get(cudf_test_fragments::asymmetric_squared_loss_udf)}};
  return {.symbol = "asymmetric_squared_loss", .fragments = fragments};
}

void launch_ternary_op(cudf::transform_udf const& udf,
                       cudf::column_device_view_core const* device_inputs,
                       cudf::mutable_column_device_view_core const* device_output,
                       cudf::size_type n,
                       cudf::error_int* device_error,
                       cudf::data_type t0,
                       cudf::data_type t1,
                       cudf::data_type t2,
                       cudf::data_type output_type,
                       cuda::stream_ref stream)
{
  if (n == 0) { return; }

  auto kernel = get_kernel(udf, t0, t1, t2, output_type);

  // Launch the kernel with device-side column views.
  cudf::detail::grid_1d const grid{n, 256};
  kernel.launch_with({static_cast<uint32_t>(grid.num_blocks)},
                     {static_cast<uint32_t>(grid.num_threads_per_block)},
                     0,
                     stream,
                     device_inputs,
                     device_output,
                     n,
                     device_error);
}

void asymmetric_squared_loss(cudf::column_device_view_core const* device_inputs,
                             cudf::mutable_column_device_view_core const* device_output,
                             cudf::size_type n,
                             cudf::error_int* device_error,
                             cuda::stream_ref stream)
{
  launch_ternary_op(asymmetric_squared_loss_udf(),
                    device_inputs,
                    device_output,
                    n,
                    device_error,
                    cudf::data_type{cudf::type_id::FLOAT64},
                    cudf::data_type{cudf::type_id::FLOAT64},
                    cudf::data_type{cudf::type_id::FLOAT64},
                    cudf::data_type{cudf::type_id::FLOAT64},
                    stream);
}

struct KernelAuthoringTest : public cudf::test::BaseFixture {};

TEST_F(KernelAuthoringTest, AsymmetricSquaredLoss)
{
  auto const stream = cudf::test::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  cudf::test::fixed_width_column_wrapper<double> actual{{3, 1, 2, 0, -1, 5}};
  cudf::test::fixed_width_column_wrapper<double> target{{1, 3, 2, 1, 1, 2}};
  cudf::test::fixed_width_column_wrapper<double> penalty{{2.0, 2.0, 3.0, 0.5, 1.5, 4.0}};

  auto actual_view  = cudf::column_device_view::create(actual, stream);
  auto target_view  = cudf::column_device_view::create(target, stream);
  auto penalty_view = cudf::column_device_view::create(penalty, stream);
  auto const n      = actual_view->size();
  auto output       = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::FLOAT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto output_view = cudf::mutable_column_device_view::create(output->mutable_view(), stream);

  std::array<cudf::column_device_view_core, 3> h_inputs{*actual_view, *target_view, *penalty_view};
  std::array<cudf::mutable_column_device_view_core, 1> h_outputs{*output_view};
  auto d_inputs  = cudf::detail::make_device_uvector_async(h_inputs, stream, mr);
  auto d_outputs = cudf::detail::make_device_uvector_async(h_outputs, stream, mr);
  cudf::detail::device_scalar<cudf::error_int> error{0, stream, mr};

  asymmetric_squared_loss(d_inputs.data(), d_outputs.data(), n, error.data(), stream);

  EXPECT_EQ(error.value(stream), static_cast<cudf::error_int>(cudf::errc::SUCCESS));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    output->view(), cudf::test::fixed_width_column_wrapper<double>({4.0, 8.0, 0.0, 0.5, 6.0, 9.0}));
}

}  // namespace
