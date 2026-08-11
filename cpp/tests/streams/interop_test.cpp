/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>

#include <dlpack/dlpack.h>

namespace {
struct dlpack_deleter {
  void operator()(DLManagedTensor* tensor) { tensor->deleter(tensor); }
};
}  // namespace

struct DLPackTest : public cudf::test::BaseFixture {};

TEST_F(DLPackTest, ToDLPack)
{
  cudf::table_view empty(std::vector<cudf::column_view>{});
  cudf::to_dlpack(empty, cudf::test::get_default_stream());
}

TEST_F(DLPackTest, FromDLPack)
{
  using unique_managed_tensor = std::unique_ptr<DLManagedTensor, dlpack_deleter>;
  cudf::test::fixed_width_column_wrapper<int32_t> col1({});
  cudf::test::fixed_width_column_wrapper<int32_t> col2({});
  cudf::table_view input({col1, col2});
  unique_managed_tensor tensor(cudf::to_dlpack(input, cudf::test::get_default_stream()));
  auto result = cudf::from_dlpack(tensor.get(), cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
