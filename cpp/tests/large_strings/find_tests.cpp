/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <limits>
#include <string>

struct FindTest : public cudf::test::StringsLargeTest {};

TEST_F(FindTest, ContainsLongestRow)
{
  // A single row of the maximum size with the only match near its end
  auto constexpr max_size = std::numeric_limits<cudf::size_type>::max();
  auto const target       = std::string("bcd");
  auto const stream       = cudf::get_default_stream();
  auto const expected     = cudf::test::fixed_width_column_wrapper<bool>({true});

  // match at the very end and match 5000 bytes from the end
  for (cudf::size_type suffix : {0, 4997}) {
    auto chars = rmm::device_buffer(max_size, stream);
    CUDF_CUDA_TRY(cudaMemsetAsync(chars.data(), 'a', max_size, stream.get()));
    auto const pos = max_size - static_cast<cudf::size_type>(target.size()) - suffix;
    CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<char*>(chars.data()) + pos,
                                  target.data(),
                                  target.size(),
                                  cudaMemcpyDefault,
                                  stream.get()));
    auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, max_size}).release();
    auto const input =
      cudf::make_strings_column(1,
                                std::move(offsets),
                                std::move(chars),
                                0,
                                cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

    auto const results = cudf::strings::contains(cudf::strings_column_view(input->view()),
                                                 cudf::string_scalar(target));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}
