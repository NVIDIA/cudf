/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <cstddef>
#include <vector>

class CudfTableTest : public cudf::test::BaseFixture {};

TEST_F(CudfTableTest, Writer)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  cudf::table_view tab({col});

  std::vector<char> buffer;
  cudf::io::experimental::write_cudftable(
    cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&buffer}, tab)
      .compression(cudf::io::compression_type::SNAPPY)
      .build(),
    cudf::test::get_default_stream());
}

TEST_F(CudfTableTest, Reader)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  cudf::table_view tab({col});

  std::vector<char> buffer;
  cudf::io::experimental::write_cudftable(
    cudf::io::experimental::cudftable_writer_options::builder(cudf::io::sink_info{&buffer}, tab)
      .compression(cudf::io::compression_type::SNAPPY)
      .build(),
    cudf::test::get_default_stream());

  auto host_buffer = cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
  cudf::io::experimental::read_cudftable(
    cudf::io::experimental::cudftable_reader_options::builder(cudf::io::source_info{host_buffer})
      .build(),
    cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
