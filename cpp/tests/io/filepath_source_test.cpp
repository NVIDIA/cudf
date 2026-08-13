/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <filesystem>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

struct FilepathSourceTest : public cudf::test::BaseFixture {};

TEST_F(FilepathSourceTest, StringConstructorsPopulateFilepathSources)
{
  auto const single = cudf::io::source_info{"test.parquet"};
  ASSERT_EQ(single.filepath_sources().size(), 1);
  EXPECT_EQ(single.filepath_sources().front().path, "test.parquet");
  EXPECT_FALSE(single.filepath_sources().front().size.has_value());
  EXPECT_EQ(single.filepaths(), std::vector<std::string>{"test.parquet"});

  auto const multi = cudf::io::source_info{std::vector<std::string>{"a.parquet", "b.parquet"}};
  ASSERT_EQ(multi.filepath_sources().size(), 2);
  EXPECT_EQ(multi.filepaths().size(), 2);
  EXPECT_FALSE(multi.filepath_sources()[1].size.has_value());
}

TEST_F(FilepathSourceTest, FilepathSourceConstructorPreservesSize)
{
  std::vector<cudf::io::filepath_source> sources{
    {"s3://bucket/object.parquet", 12345},
    {"https://example.com/data.parquet", std::nullopt},
  };

  auto const info = cudf::io::source_info{std::move(sources)};
  ASSERT_EQ(info.filepath_sources().size(), 2);
  EXPECT_EQ(info.filepath_sources()[0].path, "s3://bucket/object.parquet");
  ASSERT_TRUE(info.filepath_sources()[0].size.has_value());
  EXPECT_EQ(info.filepath_sources()[0].size.value(), 12345);
  EXPECT_FALSE(info.filepath_sources()[1].size.has_value());
  EXPECT_EQ(info.filepaths()[0], "s3://bucket/object.parquet");
  EXPECT_EQ(info.filepaths()[1], "https://example.com/data.parquet");
}

TEST_F(FilepathSourceTest, KnownSizePlumbsThroughMakeDatasources)
{
  auto const filepath = temp_env->get_temp_filepath("KnownSize.parquet");

  auto col = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3};
  cudf::table_view const table{{col}};

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table);
  cudf::io::write_parquet(write_opts);

  auto const file_size = std::filesystem::file_size(filepath);
  std::vector<cudf::io::filepath_source> sources{{filepath, file_size}};
  auto const source_info = cudf::io::source_info{std::move(sources)};

  auto datasources = cudf::io::make_datasources(source_info);
  ASSERT_EQ(datasources.size(), 1);
  EXPECT_EQ(datasources.front()->size(), file_size);

  auto const read_opts = cudf::io::parquet_reader_options::builder(source_info).build();
  auto const result    = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(table, result.tbl->view());
}
