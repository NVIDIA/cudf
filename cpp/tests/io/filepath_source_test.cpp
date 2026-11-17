/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

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

namespace {

class HostOnlySource : public cudf::io::datasource {
 public:
  explicit HostOnlySource(std::vector<std::byte> data) : _data{std::move(data)} {}

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const count = std::min(size, _data.size() - offset);
    return buffer::create(
      std::vector<std::byte>(_data.begin() + offset, _data.begin() + offset + count));
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const count = std::min(size, _data.size() - offset);
    std::memcpy(dst, _data.data() + offset, count);
    return count;
  }

  [[nodiscard]] size_t size() const override { return _data.size(); }

 private:
  std::vector<std::byte> _data;
};

class DirectDeviceReadSource : public HostOnlySource {
 public:
  using HostOnlySource::HostOnlySource;

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_direct(size_t) const override { return true; }
};

// Above KvikIO's default GDS threshold, so only the compatibility mode decides a file's answer
constexpr size_t large_read = 1 << 20;

}  // namespace

TEST_F(FilepathSourceTest, IsDeviceReadDirect)
{
  std::vector<std::byte> const data(64, std::byte{1});

  // libcudf defaults KvikIO to compatibility mode, which stages device reads through host memory.
  // libcudf reads the mode once per process, so an override can't be undone here; skip instead.
  auto const* compat_mode = std::getenv("KVIKIO_COMPAT_MODE");
  if (compat_mode == nullptr || std::string_view{compat_mode} == "ON") {
    auto const filepath = temp_env->get_temp_filepath("IsDeviceReadDirect.bin");
    std::ofstream(filepath, std::ios::binary)
      .write(reinterpret_cast<char const*>(data.data()), data.size());
    EXPECT_FALSE(cudf::io::datasource::create(filepath)->is_device_read_direct(large_read));
  }

  EXPECT_FALSE(cudf::io::datasource::create(cudf::host_span<std::byte const>{data})
                 ->is_device_read_direct(large_read));

  auto const stream = cudf::get_default_stream();
  cuda::device_buffer<std::byte> d_data(
    stream, cudf::get_current_device_resource_ref(), data.size(), cuda::no_init);
  EXPECT_TRUE(
    cudf::io::datasource::create(cudf::device_span<std::byte const>{d_data.data(), d_data.size()})
      ->is_device_read_direct(large_read));

  // A user source's answer reaches the caller through cudf's wrapper: false here...
  HostOnlySource host_only{data};
  EXPECT_FALSE(cudf::io::datasource::create(&host_only)->is_device_read_direct(large_read));

  // ...and true here, which the wrapper can only report by forwarding the call
  DirectDeviceReadSource direct{data};
  EXPECT_TRUE(cudf::io::datasource::create(&direct)->is_device_read_direct(large_read));
}

CUDF_TEST_PROGRAM_MAIN()
