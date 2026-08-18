/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../compression_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/detail/bgzip_utils.hpp>

#include <fstream>
#include <random>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

struct DataChunkSourceTest : public cudf::test::BaseFixture {};

std::string chunk_to_host(cudf::io::text::device_data_chunk const& chunk)
{
  std::string result(chunk.size(), '\0');
  CUDF_CUDA_TRY(cudaMemcpy(result.data(), chunk.data(), chunk.size(), cudaMemcpyDefault));
  return result;
}

void test_source(std::string const& content, cudf::io::text::data_chunk_source const& source)
{
  static cudf::test::pinned_pool mr{size_t{128} * 1024 * 1024};
  auto last_mr = cudf::set_pinned_memory_resource(mr);

  auto stream = cudf::get_default_stream();
  {
    // full contents
    auto reader      = source.create_reader();
    auto const chunk = reader->get_next_chunk(content.size(), stream);
    stream.synchronize();
    EXPECT_EQ(chunk->size(), content.size());
    EXPECT_EQ(chunk_to_host(*chunk), content);
  }
  {
    // skipping contents
    auto reader = source.create_reader();
    reader->skip_bytes(4);
    auto const chunk = reader->get_next_chunk(content.size(), stream);
    stream.synchronize();
    EXPECT_EQ(chunk->size(), content.size() - 4);
    EXPECT_EQ(chunk_to_host(*chunk), content.substr(4));
  }
  {
    // reading multiple chunks, starting with a small one
    auto reader       = source.create_reader();
    auto const chunk1 = reader->get_next_chunk(5, stream);
    auto const chunk2 = reader->get_next_chunk(content.size() - 5, stream);
    stream.synchronize();
    EXPECT_EQ(chunk1->size(), 5);
    EXPECT_EQ(chunk2->size(), content.size() - 5);
    EXPECT_EQ(chunk_to_host(*chunk1), content.substr(0, 5));
    EXPECT_EQ(chunk_to_host(*chunk2), content.substr(5));
  }
  {
    // reading multiple chunks
    auto reader       = source.create_reader();
    auto const chunk1 = reader->get_next_chunk(content.size() / 2, stream);
    auto const chunk2 = reader->get_next_chunk(content.size() - content.size() / 2, stream);
    stream.synchronize();
    EXPECT_EQ(chunk1->size(), content.size() / 2);
    EXPECT_EQ(chunk2->size(), content.size() - content.size() / 2);
    EXPECT_EQ(chunk_to_host(*chunk1), content.substr(0, content.size() / 2));
    EXPECT_EQ(chunk_to_host(*chunk2), content.substr(content.size() / 2));
  }
  {
    // reading too many bytes
    auto reader      = source.create_reader();
    auto const chunk = reader->get_next_chunk(content.size() + 10, stream);
    stream.synchronize();
    EXPECT_EQ(chunk->size(), content.size());
    EXPECT_EQ(chunk_to_host(*chunk), content);
    auto next_chunk = reader->get_next_chunk(1, stream);
    stream.synchronize();
    EXPECT_EQ(next_chunk->size(), 0);
  }
  {
    // skipping past the end
    auto reader = source.create_reader();
    reader->skip_bytes(content.size() + 10);
    auto const next_chunk = reader->get_next_chunk(1, stream);
    stream.synchronize();
    EXPECT_EQ(next_chunk->size(), 0);
  }
  cudf::set_pinned_memory_resource(last_mr);
}

TEST_F(DataChunkSourceTest, DataSourceHost)
{
  std::string const content = "host buffer source";
  auto const span           = cudf::host_span<std::byte const>(
    reinterpret_cast<std::byte const*>(content.data()), content.size());
  auto const datasource = cudf::io::datasource::create(span);
  auto const source     = cudf::io::text::make_source(*datasource);

  test_source(content, *source);
}

TEST_F(DataChunkSourceTest, DataSourceFile)
{
  std::string content = "file datasource";
  // make it big enough to have is_device_read_preferred return true
  content.reserve(content.size() << 20);
  for (int i = 0; i < 20; i++) {
    content += content;
  }
  auto const filename = temp_env->get_temp_filepath("file_source");
  {
    std::ofstream file{filename};
    file << content;
  }
  auto const datasource = cudf::io::datasource::create(filename);
  auto const source     = cudf::io::text::make_source(*datasource);

  test_source(content, *source);
}

TEST_F(DataChunkSourceTest, Device)
{
  std::string const content = "device buffer source";
  cudf::string_scalar scalar(content);
  auto const source = cudf::io::text::make_source(scalar);

  test_source(content, *source);
}

TEST_F(DataChunkSourceTest, File)
{
  std::string const content = "file source";
  auto const filename       = temp_env->get_temp_filepath("file_source");
  {
    std::ofstream file{filename};
    file << content;
  }
  auto const source = cudf::io::text::make_source_from_file(filename);

  test_source(content, *source);
}

TEST_F(DataChunkSourceTest, Host)
{
  std::string const content = "host buffer source";
  auto const source         = cudf::io::text::make_source(content);

  test_source(content, *source);
}

enum class compression { ENABLED, DISABLED };

enum class eof { ADD_EOF_BLOCK, NO_EOF_BLOCK };

uint64_t virtual_offset(std::size_t block_offset, std::size_t local_offset)
{
  return (block_offset << 16) | local_offset;
}

void write_bgzip(std::ostream& output_stream,
                 cudf::host_span<char const> data,
                 std::default_random_engine& rng,
                 compression compress,
                 eof add_eof)
{
  std::vector<char> const extra_garbage_fields1{{13,  // magic number
                                                 37,  // magic number
                                                 7,   // field length
                                                 0,   // field length
                                                 1,
                                                 2,
                                                 3,
                                                 4,
                                                 5,
                                                 6,
                                                 7}};
  std::vector<char> const extra_garbage_fields2{{12,  // magic number
                                                 34,  // magic number
                                                 2,   // field length
                                                 0,   // field length
                                                 1,  2,
                                                 56,  // magic number
                                                 78,  // magic number
                                                 1,   // field length
                                                 0,   // field length
                                                 3,   //
                                                 90,  // magic number
                                                 12,  // magic number
                                                 8,   // field length
                                                 0,   // field length
                                                 1,  2, 3, 4, 5, 6, 7, 8}};
  // make sure the block size with header stays below 65536
  std::uniform_int_distribution<std::size_t> block_size_dist{1, 65000};
  auto begin     = data.begin();
  auto const end = data.end();
  int i          = 0;
  while (begin < end) {
    using cudf::host_span;
    auto len = std::min<std::size_t>(end - begin, block_size_dist(rng));
    host_span<char const> const garbage_before =
      i & 1 ? extra_garbage_fields1 : host_span<char const>{};
    host_span<char const> const garbage_after =
      i & 2 ? extra_garbage_fields2 : host_span<char const>{};
    if (compress == compression::ENABLED) {
      cudf::io::text::detail::bgzip::write_compressed_block(
        output_stream, {begin, len}, garbage_before, garbage_after);
    } else {
      cudf::io::text::detail::bgzip::write_uncompressed_block(
        output_stream, {begin, len}, garbage_before, garbage_after);
    }
    begin += len;
    i++;
  }
  if (add_eof == eof::ADD_EOF_BLOCK) {
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream, {});
  }
}
using DataChunkDecompressionTest = DecompressionTest<DataChunkSourceTest>;

TEST_P(DataChunkDecompressionTest, BgzipSource)
{
  auto const filename = temp_env->get_temp_filepath("bgzip_source");
  std::string input{"bananarama"};
  input.reserve(input.size() << 25);
  for (int i = 0; i < 24; i++) {
    input = input + input;
  }
  {
    std::ofstream output_stream{filename};
    std::default_random_engine rng{};
    write_bgzip(output_stream, input, rng, compression::DISABLED, eof::ADD_EOF_BLOCK);
  }

  auto const source = cudf::io::text::make_source_from_bgzip_file(filename);

  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipSourceVirtualOffsets)
{
  auto const filename = temp_env->get_temp_filepath("bgzip_source_offsets");
  std::string input{"bananarama"};
  input.reserve(input.size() << 25);
  for (int i = 0; i < 24; i++) {
    input = input + input;
  }
  std::string const padding_garbage(10000, 'g');
  std::string const data_garbage{"GARBAGE"};
  std::string const begininput{"begin of bananarama"};
  std::string const endinput{"end of bananarama"};
  std::size_t begin_compressed_offset{};
  std::size_t end_compressed_offset{};
  std::size_t const begin_local_offset{data_garbage.size()};
  std::size_t const end_local_offset{endinput.size()};
  {
    std::ofstream output_stream{filename};
    output_stream.write(padding_garbage.data(), padding_garbage.size());
    std::default_random_engine rng{};
    begin_compressed_offset = output_stream.tellp();
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream,
                                                            data_garbage + begininput);
    write_bgzip(output_stream, input, rng, compression::DISABLED, eof::NO_EOF_BLOCK);
    end_compressed_offset = output_stream.tellp();
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream,
                                                            endinput + data_garbage + data_garbage);
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream, {});
    output_stream.write(padding_garbage.data(), padding_garbage.size());
  }
  input = begininput + input + endinput;

  auto const source = cudf::io::text::make_source_from_bgzip_file(
    filename,
    virtual_offset(begin_compressed_offset, begin_local_offset),
    virtual_offset(end_compressed_offset, end_local_offset));

  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipSourceVirtualOffsetsSingleGZipBlock)
{
  auto const filename = temp_env->get_temp_filepath("bgzip_source_offsets_single_block");
  std::string const input{"collection unit brings"};
  std::string const head_garbage{"garbage"};
  std::string const tail_garbage{"GARBAGE"};
  std::size_t const begin_local_offset{head_garbage.size()};
  std::size_t const end_local_offset{head_garbage.size() + input.size()};
  {
    std::ofstream output_stream{filename};
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream,
                                                            head_garbage + input + tail_garbage);
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream, {});
  }

  auto const source = cudf::io::text::make_source_from_bgzip_file(
    filename, virtual_offset(0, begin_local_offset), virtual_offset(0, end_local_offset));

  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipSourceVirtualOffsetsSingleChunk)
{
  auto const filename = temp_env->get_temp_filepath("bgzip_source_offsets_single_chunk");
  std::string const input{"collection unit brings"};
  std::string const head_garbage{"garbage"};
  std::string const tail_garbage{"GARBAGE"};
  std::size_t end_compressed_offset{};
  std::size_t const begin_local_offset{head_garbage.size()};
  std::size_t const end_local_offset{input.size() - 10};
  {
    std::ofstream output_stream{filename};
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream,
                                                            head_garbage + input.substr(0, 10));
    end_compressed_offset = output_stream.tellp();
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream,
                                                            input.substr(10) + tail_garbage);
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream, {});
  }

  auto const source = cudf::io::text::make_source_from_bgzip_file(
    filename,
    virtual_offset(0, begin_local_offset),
    virtual_offset(end_compressed_offset, end_local_offset));

  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipCompressedSourceVirtualOffsets)
{
  auto const filename = temp_env->get_temp_filepath("bgzip_source_compressed_offsets");
  std::string input{"bananarama"};
  input.reserve(input.size() << 25);
  for (int i = 0; i < 24; i++) {
    input = input + input;
  }
  std::string const padding_garbage(10000, 'g');
  std::string const data_garbage{"GARBAGE"};
  std::string const begininput{"begin of bananarama"};
  std::string const endinput{"end of bananarama"};
  std::size_t begin_compressed_offset{};
  std::size_t end_compressed_offset{};
  std::size_t const begin_local_offset{data_garbage.size()};
  std::size_t const end_local_offset{endinput.size()};
  {
    std::ofstream output_stream{filename};
    output_stream.write(padding_garbage.data(), padding_garbage.size());
    std::default_random_engine rng{};
    begin_compressed_offset = output_stream.tellp();
    cudf::io::text::detail::bgzip::write_compressed_block(output_stream, data_garbage + begininput);
    write_bgzip(output_stream, input, rng, compression::ENABLED, eof::NO_EOF_BLOCK);
    end_compressed_offset = output_stream.tellp();
    cudf::io::text::detail::bgzip::write_compressed_block(output_stream,
                                                          endinput + data_garbage + data_garbage);
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream, {});
    output_stream.write(padding_garbage.data(), padding_garbage.size());
  }
  input = begininput + input + endinput;

  auto source = cudf::io::text::make_source_from_bgzip_file(
    filename,
    virtual_offset(begin_compressed_offset, begin_local_offset),
    virtual_offset(end_compressed_offset, end_local_offset));
  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipSourceVirtualOffsetsSingleCompressedGZipBlock)
{
  auto const filename = temp_env->get_temp_filepath("bgzip_source_offsets_single_compressed_block");
  std::string const input{"collection unit brings"};
  std::string const head_garbage(10000, 'g');
  std::string const tail_garbage{"GARBAGE"};
  std::size_t const begin_local_offset{head_garbage.size()};
  std::size_t const end_local_offset{head_garbage.size() + input.size()};
  {
    std::ofstream output_stream{filename};
    cudf::io::text::detail::bgzip::write_compressed_block(output_stream,
                                                          head_garbage + input + tail_garbage);
    cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream, {});
  }

  auto const source = cudf::io::text::make_source_from_bgzip_file(
    filename, virtual_offset(0, begin_local_offset), virtual_offset(0, end_local_offset));

  test_source(input, *source);
}

INSTANTIATE_TEST_CASE_P(Nvcomp,
                        DataChunkDecompressionTest,
                        ::testing::Combine(::testing::Values("NVCOMP", "DEVICE_INTERNAL", "HOST"),
                                           ::testing::Values(cudf::io::compression_type::ZLIB)));

CUDF_TEST_PROGRAM_MAIN()
