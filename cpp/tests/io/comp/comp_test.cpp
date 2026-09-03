/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/compression.hpp"
#include "io/comp/decompression.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/detail/codec.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/iterator>

#include <src/io/comp/nvcomp_adapter.hpp>

#include <vector>

using cudf::device_span;
using cudf::io::detail::codec_exec_result;
using cudf::io::detail::codec_status;
namespace nvcomp = cudf::io::detail::nvcomp;

enum class hw { CPU, GPU };

/**
 * @brief Base test fixture for decompression
 *
 * Calls into Decompressor fixture to dispatch actual decompression work,
 * whose interface and setup is different for each codec.
 */
template <typename Decompressor>
struct DecompressTest
  : public cudf::test::BaseFixture,
    public testing::WithParamInterface<std::tuple<hw, cudf::io::compression_type>> {
  [[nodiscard]] std::vector<uint8_t> vector_from_string(std::string const str) const
  {
    return {reinterpret_cast<uint8_t const*>(str.c_str()),
            reinterpret_cast<uint8_t const*>(str.c_str()) + strlen(str.c_str())};
  }

  std::vector<uint8_t> Decompress(std::tuple<hw, cudf::io::compression_type> type,
                                  cudf::host_span<uint8_t const> compressed,
                                  size_t uncompressed_size)
  {
    auto hw_type   = std::get<0>(GetParam());
    auto comp_type = std::get<1>(GetParam());
    if (hw_type == hw::GPU) {
      if constexpr (has_gpu_impl<Decompressor>::value) {
        return DeviceDecompress(compressed, uncompressed_size);
      } else {
        CUDF_FAIL("Device decompression has not been implemented");
      }
    } else {
      if constexpr (has_cpu_impl<Decompressor>::value) {
        return HostDecompress(comp_type, compressed, uncompressed_size);
      } else {
        CUDF_FAIL("Host decompression has not been implemented");
      }
    }
  }

  std::vector<uint8_t> DeviceDecompress(cudf::host_span<uint8_t const> compressed,
                                        size_t uncompressed_size)
  {
    auto stream = cudf::get_default_stream();
    std::vector<uint8_t> decompressed(uncompressed_size);
    rmm::device_buffer src{compressed.data(), compressed.size(), stream};
    rmm::device_uvector<uint8_t> dst{decompressed.size(), stream};

    cudf::detail::hostdevice_vector<device_span<uint8_t const>> inf_in(1, stream);
    inf_in[0] = {static_cast<uint8_t const*>(src.data()), src.size()};
    inf_in.host_to_device_async(stream);

    cudf::detail::hostdevice_vector<device_span<uint8_t>> inf_out(1, stream);
    inf_out[0] = dst;
    inf_out.host_to_device_async(stream);

    cudf::detail::hostdevice_vector<codec_exec_result> inf_stat(1, stream);
    inf_stat[0] = {};
    inf_stat.host_to_device_async(stream);

    static_cast<Decompressor*>(this)->device_dispatch(inf_in, inf_out, inf_stat);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      decompressed.data(), dst.data(), dst.size(), cudaMemcpyDefault, stream.get()));
    inf_stat.device_to_host(stream);
    CUDF_EXPECTS(inf_stat[0].status == codec_status::SUCCESS, "Failure in device decompression");

    return decompressed;
  }

  std::vector<uint8_t> HostDecompress(cudf::io::compression_type comp_type,
                                      cudf::host_span<uint8_t const> compressed,
                                      size_t uncompressed_size)
  {
    return static_cast<Decompressor*>(this)->host_dispatch(
      comp_type, compressed, uncompressed_size);
  }

  template <typename T, typename = void>
  struct has_gpu_impl : std::false_type {};

  template <typename T>
  struct has_gpu_impl<T, std::void_t<decltype(&T::device_dispatch)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_cpu_impl : std::false_type {};

  template <typename T>
  struct has_cpu_impl<T, std::void_t<decltype(&T::host_dispatch)>> : std::true_type {};
};

struct HostCompressTest : public cudf::test::BaseFixture,
                          public ::testing::WithParamInterface<cudf::io::compression_type> {
  HostCompressTest()
  {
    setenv("LIBCUDF_HOST_COMPRESSION", "ON", 1);
    setenv("LIBCUDF_NVCOMP_POLICY", "ALWAYS", 1);
  }
  ~HostCompressTest() override
  {
    unsetenv("LIBCUDF_HOST_COMPRESSION");
    unsetenv("LIBCUDF_NVCOMP_POLICY");
  }
};

struct HostDecompressTest : public cudf::test::BaseFixture,
                            public ::testing::WithParamInterface<cudf::io::compression_type> {
  HostDecompressTest()
  {
    setenv("LIBCUDF_HOST_DECOMPRESSION", "ON", 1);
    setenv("LIBCUDF_NVCOMP_POLICY", "ALWAYS", 1);
  }
  ~HostDecompressTest() override
  {
    unsetenv("LIBCUDF_HOST_DECOMPRESSION");
    unsetenv("LIBCUDF_NVCOMP_POLICY");
  }
};

/**
 * @brief Derived fixture for GZIP decompression
 */
struct GzipDecompressTest : public DecompressTest<GzipDecompressTest> {
  void device_dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                       device_span<device_span<uint8_t>> d_inf_out,
                       device_span<codec_exec_result> d_inf_stat)
  {
    cudf::io::detail::gpuinflate(d_inf_in,
                                 d_inf_out,
                                 d_inf_stat,
                                 cudf::io::detail::gzip_header_included::YES,
                                 cudf::get_default_stream());
  }

  std::vector<uint8_t> host_dispatch(cudf::io::compression_type comp_type,
                                     cudf::host_span<uint8_t const> compressed,
                                     size_t uncompressed_size)
  {
    CUDF_EXPECTS(uncompressed_size <= cudf::io::detail::get_uncompressed_size(
                                        cudf::io::compression_type::AUTO, compressed),
                 "Underestimating uncompressed size!");
    CUDF_EXPECTS(comp_type == cudf::io::compression_type::AUTO ||
                   comp_type == cudf::io::compression_type::GZIP,
                 "Invalid compression type");
    return cudf::io::detail::decompress(comp_type, compressed);
  }
};

/**
 * @brief Derived fixture for GZIP decompression
 */
struct ZstdDecompressTest : public DecompressTest<ZstdDecompressTest> {
  std::vector<uint8_t> host_dispatch(cudf::io::compression_type comp_type,
                                     cudf::host_span<uint8_t const> compressed,
                                     size_t uncompressed_size)
  {
    CUDF_EXPECTS(uncompressed_size <= cudf::io::detail::get_uncompressed_size(
                                        cudf::io::compression_type::AUTO, compressed),
                 "Underestimating uncompressed size!");
    CUDF_EXPECTS(comp_type == cudf::io::compression_type::AUTO ||
                   comp_type == cudf::io::compression_type::ZSTD,
                 "Invalid compression type");
    return cudf::io::detail::decompress(comp_type, compressed);
  }
};

/**
 * @brief Derived fixture for Snappy decompression
 */
struct SnappyDecompressTest : public DecompressTest<SnappyDecompressTest> {
  void device_dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                       device_span<device_span<uint8_t>> d_inf_out,
                       device_span<codec_exec_result> d_inf_stat)
  {
    cudf::io::detail::gpu_unsnap(d_inf_in, d_inf_out, d_inf_stat, cudf::get_default_stream());
  }

  std::vector<uint8_t> host_dispatch(cudf::io::compression_type comp_type,
                                     cudf::host_span<uint8_t const> compressed,
                                     size_t uncompressed_size)
  {
    CUDF_EXPECTS(uncompressed_size <= cudf::io::detail::get_uncompressed_size(
                                        cudf::io::compression_type::AUTO, compressed),
                 "Underestimating uncompressed size!");
    CUDF_EXPECTS(comp_type == cudf::io::compression_type::AUTO ||
                   comp_type == cudf::io::compression_type::SNAPPY,
                 "Invalid compression type");
    return cudf::io::detail::decompress(comp_type, compressed);
  }
};

/**
 * @brief Derived fixture for Brotli decompression
 */
struct BrotliDecompressTest : public DecompressTest<BrotliDecompressTest> {
  void device_dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                       device_span<device_span<uint8_t>> d_inf_out,
                       device_span<codec_exec_result> d_inf_stat)
  {
    rmm::device_buffer d_scratch{cudf::io::detail::get_gpu_debrotli_scratch_size(1),
                                 cudf::get_default_stream()};

    cudf::io::detail::gpu_debrotli(d_inf_in, d_inf_out, d_inf_stat, cudf::get_default_stream());
  }
};

INSTANTIATE_TEST_CASE_P(GzipDecompressTest,
                        GzipDecompressTest,
                        ::testing::Combine(::testing::Values(hw::CPU, hw::GPU),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::GZIP)));

TEST_P(GzipDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 31> compressed{
    0x1f, 0x8b, 0x8,  0x0,  0x9,  0x63, 0x99, 0x5c, 0x2,  0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57,
    0x28, 0xcf, 0x2f, 0xca, 0x49, 0x1,  0x0,  0x85, 0x11, 0x4a, 0xd,  0xb,  0x0,  0x0,  0x0};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

INSTANTIATE_TEST_CASE_P(SnappyDecompressTest,
                        SnappyDecompressTest,
                        ::testing::Combine(::testing::Values(hw::CPU, hw::GPU),
                                           ::testing::Values(cudf::io::compression_type::AUTO,
                                                             cudf::io::compression_type::SNAPPY)));

TEST_P(SnappyDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 13> compressed = {
    0xb, 0x28, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

TEST_P(SnappyDecompressTest, ShortLiteralAfterLongCopyAtStartup)
{
  std::string const uncompressed{"Aaaaaaaaaaaah!"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 10> compressed = {
    14, 0x0, 'A', 0x0, 'a', (10 - 4) * 4 + 1, 1, 0x4, 'h', '!'};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

TEST_P(SnappyDecompressTest, MalformedAllOffsetZeroCopies)
{
  // uncompressed_size = 100 (varint = 0x64).
  // Then 25 copies of {0x01, 0x00}: each is "1-byte-offset copy, len=4, offset=0".
  std::vector<uint8_t> compressed{0x64};
  std::for_each(cuda::counting_iterator(0), cuda::counting_iterator(25), [&](auto) {
    compressed.push_back(0x01);
    compressed.push_back(0x00);
  });

  EXPECT_THROW(Decompress(GetParam(), {compressed.data(), compressed.size()}, 100),
               cudf::logic_error);
}

TEST_P(SnappyDecompressTest, MalformedOffsetZeroAfterLiteral)
{
  // uncompressed_size = 100
  // 4-byte literal: tag = (4-1) << 2 = 0x0C, then 4 data bytes.
  // 24 copies of {0x01, 0x00} = (len=4, offset=0). 24*4 = 96 bytes -> total 100.
  std::vector<uint8_t> compressed{0x64, 0x0c, 'A', 'B', 'C', 'D'};
  std::for_each(cuda::counting_iterator(0), cuda::counting_iterator(24), [&](auto) {
    compressed.push_back(0x01);
    compressed.push_back(0x00);
  });

  EXPECT_THROW(Decompress(GetParam(), {compressed.data(), compressed.size()}, 100),
               cudf::logic_error);
}

INSTANTIATE_TEST_CASE_P(
  BrotliDecompressTest,
  BrotliDecompressTest,
  ::testing::Values(std::make_tuple(hw::GPU, cudf::io::compression_type::AUTO),
                    std::make_tuple(hw::GPU, cudf::io::compression_type::BROTLI)));

TEST_P(BrotliDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 15> compressed = {
    0xb, 0x5, 0x80, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64, 0x3};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

INSTANTIATE_TEST_CASE_P(
  ZstdDecompressTest,
  ZstdDecompressTest,
  ::testing::Values(std::make_tuple(hw::CPU, cudf::io::compression_type::AUTO),
                    std::make_tuple(hw::CPU, cudf::io::compression_type::ZSTD)));

TEST_P(ZstdDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto compressed            = cudf::io::detail::compress(cudf::io::compression_type::ZSTD, input);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

struct NvcompConfigTest : public cudf::test::BaseFixture {};

TEST_F(NvcompConfigTest, Compression)
{
  using nvcomp::compression_type;
  auto const& comp_disabled = nvcomp::is_compression_disabled;

  EXPECT_FALSE(comp_disabled(compression_type::DEFLATE, {true, true}));
  EXPECT_FALSE(comp_disabled(compression_type::DEFLATE, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::DEFLATE, {false, false}));

  EXPECT_FALSE(comp_disabled(compression_type::ZSTD, {true, true}));
  EXPECT_FALSE(comp_disabled(compression_type::ZSTD, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::ZSTD, {false, false}));

  EXPECT_FALSE(comp_disabled(compression_type::SNAPPY, {true, true}));
  EXPECT_FALSE(comp_disabled(compression_type::SNAPPY, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::SNAPPY, {false, false}));

  // GZIP compression is only available with nvCOMP 5.3 and later
  if (cudf::io::detail::is_device_compression_supported(cudf::io::compression_type::GZIP)) {
    EXPECT_FALSE(comp_disabled(compression_type::GZIP, {true, true}));
    EXPECT_FALSE(comp_disabled(compression_type::GZIP, {false, true}));
    // stable integrations enabled required
    EXPECT_TRUE(comp_disabled(compression_type::GZIP, {false, false}));
  } else {
    EXPECT_TRUE(comp_disabled(compression_type::GZIP, {true, true}));
  }
}

TEST_F(NvcompConfigTest, Decompression)
{
  using nvcomp::compression_type;
  auto const& decomp_disabled = nvcomp::is_decompression_disabled;

  EXPECT_FALSE(decomp_disabled(compression_type::DEFLATE, {true, true}));
  EXPECT_FALSE(decomp_disabled(compression_type::DEFLATE, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::DEFLATE, {false, false}));

  EXPECT_FALSE(decomp_disabled(compression_type::ZSTD, {true, true}));
  EXPECT_FALSE(decomp_disabled(compression_type::ZSTD, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::ZSTD, {false, false}));

  EXPECT_FALSE(decomp_disabled(compression_type::SNAPPY, {true, true}));
  EXPECT_FALSE(decomp_disabled(compression_type::SNAPPY, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::SNAPPY, {false, false}));
}

void roundtrip_test(cudf::io::compression_type compression)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  std::vector<uint8_t> expected;
  expected.reserve(8 * (8 << 20));
  for (size_t size = 1; size < 8 << 20; size *= 2) {
    // Using number strings to generate data that is compressible, but not trivially so
    for (size_t i = size / 2; i < size; ++i) {
      auto const num_string = std::to_string(i);
      // Keep adding to the test data
      expected.insert(expected.end(), num_string.begin(), num_string.end());
    }
    if (cudf::io::detail::compress_max_allowed_chunk_size(compression)
          .value_or(std::numeric_limits<size_t>::max()) < expected.size()) {
      // Skip if the data is too large for the compressor
      return;
    }

    auto d_comp = rmm::device_uvector<uint8_t>(
      cudf::io::detail::max_compressed_size(compression, expected.size()), stream, mr);
    {
      auto const d_orig = cudf::detail::make_device_uvector_async(expected, stream, mr);
      auto hd_srcs      = cudf::detail::hostdevice_vector<device_span<uint8_t const>>(1, stream);
      hd_srcs[0]        = d_orig;
      hd_srcs.host_to_device_async(stream);

      auto hd_dsts = cudf::detail::hostdevice_vector<device_span<uint8_t>>(1, stream);
      hd_dsts[0]   = d_comp;
      hd_dsts.host_to_device_async(stream);

      auto hd_stats = cudf::detail::hostdevice_vector<codec_exec_result>(1, stream);
      hd_stats[0]   = codec_exec_result{0, codec_status::FAILURE};
      hd_stats.host_to_device_async(stream);

      cudf::io::detail::compress(compression, hd_srcs, hd_dsts, hd_stats, stream);
      hd_stats.device_to_host(stream);
      ASSERT_EQ(hd_stats[0].status, codec_status::SUCCESS);
      d_comp.resize(hd_stats[0].bytes_written, stream);
    }

    auto d_got = rmm::device_uvector<uint8_t>(expected.size(), stream);
    {
      auto hd_srcs = cudf::detail::hostdevice_vector<device_span<uint8_t const>>(1, stream);
      hd_srcs[0]   = d_comp;
      hd_srcs.host_to_device_async(stream);

      auto hd_dsts = cudf::detail::hostdevice_vector<device_span<uint8_t>>(1, stream);
      hd_dsts[0]   = d_got;
      hd_dsts.host_to_device_async(stream);

      auto hd_stats = cudf::detail::hostdevice_vector<codec_exec_result>(1, stream);
      hd_stats[0]   = codec_exec_result{0, codec_status::FAILURE};
      hd_stats.host_to_device_async(stream);

      cudf::io::detail::decompress(
        compression, hd_srcs, hd_dsts, hd_stats, expected.size(), expected.size(), stream);
      hd_stats.device_to_host(stream);
      ASSERT_EQ(hd_stats[0].status, codec_status::SUCCESS);
    }

    auto const got = cudf::detail::make_std_vector(d_got, stream);

    EXPECT_EQ(expected, got);
  }
}

TEST_P(HostCompressTest, HostCompression) { roundtrip_test(GetParam()); }

INSTANTIATE_TEST_CASE_P(HostCompression,
                        HostCompressTest,
                        ::testing::Values(cudf::io::compression_type::GZIP,
                                          cudf::io::compression_type::SNAPPY,
                                          cudf::io::compression_type::ZSTD));

TEST_P(HostDecompressTest, HostDecompression) { roundtrip_test(GetParam()); }

INSTANTIATE_TEST_CASE_P(HostDecompression,
                        HostDecompressTest,
                        ::testing::Values(cudf::io::compression_type::GZIP,
                                          cudf::io::compression_type::SNAPPY,
                                          cudf::io::compression_type::ZLIB,
                                          cudf::io::compression_type::ZSTD));

CUDF_TEST_PROGRAM_MAIN()
