/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/parquet/experimental/variant_json_string.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace pqed = cudf::io::parquet::experimental::detail;

/// Packs strings into one device buffer with offsets
struct device_inputs {
  rmm::device_uvector<char> chars;
  rmm::device_uvector<int> offsets;

  explicit device_inputs(std::vector<std::string> const& inputs)
    : chars(0, cudf::get_default_stream()), offsets(0, cudf::get_default_stream())
  {
    std::string all;
    std::vector<int> offs{0};
    for (auto const& s : inputs) {
      all += s;
      offs.push_back(static_cast<int>(all.size()));
    }
    chars = cudf::detail::make_device_uvector(cudf::host_span<char const>{all.data(), all.size()},
                                              cudf::get_default_stream(),
                                              cudf::get_current_device_resource_ref());
    offsets =
      cudf::detail::make_device_uvector(cudf::host_span<int const>{offs.data(), offs.size()},
                                        cudf::get_default_stream(),
                                        cudf::get_current_device_resource_ref());
  }
};

CUDF_KERNEL void decode_kernel(char const* chars,
                               int const* offsets,
                               int num_inputs,
                               int const* out_offsets,
                               uint8_t* decoded,
                               pqed::string_stats* stats)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(num_inputs)) { return; }
  auto const* begin = chars + offsets[i];
  auto const* end   = chars + offsets[i + 1];
  // The sizing pass must agree with the writing pass
  auto const sized = pqed::decode_json_string<false>(begin, end, nullptr);
  auto const wrote = pqed::decode_json_string<true>(begin, end, decoded + out_offsets[i]);
  stats[i]         = {sized.utf8_bytes == wrote.utf8_bytes ? wrote.utf8_bytes : -1,
              sized.utf16_units == wrote.utf16_units ? wrote.utf16_units : -1};
}

CUDF_KERNEL void compare_kernel(
  char const* chars, int const* offsets, int num_pairs, int* compare_out, uint64_t* code_out)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<std::size_t>(num_pairs)) { return; }
  auto const* a       = reinterpret_cast<uint8_t const*>(chars) + offsets[2 * i];
  auto const a_size   = offsets[2 * i + 1] - offsets[2 * i];
  auto const* b       = reinterpret_cast<uint8_t const*>(chars) + offsets[2 * i + 1];
  auto const b_size   = offsets[2 * i + 2] - offsets[2 * i + 1];
  compare_out[i]      = pqed::utf16_compare(a, a_size, b, b_size);
  code_out[2 * i]     = pqed::short_key_code(a, a_size);
  code_out[2 * i + 1] = pqed::short_key_code(b, b_size);
}

struct decoded_string {
  std::string bytes;
  int64_t utf16_units;
};

/// Runs both decode passes over each JSON string body, returning the decoded bytes
std::vector<decoded_string> decode(std::vector<std::string> const& bodies)
{
  auto const stream = cudf::get_default_stream();
  device_inputs d{bodies};

  // The decoded form is never longer than the escaped form
  std::vector<int> out_offs{0};
  for (auto const& b : bodies) {
    out_offs.push_back(out_offs.back() + static_cast<int>(b.size()) + 4);
  }
  auto d_out_offs =
    cudf::detail::make_device_uvector(cudf::host_span<int const>{out_offs.data(), out_offs.size()},
                                      stream,
                                      cudf::get_current_device_resource_ref());
  rmm::device_uvector<uint8_t> decoded(out_offs.back(), stream);
  rmm::device_uvector<pqed::string_stats> stats(bodies.size(), stream);

  decode_kernel<<<cudf::util::div_rounding_up_safe<int>(bodies.size(), 128),
                  128,
                  0,
                  stream.get()>>>(d.chars.data(),
                                  d.offsets.data(),
                                  static_cast<int>(bodies.size()),
                                  d_out_offs.data(),
                                  decoded.data(),
                                  stats.data());

  auto const h_decoded = cudf::detail::make_std_vector(
    cudf::device_span<uint8_t const>{decoded.data(), decoded.size()}, stream);
  auto const h_stats = cudf::detail::make_std_vector(
    cudf::device_span<pqed::string_stats const>{stats.data(), stats.size()}, stream);

  std::vector<decoded_string> out;
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    EXPECT_GE(h_stats[i].utf8_bytes, 0) << "sizing and writing passes disagree: " << bodies[i];
    out.push_back({std::string(h_decoded.begin() + out_offs[i],
                               h_decoded.begin() + out_offs[i] + h_stats[i].utf8_bytes),
                   h_stats[i].utf16_units});
  }
  return out;
}

}  // namespace

struct VariantJsonStringTest : public cudf::test::BaseFixture {};

TEST_F(VariantJsonStringTest, SimpleEscapes)
{
  auto const out = decode({R"(a\nb)", R"(\b\f\n\r\t)", R"(\"\\\/)", R"(no escapes)", ""});
  EXPECT_EQ(out[0].bytes, std::string("a\nb"));
  EXPECT_EQ(out[1].bytes, std::string("\b\f\n\r\t"));
  EXPECT_EQ(out[2].bytes, std::string("\"\\/"));
  EXPECT_EQ(out[3].bytes, std::string("no escapes"));
  EXPECT_EQ(out[4].bytes, std::string(""));
}

TEST_F(VariantJsonStringTest, UnicodeEscapes)
{
  auto const out = decode({R"(\u0041)",        // ASCII
                           R"(\u00e9)",        // two UTF-8 bytes
                           R"(\u20ac)",        // three UTF-8 bytes
                           R"(\ud83d\ude00)",  // surrogate pair, four UTF-8 bytes
                           R"(\u0000)"});
  EXPECT_EQ(out[0].bytes, std::string("A"));
  EXPECT_EQ(out[1].bytes, std::string("\xc3\xa9"));
  EXPECT_EQ(out[2].bytes, std::string("\xe2\x82\xac"));
  EXPECT_EQ(out[3].bytes, std::string("\xf0\x9f\x98\x80"));
  EXPECT_EQ(out[4].bytes, std::string("\0", 1));

  // UTF-16 counts: a surrogate pair is two units, everything else here is one
  EXPECT_EQ(out[0].utf16_units, 1);
  EXPECT_EQ(out[3].utf16_units, 2);
}

TEST_F(VariantJsonStringTest, UnpairedSurrogatesBecomeQuestionMark)
{
  // Java replaces an unpaired surrogate with '?' when the String is encoded as UTF-8
  auto const out = decode({R"(\ud83d)",              // high surrogate alone
                           R"(\ude00)",              // low surrogate alone
                           R"(\ud83dx)",             // high surrogate then a plain character
                           R"(\ud83d\ud83d\ude00)",  // high surrogate then a valid pair
                           R"(\ude00\ud83d)"});      // reversed order is never a pair
  EXPECT_EQ(out[0].bytes, std::string("?"));
  EXPECT_EQ(out[1].bytes, std::string("?"));
  EXPECT_EQ(out[2].bytes, std::string("?x"));
  EXPECT_EQ(out[3].bytes, std::string("?\xf0\x9f\x98\x80"));
  EXPECT_EQ(out[4].bytes, std::string("??"));
}

TEST_F(VariantJsonStringTest, Utf16CompareOrder)
{
  // Java String.compareTo orders by UTF-16 code unit, which differs from UTF-8 byte order for
  // supplementary characters: U+1F600 is a surrogate pair starting at 0xD83D, so it sorts below
  // U+E000 even though its UTF-8 lead byte (0xF0) is higher.
  std::vector<std::pair<std::string, std::string>> const pairs{
    {"a", "b"},
    {"a", "a"},
    {"a", "ab"},
    {"", "a"},
    {"\xf0\x9f\x98\x80", "\xee\x80\x80"},  // U+1F600 before U+E000
    {"\xee\x80\x80", "z"},                 // U+E000 after ASCII
    {"abcdefgh", "abcdefghi"},             // at the 8-byte fast-path boundary
    {"\xc3\xa9", "\xc3\xa8"},
  };
  std::vector<int> const expected{-1, 0, -1, -1, -1, 1, -1, 1};

  std::vector<std::string> flat;
  for (auto const& [a, b] : pairs) {
    flat.push_back(a);
    flat.push_back(b);
  }
  auto const stream = cudf::get_default_stream();
  device_inputs d{flat};
  rmm::device_uvector<int> cmp(pairs.size(), stream);
  rmm::device_uvector<uint64_t> codes(flat.size(), stream);
  compare_kernel<<<cudf::util::div_rounding_up_safe<int>(pairs.size(), 128),
                   128,
                   0,
                   stream.get()>>>(
    d.chars.data(), d.offsets.data(), static_cast<int>(pairs.size()), cmp.data(), codes.data());

  auto const h_cmp =
    cudf::detail::make_std_vector(cudf::device_span<int const>{cmp.data(), cmp.size()}, stream);
  auto const h_codes = cudf::detail::make_std_vector(
    cudf::device_span<uint64_t const>{codes.data(), codes.size()}, stream);

  for (std::size_t i = 0; i < pairs.size(); ++i) {
    EXPECT_EQ(h_cmp[i], expected[i]) << pairs[i].first << " vs " << pairs[i].second;

    // Where the short-key fast path applies to both keys, its integer order must agree with the
    // full comparison, since the encoder uses it to sort keys
    auto const a_code = h_codes[2 * i];
    auto const b_code = h_codes[2 * i + 1];
    if (a_code != pqed::long_key_code && b_code != pqed::long_key_code) {
      auto const code_order = (a_code > b_code) - (a_code < b_code);
      EXPECT_EQ(code_order, h_cmp[i]) << pairs[i].first << " vs " << pairs[i].second;
    }
  }
}

CUDF_TEST_PROGRAM_MAIN()
