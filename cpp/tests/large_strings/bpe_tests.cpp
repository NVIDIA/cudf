/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "large_strings_fixture.hpp"

#include <tests/text/bpe_data_generator.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

struct BPETest : public cudf::test::StringsLargeTest {};

namespace {
std::vector<cudf::size_type> make_splits(cudf::size_type rows, int multiplier)
{
  std::vector<cudf::size_type> splits;
  for (int n = 1; n < multiplier; ++n) {
    splits.push_back(rows * n);
  }
  return splits;
}
}  // namespace

TEST_F(BPETest, LargeOutput)
{
  // synthetic GPT-2 style table and byte-mapped text
  namespace bpe = cudf::test::bpe;
  bpe::text_generator gen(3);
  auto const text   = bpe::gpt2_byte_map(gen.generate(4'000'000));
  auto const merges = bpe::train_merges(gen.generate(1'000'000), 4000);
  std::vector<std::string> rows;
  std::size_t pos = 0;
  while (pos < text.size()) {
    auto end = std::min(pos + 4096, text.size());
    while (end < text.size() && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) {
      ++end;
    }
    rows.push_back(text.substr(pos, end - pos));
    pos = end;
  }
  auto const input    = cudf::test::strings_column_wrapper(rows.begin(), rows.end());
  auto const mpt      = cudf::test::strings_column_wrapper(merges.begin(), merges.end());
  auto const mp       = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));
  auto const expected = nvtext::byte_pair_encoding(cudf::strings_column_view(input), *mp);

  // replicate the input so the input and output chars exceed max size_type
  auto const view             = cudf::column_view(input);
  int constexpr max_size_type = std::numeric_limits<cudf::size_type>::max();
  auto const multiplier =
    static_cast<int>(max_size_type /
                     cudf::strings_column_view(view).chars_size(cudf::get_default_stream())) +
    1;
  auto const large_input = cudf::concatenate(std::vector<cudf::column_view>(multiplier, view));
  auto const sv          = cudf::strings_column_view(large_input->view());
  EXPECT_EQ(sv.offsets().type(), cudf::data_type{cudf::type_id::INT64});

  auto const result = nvtext::byte_pair_encoding(sv, *mp);
  EXPECT_EQ(cudf::strings_column_view(result->view()).offsets().type(),
            cudf::data_type{cudf::type_id::INT64});
  for (auto const& c : cudf::split(result->view(), make_splits(view.size(), multiplier))) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, expected->view());
  }
}

// Disabled since this requires about 40GB of device memory.
// Run with --gtest_also_run_disabled_tests to verify more than max size_type segments.
TEST_F(BPETest, DISABLED_ManySegments)
{
  // every character is its own (unpairable) segment so the number
  // of segments exceeds max size_type
  std::size_t free_mem = 0, total_mem = 0;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (free_mem < 40UL * 1024 * 1024 * 1024) { GTEST_SKIP() << "requires 40GB of device memory"; }

  auto const mpt = cudf::test::strings_column_wrapper({"a b", "ab c"});
  auto const mp  = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  auto const rows     = std::vector<std::string>(1000, std::string(1000, 'x'));
  auto const input    = cudf::test::strings_column_wrapper(rows.begin(), rows.end());
  std::string encoded = "x";
  for (int i = 1; i < 1000; ++i) {
    encoded += " x";
  }
  auto const encoded_rows = std::vector<std::string>(1000, encoded);
  auto const expected =
    cudf::test::strings_column_wrapper(encoded_rows.begin(), encoded_rows.end());

  auto const view             = cudf::column_view(input);
  int constexpr max_size_type = std::numeric_limits<cudf::size_type>::max();
  auto const multiplier =
    static_cast<int>(max_size_type /
                     cudf::strings_column_view(view).chars_size(cudf::get_default_stream())) +
    1;
  auto const large_input = cudf::concatenate(std::vector<cudf::column_view>(multiplier, view));

  auto const result =
    nvtext::byte_pair_encoding(cudf::strings_column_view(large_input->view()), *mp);
  for (auto const& c : cudf::split(result->view(), make_splits(view.size(), multiplier))) {
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(c, expected);
  }
}
