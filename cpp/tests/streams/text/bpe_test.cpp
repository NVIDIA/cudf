/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/byte_pair_encoding.hpp>

struct TextBytePairEncoding : public cudf::test::BaseFixture {};

TEST_F(TextBytePairEncoding, BytePairEncoding)
{
  auto stream = cudf::test::get_default_stream();
  // partial table based on values from https://huggingface.co/gpt2/raw/main/merges.txt
  auto mpt = cudf::test::strings_column_wrapper({
    "e n",    // 14
    "i t",    // 16
    "i s",    // 17
    "e s",    // 20
    "en t",   // 44
    "c e",    // 90
    "es t",   // 141
    "en ce",  // 340
    "t h",    // 146
    "h i",    // 5049
    "th is",  // 5407
    "t est",  // 9034
    "s i",    // 13142
    "s ent"   // 33832
  });

  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt), stream);

  auto validity = cudf::test::iterators::null_at(4);
  cudf::test::strings_column_wrapper input(
    {"thisisit", "thisis test-sentence-1", "thisistestsentence-2", "this-istestsentence 3", "", ""},
    validity);
  auto sv = cudf::strings_column_view(input);

  auto results =
    nvtext::byte_pair_encoding(sv, *merge_pairs, cudf::string_scalar(" ", true, stream), stream);
}
