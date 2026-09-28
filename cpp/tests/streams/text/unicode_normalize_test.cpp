/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <cstdint>

class TextUnicodeNormalizeStreamTest : public cudf::test::BaseFixture {};

TEST_F(TextUnicodeNormalizeStreamTest, NormalizeUnicode)
{
  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  auto stream = cudf::test::get_default_stream();

  auto normalizer = nvtext::create_unicode_normalizer(
    unicode_data, nvtext::unicode_normalization_form::NFC, stream);

  auto const input = cudf::test::strings_column_wrapper({"e\xCC\x81", "\xC3\xA9", "hello"});
  nvtext::normalize_unicode(cudf::strings_column_view(input), *normalizer, stream);
}
