/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <vector>

struct TextUnicodeNormalizeTest : public cudf::test::BaseFixture {};

TEST_F(TextUnicodeNormalizeTest, NullStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {false, false, false});
  cudf::strings_column_view input(strings);

  cudf::test::strings_column_wrapper codepoints({"A"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, strings);
}

TEST_F(TextUnicodeNormalizeTest, MixedNullStrings)
{
  // Only row 1 is null; rows 0 and 2 have content that requires normalization.
  // Verifies that the null mask is propagated correctly and that null rows do
  // not corrupt the byte offsets used by the non-null rows on either side.
  //   row 0: "e\xCC\x81"   (e + U+0301) → NFC → "é" (U+00E9)
  //   row 1: null           → null
  //   row 2: "caf\xC3\xA9" (café, already NFC) → NFC → "café"
  cudf::test::strings_column_wrapper input_strings({"e\xCC\x81", "", "caf\xC3\xA9"},
                                                   {true, false, true});
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"\xC3\xA9", "", "caf\xC3\xA9"}, {true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, AsciiPassthrough)
{
  // ASCII-only input should be unchanged for all four normalization forms
  cudf::test::strings_column_wrapper strings({"hello", "world", "abc 123", ""});
  cudf::strings_column_view input(strings);

  // No codepoints needed for pure ASCII
  cudf::test::strings_column_wrapper codepoints({"A"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  for (auto form : {nvtext::unicode_normalization_form::NFD,
                    nvtext::unicode_normalization_form::NFC,
                    nvtext::unicode_normalization_form::NFKD,
                    nvtext::unicode_normalization_form::NFKC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, strings);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFD_BasicDecomp)
{
  // U+00E9 "é" decomposes to U+0065 "e" + U+0301 combining acute accent
  // Table entry: codepoint="00E9", CCC=0, decomp="0065 0301"
  cudf::test::strings_column_wrapper input_strings({"\xC3\xA9"});  // é in UTF-8
  cudf::strings_column_view input(input_strings);

  // U+0301 (CCC=230) must be in the table so the compose kernel treats it
  // as a combining mark (CCC>0) rather than a new starter.
  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  // NFD: é → e + combining acute (U+0065 U+0301)
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  // Expected: "e" + U+0301 combining acute accent
  cudf::test::strings_column_wrapper expected({"e\xCC\x81"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFD_Hangul)
{
  // U+AC00 "가" algorithmically decomposes to U+1100 + U+1161 under NFD
  // No table entry needed for algorithmic Hangul decomposition
  cudf::test::strings_column_wrapper input_strings({"\xEA\xB0\x80"});  // 가 in UTF-8
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"A"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  // Expected: U+1100 (ᄀ) + U+1161 (ᅡ) in UTF-8
  cudf::test::strings_column_wrapper expected({"\xE1\x84\x80\xE1\x85\xA1"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFC_Compose)
{
  // NFC: e + combining acute → é (U+00E9)
  // Input is already decomposed: U+0065 + U+0301
  // Table needs U+00E9 so the composition lookup can find it
  cudf::test::strings_column_wrapper input_strings({"e\xCC\x81"});  // e + combining acute
  cudf::strings_column_view input(input_strings);

  // U+0301 (CCC=230) must be in the table so the compose kernel treats it
  // as a combining mark (CCC>0) rather than a new starter.
  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  // Expected: é (U+00E9) in UTF-8
  cudf::test::strings_column_wrapper expected({"\xC3\xA9"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFC_HangulBlockingRule)
{
  // L jamo + U+0300 (CCC=230) + V jamo: the combining mark blocks L+V composition.
  // NFC must NOT compose U+1100 + U+1161 across the intervening non-starter.
  // Input:    U+1100 + U+0300 + U+1161  (ᄀ + combining grave + ᅡ)
  // Expected: unchanged — blocking rule prevents algorithmic Hangul composition.
  cudf::test::strings_column_wrapper input_strings(
    {"\xE1\x84\x80\xCC\x80\xE1\x85\xA1"});  // U+1100 + U+0300 + U+1161
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"0300"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({230});
  cudf::test::strings_column_wrapper decomp_mappings({""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"\xE1\x84\x80\xCC\x80\xE1\x85\xA1"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFC_CompositionExclusion)
{
  // U+2ADC (FORKING) is in the composition exclusion list, so NFC must NOT
  // compose U+2ADD + U+0338 → U+2ADC even though U+2ADC has a canonical
  // two-token decomposition.  This exercises the binary search over
  // COMPOSITION_EXCLUSIONS, which must be sorted for correct results.
  // Input:    U+2ADD + U+0338  (⫝ + combining long solidus overlay)
  // Expected: unchanged — exclusion prevents composition.
  cudf::test::strings_column_wrapper input_strings({"\xE2\xAB\x9D\xCC\xB8"});  // U+2ADD + U+0338
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"2ADC", "0338"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 1});
  cudf::test::strings_column_wrapper decomp_mappings({"2ADD 0338", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"\xE2\xAB\x9D\xCC\xB8"});  // unchanged
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFC_HangulCompose)
{
  // Hangul jamo L (U+1100 ᄀ) + V (U+1161 ᅡ) should compose to U+AC00 (가) under NFC.
  // V jamo is NFC_QC=Maybe but has CCC=0 and no decomp entry, so the quick-check
  // predicate must explicitly detect the V/T jamo ranges or the early-return fires
  // and the composition pass never runs.
  // Also tests L + V + T: U+1100 + U+1161 + U+11A8 → U+AC01 (각).
  cudf::test::strings_column_wrapper input_strings(
    {"\xE1\x84\x80\xE1\x85\xA1",                // ᄀ + ᅡ  (L + V)
     "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8"});  // ᄀ + ᅡ + ᆨ  (L + V + T)
  cudf::strings_column_view input(input_strings);

  // No table entries needed: Hangul composition is purely algorithmic.
  cudf::test::strings_column_wrapper codepoints({"0041"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"\xEA\xB0\x80",    // U+AC00 가
                                               "\xEA\xB0\x81"});  // U+AC01 각
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFKD_CompatDecomp)
{
  // U+FB01 "ﬁ" (fi ligature) has compatibility decomposition: "0066 0069" (fi)
  // Under NFD it is unchanged (no canonical decomposition)
  // Under NFKD it expands to "fi"
  cudf::test::strings_column_wrapper input_strings({"\xEF\xAC\x81"});  // ﬁ in UTF-8
  cudf::strings_column_view input(input_strings);

  // Compat decomp is indicated by "<compat>" prefix in the decomp mapping
  cudf::test::strings_column_wrapper codepoints({"FB01"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0066 0069"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  // NFD: ﬁ is unchanged (compatibility decomp not applied)
  auto normalizer_nfd =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);
  auto result = nvtext::normalize_unicode(input, *normalizer_nfd);
  cudf::test::strings_column_wrapper expected_nfd({"\xEF\xAC\x81"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_nfd);

  // NFKD: ﬁ → "fi"
  auto normalizer_nfkd =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKD);
  result = nvtext::normalize_unicode(input, *normalizer_nfkd);
  cudf::test::strings_column_wrapper expected_nfkd({"fi"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_nfkd);
}

TEST_F(TextUnicodeNormalizeTest, NFKC_CompatThenCompose)
{
  // U+FB01 "ﬁ" → NFKC → "fi"
  // compat decomp gives "fi"; f+i has no canonical composition so stays "fi"
  cudf::test::strings_column_wrapper input_strings({"\xEF\xAC\x81"});  // ﬁ in UTF-8
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"FB01"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0066 0069"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"fi"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, CanonicalReorder)
{
  // Test that combining marks are reordered by CCC after NFD decomposition.
  // Construct a string with two combining marks in the wrong order:
  //   U+0041 'A' + U+0316 (CCC=220) + U+0300 (CCC=230)
  // After canonical decomposition and reorder the CCC=220 mark stays before CCC=230.
  // Here we use a string that already has a base + two combining characters
  // where the higher CCC comes first — reorder should swap them.
  //
  // U+0300: combining grave accent      CCC=230
  // U+0316: combining grave accent below CCC=220
  //
  // Input: 'A' + U+0300 (CCC=230) + U+0316 (CCC=220)  — wrong order
  // After NFD reorder: 'A' + U+0316 (CCC=220) + U+0300 (CCC=230) — correct order

  // UTF-8: A=0x41, U+0300=0xCC 0x80, U+0316=0xCC 0x96
  cudf::test::strings_column_wrapper input_strings({"A\xCC\x80\xCC\x96"});
  cudf::strings_column_view input(input_strings);

  // Table entries for the two combining marks (no decomposition, just CCC values)
  cudf::test::strings_column_wrapper codepoints({"0300", "0316"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({230, 220});
  cudf::test::strings_column_wrapper decomp_mappings({"", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  // Expected: 'A' + U+0316 (CCC=220) + U+0300 (CCC=230)
  cudf::test::strings_column_wrapper expected({"A\xCC\x96\xCC\x80"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, MultiStringBatch)
{
  // A batch of 5 strings with mixed content; verify per-row correctness under NFC.
  // Rows:
  //   0: "hello"      → "hello"   (ASCII passthrough)
  //   1: ""           → ""        (empty)
  //   2: nullptr      → nullptr   (null)
  //   3: "e\xCC\x81" → "é"       (compose e + acute → U+00E9)
  //   4: "café"       → "café"    (already NFC)

  cudf::test::strings_column_wrapper input_strings({"hello", "", "", "e\xCC\x81", "caf\xC3\xA9"},
                                                   {true, true, false, true, true});
  cudf::strings_column_view input(input_strings);

  // Table: U+00E9 with canonical decomp "0065 0301"
  // U+0301 (CCC=230) must be in the table so the compose kernel treats it
  // as a combining mark (CCC>0) rather than a new starter.
  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"hello", "", "", "\xC3\xA9", "caf\xC3\xA9"},
                                              {true, true, false, true, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFD_SlicedInput)
{
  // Full column: ["hello", "é", "café"]
  // Slice [1, 3] → ["é", "café"] — chars start at a non-zero offset.
  // NFD should decompose é (U+00E9) → e + U+0301 in both strings.
  cudf::test::strings_column_wrapper full_col({"hello", "\xC3\xA9", "caf\xC3\xA9"});
  auto sliced = cudf::slice(full_col, {1, 3}).front();
  cudf::strings_column_view input(sliced);

  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"e\xCC\x81", "cafe\xCC\x81"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFC_SlicedInput)
{
  // Full column: ["skip", "e\xCC\x81", "caf\xC3\xA9", "skip"]
  // Slice [1, 3] → ["e\xCC\x81", "café"] — chars start at a non-zero offset.
  // NFC: e + U+0301 composes to U+00E9 (é); café is already NFC (quick-check pass).
  cudf::test::strings_column_wrapper full_col({"skip", "e\xCC\x81", "caf\xC3\xA9", "skip"});
  auto sliced = cudf::slice(full_col, {1, 3}).front();
  cudf::strings_column_view input(sliced);

  cudf::test::strings_column_wrapper codepoints({"00E9", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0065 0301", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"\xC3\xA9", "caf\xC3\xA9"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

// ---------------------------------------------------------------------------
// Sample-driven tests covering character types beyond Latin-1 Supplement
// ---------------------------------------------------------------------------

TEST_F(TextUnicodeNormalizeTest, NFC_SingletonCanonical)
{
  // U+212B ANGSTROM SIGN has a canonical (non-compat) singleton decomposition
  // to U+00C5 (Å), which itself decomposes to A + U+030A.  CCC=0 throughout,
  // so the NFC quick check must consult the compat bitset to detect instability.
  //   NFD(U+212B) → A + U+030A
  //   NFC(U+212B) → U+00C5
  cudf::test::strings_column_wrapper input_strings({"\xE2\x84\xAB"});  // U+212B
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"212B", "00C5", "030A"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"00C5", "0041 030A", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  auto nfd =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFD);
  cudf::test::strings_column_wrapper expected_nfd({"A\xCC\x8A"});  // A + U+030A
  auto result = nvtext::normalize_unicode(input, *nfd);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_nfd);

  auto nfc =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);
  result = nvtext::normalize_unicode(input, *nfc);
  cudf::test::strings_column_wrapper expected_nfc({"\xC3\x85"});  // U+00C5
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_nfc);
}

TEST_F(TextUnicodeNormalizeTest, NFC_DecomposedCombining)
{
  // A + U+030A (combining ring above) should compose to U+00C5 under NFC.
  // The quick check must detect the combining mark and run the composition pass.
  cudf::test::strings_column_wrapper input_strings({"A\xCC\x8A"});  // A + U+030A
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"00C5", "030A"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 230});
  cudf::test::strings_column_wrapper decomp_mappings({"0041 030A", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto nfc =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);

  auto result = nvtext::normalize_unicode(input, *nfc);
  cudf::test::strings_column_wrapper expected({"\xC3\x85"});  // U+00C5
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFKD_FullwidthToASCII)
{
  // Fullwidth Latin letters and digits (U+FF21–FF23, U+FF11–FF13) have
  // compatibility decompositions to their ASCII equivalents.
  // NFD/NFC leave them unchanged; NFKD/NFKC map them to "ABC123".
  cudf::test::strings_column_wrapper input_strings(
    {"\xEF\xBC\xA1\xEF\xBC\xA2\xEF\xBC\xA3\xEF\xBC\x91\xEF\xBC\x92\xEF\xBC\x93"});  // ＡＢＣ１２３
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"FF21", "FF22", "FF23", "FF11", "FF12", "FF13"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 0, 0, 0, 0, 0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0041",
                                                      "<compat> 0042",
                                                      "<compat> 0043",
                                                      "<compat> 0031",
                                                      "<compat> 0032",
                                                      "<compat> 0033"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  // NFD/NFC: no canonical decomposition → unchanged
  for (auto form :
       {nvtext::unicode_normalization_form::NFD, nvtext::unicode_normalization_form::NFC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, input_strings);
  }

  // NFKD/NFKC: compatibility decomposition applied → ASCII
  cudf::test::strings_column_wrapper expected({"ABC123"});
  for (auto form :
       {nvtext::unicode_normalization_form::NFKD, nvtext::unicode_normalization_form::NFKC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFKD_HalfwidthKatakana)
{
  // Halfwidth Katakana (U+FF76, U+FF80, U+FF85) have compatibility decompositions
  // to their fullwidth Katakana equivalents.  Input spells "katakana" (ｶﾀｶﾅ).
  cudf::test::strings_column_wrapper input_strings(
    {"\xEF\xBD\xB6\xEF\xBE\x80\xEF\xBD\xB6\xEF\xBE\x85"});  // ｶﾀｶﾅ
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"FF76", "FF80", "FF85"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 0, 0});
  cudf::test::strings_column_wrapper decomp_mappings(
    {"<compat> 30AB", "<compat> 30BF", "<compat> 30CA"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  // NFKD/NFKC: halfwidth → fullwidth katakana (カタカナ)
  cudf::test::strings_column_wrapper expected(
    {"\xE3\x82\xAB\xE3\x82\xBF\xE3\x82\xAB\xE3\x83\x8A"});  // カタカナ
  for (auto form :
       {nvtext::unicode_normalization_form::NFKD, nvtext::unicode_normalization_form::NFKC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFKD_CircledDigit)
{
  // U+2460 CIRCLED DIGIT ONE has a compatibility decomposition to "1" (U+0031).
  cudf::test::strings_column_wrapper input_strings({"\xE2\x91\xA0"});  // ①
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"2460"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0031"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKD);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"1"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, NFKD_Ligature_FFI)
{
  // U+FB03 LATIN SMALL LIGATURE FFI has a compatibility decomposition to "ffi".
  cudf::test::strings_column_wrapper input_strings({"\xEF\xAC\x83"});  // ﬃ
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"FB03"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0066 0066 0069"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKD);

  auto result = nvtext::normalize_unicode(input, *normalizer);
  cudf::test::strings_column_wrapper expected({"ffi"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(TextUnicodeNormalizeTest, EmptyDecompositionTable)
{
  cudf::test::strings_column_wrapper strcol(std::initializer_list<std::string>{});
  cudf::test::fixed_width_column_wrapper<int32_t> intcol(std::initializer_list<int32_t>{});
  cudf::table_view t({strcol, intcol, strcol});
  EXPECT_THROW(nvtext::create_unicode_normalizer(t, nvtext::unicode_normalization_form::NFC),
               std::invalid_argument);
}

TEST_F(TextUnicodeNormalizeTest, ErrorWrongColumnCount)
{
  cudf::test::strings_column_wrapper strcol({"00E9"});
  cudf::test::fixed_width_column_wrapper<int32_t> intcol({0});
  cudf::table_view t({strcol, intcol, strcol, strcol});  // 4 columns instead of 3
  EXPECT_THROW(nvtext::create_unicode_normalizer(t, nvtext::unicode_normalization_form::NFC),
               std::invalid_argument);
}

TEST_F(TextUnicodeNormalizeTest, ErrorWrongColumnType)
{
  auto const form = nvtext::unicode_normalization_form::NFC;
  cudf::test::fixed_width_column_wrapper<int32_t> intcol({0x00E9});
  cudf::test::strings_column_wrapper strcol({"0065 0301"});

  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({intcol, intcol, strcol}), form),
               std::invalid_argument);
  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({strcol, strcol, strcol}), form),
               std::invalid_argument);
  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({strcol, intcol, intcol}), form),
               std::invalid_argument);
}

TEST_F(TextUnicodeNormalizeTest, ErrorNullsInColumns)
{
  auto const form = nvtext::unicode_normalization_form::NFC;
  cudf::test::strings_column_wrapper col0({"00E9"}, {false});
  cudf::test::fixed_width_column_wrapper<int32_t> col1({0});
  cudf::test::strings_column_wrapper col2({"0065 0301"});
  EXPECT_THROW(nvtext::create_unicode_normalizer(cudf::table_view({col0, col1, col2}), form),
               std::invalid_argument);
}

TEST_F(TextUnicodeNormalizeTest, NFC_ScriptExclusion)
{
  // U+0958 DEVANAGARI LETTER QA is a composition exclusion (Unicode SpecialCasing /
  // CompositionExclusions.txt).  It has a canonical decomposition to U+0915 + U+093C
  // but must never recompose.  Before the fix the NFC quick check skipped U+0958
  // (CCC=0, no compat tag, not a singleton), so NFC returned the undecomposed form.
  //
  // Input : U+0958  (\xE0\xA5\x98)
  // NFC   : U+0915 U+093C  (\xE0\xA4\x95\xE0\xA4\xBC)  — decomposed, not recomposed
  cudf::test::strings_column_wrapper input_strings({"\xE0\xA5\x98"});
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"0958", "0915", "093C"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 0, 7});
  cudf::test::strings_column_wrapper decomp_mappings({"0915 093C", "", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  for (auto form :
       {nvtext::unicode_normalization_form::NFD, nvtext::unicode_normalization_form::NFC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    cudf::test::strings_column_wrapper expected({"\xE0\xA4\x95\xE0\xA4\xBC"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFC_NonStarterDecomposition)
{
  // U+0F73 TIBETAN VOWEL SIGN II has a canonical decomposition to U+0F71 (CCC=129)
  // followed by U+0F72 (CCC=130).  Because the first token has CCC != 0, this is a
  // non-starter decomposition and U+0F73 is excluded from recomposition (NFC_QC=No).
  // Before the fix the quick check did not flag U+0F73, so NFC silently returned the
  // undecomposed form.
  //
  // Input : U+0F73  (\xE0\xBD\xB3)
  // NFC   : U+0F71 U+0F72  (\xE0\xBD\xB1\xE0\xBD\xB2)  — decomposed, not recomposed
  cudf::test::strings_column_wrapper input_strings({"\xE0\xBD\xB3"});
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"0F73", "0F71", "0F72"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 129, 130});
  cudf::test::strings_column_wrapper decomp_mappings({"0F71 0F72", "", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  for (auto form :
       {nvtext::unicode_normalization_form::NFD, nvtext::unicode_normalization_form::NFC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    cudf::test::strings_column_wrapper expected({"\xE0\xBD\xB1\xE0\xBD\xB2"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(TextUnicodeNormalizeTest, NullCodepointPassthrough)
{
  // U+0000 (NULL) is a valid Unicode codepoint that must survive all four
  // normalization forms.  The old implementation used 0 as a "consumed by
  // composition" sentinel, so U+0000 was silently dropped.
  // Input: "\x00A\x00" — NUL, 'A', NUL
  cudf::test::strings_column_wrapper input_strings(
    {std::string("\x00"
                 "A\x00",
                 3)});
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"0041"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  for (auto form : {nvtext::unicode_normalization_form::NFD,
                    nvtext::unicode_normalization_form::NFC,
                    nvtext::unicode_normalization_form::NFKD,
                    nvtext::unicode_normalization_form::NFKC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, input_strings);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFKC_IndirectCompatDecomposition)
{
  // U+0385 GREEK DIALYTIKA TONOS has a canonical decomposition to U+00A8 + U+0301.
  // U+00A8 DIAERESIS has a compatibility decomposition to U+0020 + U+0308.
  // U+0385 has CCC=0, no compat tag, and count≠1, so scatter_compat_flag_fn skips
  // it.  The new propagation pass detects that U+00A8 (a canonical decomp token) is
  // already flagged and flags U+0385 too.
  //
  // Input : U+0385                   (\xCE\x85)
  // NFC   : U+0385                   (\xCE\x85)          — recomposes from U+00A8 + U+0301
  // NFKC  : U+0020 U+0308 U+0301    ("\x20\xCC\x88\xCC\x81")
  cudf::test::strings_column_wrapper input_strings({"\xCE\x85"});
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"0385", "00A8", "0020", "0308", "0301"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 0, 0, 230, 230});
  cudf::test::strings_column_wrapper decomp_mappings(
    {"00A8 0301", "<compat> 0020 0308", "", "", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  // NFC: canonical decomp then recompose → identity
  {
    auto normalizer =
      nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);
    auto result = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, input_strings);
  }

  // NFKC: compat expands U+00A8 → U+0020 U+0308; no recomposition possible
  {
    auto normalizer =
      nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKC);
    auto result = nvtext::normalize_unicode(input, *normalizer);
    cudf::test::strings_column_wrapper expected({"\x20\xCC\x88\xCC\x81"});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFC_StarterStarterComposition)
{
  // Bengali canonical composition where both operands have CCC=0.
  // U+09CB BENGALI VOWEL SIGN O decomposes to U+09C7 + U+09BE; both components
  // have CCC=0, so the original compose_fn only tried Hangul algorithmic
  // composition and missed the table-based pair.  The quick check also had to be
  // updated to flag U+09BE (CCC=0 second operand) so it doesn't short-circuit.
  //
  // Input : U+09C7 U+09BE  (\xE0\xA7\x87 \xE0\xA6\xBE)
  // NFC   : U+09CB          (\xE0\xA7\x8B)
  cudf::test::strings_column_wrapper input_strings({"\xE0\xA7\x87\xE0\xA6\xBE"});
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"09CB", "09C7", "09BE"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0, 0, 0});
  cudf::test::strings_column_wrapper decomp_mappings({"09C7 09BE", "", ""});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  // NFD/NFKD: decomposed input has no canonical decomposition entry → unchanged
  for (auto form :
       {nvtext::unicode_normalization_form::NFD, nvtext::unicode_normalization_form::NFKD}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, input_strings);
  }

  // NFC/NFKC: must compose to U+09CB
  cudf::test::strings_column_wrapper expected({"\xE0\xA7\x8B"});
  for (auto form :
       {nvtext::unicode_normalization_form::NFC, nvtext::unicode_normalization_form::NFKC}) {
    auto normalizer = nvtext::create_unicode_normalizer(unicode_data, form);
    auto result     = nvtext::normalize_unicode(input, *normalizer);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TEST_F(TextUnicodeNormalizeTest, NFC_CompatOnlyIsStable)
{
  // U+FB01 (fi ligature) has only a compatibility decomposition to "fi"; it has
  // no canonical decomposition and is therefore NFC-stable.  The NFC quick check
  // must not flag it, so the early-return path fires and the column is returned
  // as-is without running the full normalization pipeline.
  cudf::test::strings_column_wrapper input_strings({"\xEF\xAC\x81"});  // U+FB01 ﬁ
  cudf::strings_column_view input(input_strings);

  cudf::test::strings_column_wrapper codepoints({"FB01"});
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({0});
  cudf::test::strings_column_wrapper decomp_mappings({"<compat> 0066 0069"});
  auto unicode_data = cudf::table_view({codepoints, ccc_values, decomp_mappings});

  // NFC: U+FB01 is NFC-stable — must be returned unchanged
  auto normalizer =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFC);
  auto result = nvtext::normalize_unicode(input, *normalizer);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, input_strings);

  // NFKC: compatibility decomposition is applied -> "fi"
  auto normalizer_nfkc =
    nvtext::create_unicode_normalizer(unicode_data, nvtext::unicode_normalization_form::NFKC);
  result = nvtext::normalize_unicode(input, *normalizer_nfkc);
  cudf::test::strings_column_wrapper expected_nfkc({"fi"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_nfkc);
}
