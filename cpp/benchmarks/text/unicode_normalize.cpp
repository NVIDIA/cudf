/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/unicode_normalize.hpp>

#include <nvbench/nvbench.cuh>

// char_type = "precomposed"
//
// Latin-1 Supplement precomposed characters (U+00C0–U+00FF subset, 2-byte
// UTF-8).  All are already NFC/NFKC so NFC/NFKC paths exercise the quick
// check; NFD/NFKD paths exercise the full decomposition pipeline.
//
// clang-format off
static char const PRECOMPOSED_PATTERN[] =
  "\xC3\x80\xC3\x81\xC3\x82\xC3\x83\xC3\x84\xC3\x85"  // À Á Â Ã Ä Å
  "\xC3\x87\xC3\x88\xC3\x89\xC3\x8A\xC3\x8B"            // Ç È É Ê Ë
  "\xC3\x8C\xC3\x8D\xC3\x8E\xC3\x8F\xC3\x91"            // Ì Í Î Ï Ñ
  "\xC3\x92\xC3\x93\xC3\x94\xC3\x95\xC3\x96"            // Ò Ó Ô Õ Ö
  "\xC3\x99\xC3\x9A\xC3\x9B\xC3\x9C\xC3\x9D"            // Ù Ú Û Ü Ý
  "\xC3\xA0\xC3\xA1\xC3\xA2\xC3\xA3\xC3\xA4\xC3\xA5"  // à á â ã ä å
  "\xC3\xA7\xC3\xA8\xC3\xA9\xC3\xAA\xC3\xAB"            // ç è é ê ë
  "\xC3\xAC\xC3\xAD\xC3\xAE\xC3\xAF\xC3\xB1"            // ì í î ï ñ
  "\xC3\xB2\xC3\xB3\xC3\xB4\xC3\xB5\xC3\xB6"            // ò ó ô õ ö
  "\xC3\xB9\xC3\xBA\xC3\xBB\xC3\xBC\xC3\xBD\xC3\xBF";  // ù ú û ü ý ÿ
// clang-format on
static auto const PRECOMPOSED_LEN =
  static_cast<cudf::size_type>(sizeof(PRECOMPOSED_PATTERN) - 1);  // 106 bytes, 53 chars

// char_type = "mixed"
//
// Nine 3-byte UTF-8 codepoints (27 bytes/repeat) covering all sample
// character types: fullwidth Latin (NFKD compat), circled digit (NFKD compat),
// FFI ligature (NFKD compat, multi-char output), Angstrom Sign (NFC singleton
// canonical decomposition), and halfwidth Katakana (NFKD compat).
// All four normalization forms trigger the full pipeline on this input.
//
// clang-format off
static char const MIXED_PATTERN[] =
  "\xEF\xBC\xA1\xEF\xBC\xA2\xEF\xBC\xA3"  // ＡＢＣ  fullwidth Latin  (U+FF21–FF23)
  "\xE2\x91\xA0"                            // ①      circled digit one (U+2460)
  "\xEF\xAC\x83"                            // ﬃ      FFI ligature      (U+FB03)
  "\xE2\x84\xAB"                            // Å      Angstrom Sign     (U+212B)
  "\xEF\xBD\xB6\xEF\xBE\x80\xEF\xBE\x85"; // ｶﾀﾅ   halfwidth katakana (U+FF76,FF80,FF85)
// clang-format on
static auto const MIXED_LEN =
  static_cast<cudf::size_type>(sizeof(MIXED_PATTERN) - 1);  // 27 bytes, 9 chars

// 53 precomposed Latin-1 chars + 7 combining marks = 60 rows
static std::unique_ptr<nvtext::unicode_normalizer> make_normalizer_precomposed(
  nvtext::unicode_normalization_form form)
{
  // clang-format off
  cudf::test::strings_column_wrapper codepoints({
    "00C0","00C1","00C2","00C3","00C4","00C5",
    "00C7","00C8","00C9","00CA","00CB",
    "00CC","00CD","00CE","00CF","00D1",
    "00D2","00D3","00D4","00D5","00D6",
    "00D9","00DA","00DB","00DC","00DD",
    "00E0","00E1","00E2","00E3","00E4","00E5",
    "00E7","00E8","00E9","00EA","00EB",
    "00EC","00ED","00EE","00EF","00F1",
    "00F2","00F3","00F4","00F5","00F6",
    "00F9","00FA","00FB","00FC","00FD","00FF",
    "0300","0301","0302","0303","0308","030A","0327"
  });
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({
    0,0,0,0,0,0,   // À Á Â Ã Ä Å
    0,0,0,0,0,     // Ç È É Ê Ë
    0,0,0,0,0,     // Ì Í Î Ï Ñ
    0,0,0,0,0,     // Ò Ó Ô Õ Ö
    0,0,0,0,0,     // Ù Ú Û Ü Ý
    0,0,0,0,0,0,   // à á â ã ä å
    0,0,0,0,0,     // ç è é ê ë
    0,0,0,0,0,     // ì í î ï ñ
    0,0,0,0,0,     // ò ó ô õ ö
    0,0,0,0,0,0,   // ù ú û ü ý ÿ
    230,230,230,230,230,230,202  // combining marks
  });
  cudf::test::strings_column_wrapper decomp_mappings({
    "0041 0300","0041 0301","0041 0302","0041 0303","0041 0308","0041 030A",
    "0043 0327","0045 0300","0045 0301","0045 0302","0045 0308",
    "0049 0300","0049 0301","0049 0302","0049 0308","004E 0303",
    "004F 0300","004F 0301","004F 0302","004F 0303","004F 0308",
    "0055 0300","0055 0301","0055 0302","0055 0308","0059 0301",
    "0061 0300","0061 0301","0061 0302","0061 0303","0061 0308","0061 030A",
    "0063 0327","0065 0300","0065 0301","0065 0302","0065 0308",
    "0069 0300","0069 0301","0069 0302","0069 0308","006E 0303",
    "006F 0300","006F 0301","006F 0302","006F 0303","006F 0308",
    "0075 0300","0075 0301","0075 0302","0075 0308","0079 0301","0079 0308",
    "","","","","","",""  // combining marks have no decomp
  });
  // clang-format on
  return nvtext::create_unicode_normalizer(
    cudf::table_view({codepoints, ccc_values, decomp_mappings}), form);
}

// Mixed character set: 11 rows covering all MIXED_PATTERN codepoints and their
// decomposition targets.
static std::unique_ptr<nvtext::unicode_normalizer> make_normalizer_mixed(
  nvtext::unicode_normalization_form form)
{
  // clang-format off
  cudf::test::strings_column_wrapper codepoints({
    "FF21","FF22","FF23",     // ＡＢＣ  fullwidth Latin
    "2460",                   // ①      circled digit one
    "FB03",                   // ﬃ      FFI ligature
    "212B",                   // Å      Angstrom Sign (singleton canonical → 00C5)
    "FF76","FF80","FF85",     // ｶﾀﾅ   halfwidth katakana
    "00C5",                   // Å      canonical decomp target of 212B
    "030A"                    // ◌̊     combining ring above (CCC=230)
  });
  cudf::test::fixed_width_column_wrapper<int32_t> ccc_values({
    0,0,0,   // fullwidth Latin
    0,       // circled digit
    0,       // FFI ligature
    0,       // Angstrom Sign
    0,0,0,   // halfwidth katakana
    0,       // Å (U+00C5)
    230      // combining ring above
  });
  cudf::test::strings_column_wrapper decomp_mappings({
    "<compat> 0041","<compat> 0042","<compat> 0043",  // fullwidth Latin → ABC
    "<compat> 0031",                                   // ① → 1
    "<compat> 0066 0066 0069",                         // ﬃ → ffi
    "00C5",                                            // Angstrom → Å (singleton canonical)
    "<compat> 30AB","<compat> 30BF","<compat> 30CA",  // halfwidth katakana → fullwidth
    "0041 030A",                                       // Å → A + combining ring (canonical)
    ""                                                 // combining ring: no decomp
  });
  // clang-format on
  return nvtext::create_unicode_normalizer(
    cudf::table_view({codepoints, ccc_values, decomp_mappings}), form);
}

static void bench_unicode_normalize(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width_bytes"));
  auto const form_str  = state.get_string("form");
  auto const char_type = state.get_string("char_type");

  auto const form = [&] {
    if (form_str == "NFD") return nvtext::unicode_normalization_form::NFD;
    if (form_str == "NFKD") return nvtext::unicode_normalization_form::NFKD;
    if (form_str == "NFKC") return nvtext::unicode_normalization_form::NFKC;
    return nvtext::unicode_normalization_form::NFC;
  }();

  bool const is_mixed       = (char_type == "mixed");
  char const* const pattern = is_mixed ? MIXED_PATTERN : PRECOMPOSED_PATTERN;
  auto const pattern_len    = is_mixed ? MIXED_LEN : PRECOMPOSED_LEN;

  // Fill each row with complete repetitions of the pattern up to row_width bytes.
  // For precomposed (2-byte chars) this fills row_width exactly.
  // For mixed (3-byte chars, 27-byte pattern) each row is
  // floor(row_width / 27) * 27 bytes — slightly under row_width.
  std::string row_str;
  row_str.reserve(row_width);
  while (static_cast<cudf::size_type>(row_str.size()) + pattern_len <= row_width) {
    row_str.append(pattern, pattern_len);
  }

  std::vector<std::string> rows(num_rows, row_str);
  cudf::test::strings_column_wrapper str_col(rows.begin(), rows.end());
  auto input_col = str_col.release();
  cudf::strings_column_view input(input_col->view());

  // Normalizer is created once and reused — construction time is not measured.
  auto const normalizer =
    is_mixed ? make_normalizer_mixed(form) : make_normalizer_precomposed(form);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_global_memory_reads<nvbench::int8_t>(input_col->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(input_col->alloc_size());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = nvtext::normalize_unicode(input, *normalizer);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_unicode_normalize)
  .set_name("unicode_normalize")
  .add_string_axis("form", {"NFD", "NFC", "NFKD", "NFKC"})
  .add_string_axis("char_type", {"precomposed", "mixed"})
  .add_int64_axis("num_rows", {32768, 262144})
  .add_int64_axis("row_width_bytes", {128, 512});
