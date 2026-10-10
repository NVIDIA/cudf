/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <tests/text/bpe_data_generator.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace {

namespace bpe = cudf::test::bpe;

// generated text is not repeated so the benchmark sees realistic segment repetition;
// this should be at least as large as the largest total_bytes value below
constexpr std::size_t text_size     = 128 * 1024 * 1024;
constexpr std::size_t training_size = 2 * 1024 * 1024;  // text used to train the merges table
constexpr int num_merges            = 16384;

struct bpe_data {
  std::vector<std::string> merges;
  std::map<std::string, std::string> texts;  // text_form -> text
};

/**
 * Synthetic data in the style of GPT-2: a merges table trained on generated text.
 * text_form:
 *   raw     - spaces/newlines are not in the table (natural boundaries)
 *   mapped  - GPT-2 byte-level mapped text (space->Ġ, newline->Ċ)
 *   code    - mapped code-like text with indentation (long runs of mergeable Ġ)
 *   nospace - mapped text with whitespace removed (few natural boundaries)
 */
bpe_data const& get_bpe_data()
{
  static bpe_data const data = [] {
    bpe::text_generator gen(42);
    auto const prose = gen.generate(text_size);
    auto const code  = gen.generate(text_size, true);
    bpe_data d;
    d.merges = bpe::train_merges(prose.substr(0, training_size) + code.substr(0, training_size / 4),
                                 num_merges);
    d.texts["raw"]    = prose;
    d.texts["mapped"] = bpe::gpt2_byte_map(prose);
    d.texts["code"]   = bpe::gpt2_byte_map(code);
    // removing whitespace shrinks the text so more is generated to keep it unrepeated
    auto const strip = [](std::string text) {
      std::erase_if(text, [](char c) { return c == ' ' || c == '\n'; });
      return bpe::gpt2_byte_map(text);
    };
    auto nospace = strip(prose);
    while (nospace.size() < text_size) {
      nospace += strip(gen.generate(text_size - nospace.size()));
    }
    d.texts["nospace"] = std::move(nospace);
    return d;
  }();
  return data;
}

/**
 * Builds a strings column of approximately `total_bytes` from the start of the given text
 * with each row being `row_width` bytes (split on UTF-8 character boundaries)
 *
 * The text is not repeated so it must be at least `total_bytes` long.
 */
std::unique_ptr<cudf::column> make_text_column(std::string const& text,
                                               int64_t row_width,
                                               int64_t total_bytes)
{
  auto const is_continuation = [](char c) {
    return (static_cast<unsigned char>(c) & 0xC0) == 0x80;
  };
  CUDF_EXPECTS(static_cast<int64_t>(text.size()) >= total_bytes,
               "benchmark text is smaller than total_bytes");
  auto n = static_cast<std::size_t>(total_bytes);
  while (n < text.size() && is_continuation(text[n])) {
    ++n;
  }
  auto const chars = text.substr(0, n);
  std::vector<int64_t> offsets{0};
  auto const size = static_cast<int64_t>(chars.size());
  for (int64_t pos = 0; pos < size;) {
    auto end = std::min(pos + row_width, size);
    while (end < size && is_continuation(chars[end])) {
      ++end;
    }
    offsets.push_back(end);
    pos = end;
  }
  auto const num_rows = static_cast<cudf::size_type>(offsets.size() - 1);
  auto stream         = cudf::get_default_stream();
  auto d_chars        = rmm::device_buffer(chars.data(), chars.size(), stream);
  auto d_offsets      = [&]() -> std::unique_ptr<cudf::column> {
    if (size < std::numeric_limits<int32_t>::max()) {
      std::vector<int32_t> offsets32(offsets.begin(), offsets.end());
      return cudf::test::fixed_width_column_wrapper<int32_t>(offsets32.begin(), offsets32.end())
        .release();
    }
    return cudf::test::fixed_width_column_wrapper<int64_t>(offsets.begin(), offsets.end())
      .release();
  }();
  return cudf::make_strings_column(num_rows,
                                   std::move(d_offsets),
                                   std::move(d_chars),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace

static void bench_byte_pair_encoding(nvbench::state& state)
{
  auto const text_form   = state.get_string("text_form");
  auto const row_width   = state.get_int64("row_width");
  auto const total_bytes = state.get_int64("total_bytes");

  auto const& data = get_bpe_data();
  auto mpt         = cudf::test::strings_column_wrapper(data.merges.begin(), data.merges.end());
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  auto column = make_text_column(data.texts.at(text_form), row_width, total_bytes);
  auto input  = cudf::strings_column_view(column->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_element_count(input.chars_size(stream), "chars");
  state.add_global_memory_reads<nvbench::int8_t>(column->alloc_size());
  {
    // the output size must be registered before exec() computes the summaries
    auto result = nvtext::byte_pair_encoding(input, *merge_pairs);
    state.add_global_memory_writes<nvbench::int8_t>(result->alloc_size());
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = nvtext::byte_pair_encoding(input, *merge_pairs);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_byte_pair_encoding)
  .set_name("byte_pair_encoding")
  .add_string_axis("text_form", {"raw", "mapped", "code", "nospace"})
  .add_int64_axis("row_width", {256, 4096, 65536, 1048576})
  .add_int64_axis("total_bytes", {16 * 1024 * 1024, 128 * 1024 * 1024});
