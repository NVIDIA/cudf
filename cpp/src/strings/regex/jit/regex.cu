/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/cache.hpp"
#include "regex_ir.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/experimental/strings/regex.hpp>
#include <cudf/reduction.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/transform.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/utility>
#include <cuda_runtime_api.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf::experimental {

struct retained_kernel {
  kernel value;
  std::uint32_t threads;
};

struct regex_jit_program::regex_jit_program_impl {
  size_type capture_count;
  std::vector<retained_kernel> kernels;
  std::vector<retained_kernel> cache_size_kernels;
  std::vector<retained_kernel> cache_emit_kernels;
  std::vector<retained_kernel> sample_kernels;
  std::vector<retained_kernel> warp_literal_kernels;
  regex_ir::executor_kind executor;
  size_type cache_values;
  std::optional<std::string> validation_error;
};

struct regex_jit_program_accessor {
  static retained_kernel const& kernel(regex_jit_program const& prog,
                                       bool offset64,
                                       bool emit            = false,
                                       bool output_offset64 = false);

  static retained_kernel const& cache_size_kernel(regex_jit_program const& prog, bool offset64)
  {
    return prog._impl->cache_size_kernels.at(static_cast<std::size_t>(offset64));
  }

  static retained_kernel const& cache_emit_kernel(regex_jit_program const& prog,
                                                  bool offset64,
                                                  bool output_offset64 = false)
  {
    auto replace = prog.operation() == regex_operation::REPLACE ||
                   prog.operation() == regex_operation::REPLACE_WITH_BACKREFS;
    auto index =
      replace ? static_cast<std::size_t>(offset64) * 2U + static_cast<std::size_t>(output_offset64)
              : static_cast<std::size_t>(offset64);
    return prog._impl->cache_emit_kernels.at(index);
  }

  static retained_kernel const& sample_kernel(regex_jit_program const& prog, bool offset64)
  {
    return prog._impl->sample_kernels.at(static_cast<std::size_t>(offset64));
  }

  static retained_kernel const* warp_literal_kernel(regex_jit_program const& prog, bool offset64)
  {
    return prog._impl->warp_literal_kernels.empty()
             ? nullptr
             : &prog._impl->warp_literal_kernels.at(static_cast<std::size_t>(offset64));
  }

  static regex_ir::executor_kind executor(regex_jit_program const& prog)
  {
    return prog._impl->executor;
  }

  static size_type cache_values(regex_jit_program const& prog) { return prog._impl->cache_values; }

  static bool has_span_cache(regex_jit_program const& prog)
  {
    return !prog._impl->sample_kernels.empty();
  }

  static regex_jit_program_options const& options(regex_jit_program const& prog)
  {
    return prog._options;
  }

  static std::optional<std::string> const& validation_error(regex_jit_program const& prog)
  {
    return prog._impl->validation_error;
  }
};

namespace {

using pair_t = cuda::std::pair<char const*, size_type>;

inline constexpr std::string_view KERNEL_ENTRY = "cudf_kernel_entry";

static_assert(sizeof(pair_t) == 16);
static_assert(alignof(pair_t) == alignof(char const*));

constexpr std::size_t required_stack_size = 64U * 1024U;

std::uint32_t block_size(regex_ir::operation_kind operation, regex_ir::executor_kind executor)
{
  auto literal = executor == regex_ir::executor_kind::SINGLE_BYTE_LITERAL ||
                 executor == regex_ir::executor_kind::PACKED_ASCII_LITERAL ||
                 executor == regex_ir::executor_kind::PACKED_UTF8_LITERAL;
  // Boolean and direct-literal scans benefit from retaining each thread's sequential input window
  // in L1. Complex ordered span operations are divergence-limited and need smaller blocks for
  // occupancy and scheduling.
  return !literal && (operation == regex_ir::operation_kind::COUNT ||
                      operation == regex_ir::operation_kind::REPLACE ||
                      operation == regex_ir::operation_kind::SPLIT)
           ? 256
           : 1024;
}

struct input_data {
  char const* chars;
  void const* offsets;
  bitmask_type const* validity;
  size_type row_offset;
  size_type rows;
  std::int64_t chars_bytes;
  bool offset64;
};

input_data get_input_data(strings_column_view const& input, cuda::stream_ref stream)
{
  auto offsets = input.offsets();
  CUDF_EXPECTS(offsets.type().id() == type_id::INT32 || offsets.type().id() == type_id::INT64,
               "Unsupported strings offset type");
  return input_data{input.chars_begin(stream),
                    offsets.type().id() == type_id::INT64
                      ? static_cast<void const*>(offsets.data<std::int64_t>())
                      : static_cast<void const*>(offsets.data<std::int32_t>()),
                    input.parent().null_mask(),
                    input.offset(),
                    input.size(),
                    input.chars_size(stream),
                    offsets.type().id() == type_id::INT64};
}

std::string normalize_pattern(std::string_view pattern)
{
  std::string result;
  result.reserve(pattern.size());
  for (std::size_t position = 0; position < pattern.size();) {
    if (pattern[position] != '\\' || position + 1 == pattern.size()) {
      result.push_back(pattern[position++]);
      continue;
    }

    auto escaped = pattern[position + 1];
    if (escaped < '0' || escaped > '7') {
      result.append(pattern.substr(position, 2));
      position += 2;
      continue;
    }

    position += 1;
    std::uint32_t value = 0;
    std::size_t digits  = 0;
    while (position < pattern.size() && digits < 3 && pattern[position] >= '0' &&
           pattern[position] <= '7') {
      value = (value << 3U) | static_cast<std::uint32_t>(pattern[position] - '0');
      ++position;
      ++digits;
    }
    result += value <= 0xff ? std::format("\\x{:02X}", value) : std::format("\\u{:04X}", value);
  }
  return result;
}

regex_ir::compile_options make_compile_options(strings::regex_flags flags)
{
  regex_ir::compile_options options;
  options.case_insensitive          = strings::is_ignorecase(flags);
  options.multiline                 = strings::is_multiline(flags);
  options.dot_all                   = strings::is_dotall(flags);
  options.ascii_classes             = strings::is_ascii(flags);
  options.extended_newline          = strings::is_ext_newline(flags);
  options.find_match_end_observable = false;
  return options;
}

regex_ir::operation_kind internal_operation(regex_operation operation)
{
  switch (operation) {
    case regex_operation::CONTAINS:
    case regex_operation::MATCHES: return regex_ir::operation_kind::CONTAINS;
    case regex_operation::COUNT: return regex_ir::operation_kind::COUNT;
    case regex_operation::FIND: return regex_ir::operation_kind::FIND;
    case regex_operation::EXTRACT:
    case regex_operation::EXTRACT_SINGLE:
    case regex_operation::EXTRACT_ALL_RECORD:
    case regex_operation::FINDALL:
    case regex_operation::REPLACE:
    case regex_operation::REPLACE_WITH_BACKREFS: return regex_ir::operation_kind::EXTRACT;
    case regex_operation::SPLIT:
    case regex_operation::RSPLIT:
    case regex_operation::SPLIT_RECORD:
    case regex_operation::RSPLIT_RECORD: return regex_ir::operation_kind::SPLIT;
  }
  CUDF_FAIL("Invalid JIT regex operation");
}

void expect_operation(regex_jit_program const& prog, regex_operation expected, std::string_view api)
{
  CUDF_EXPECTS(prog.operation() == expected,
               std::format("{} requires a regex_jit_program created for that operation", api));
}

void expect_split_parameters(regex_jit_program const& prog,
                             size_type maxsplit,
                             std::string_view api)
{
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  CUDF_EXPECTS(regex_jit_program_accessor::options(prog).maxsplit == maxsplit,
               std::format("{} maxsplit does not match the regex_jit_program maxsplit", api));
}

void ensure_stack_size()
{
  std::size_t stack_size = 0;
  CUDF_CUDA_TRY(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
  if (stack_size < required_stack_size) {
    CUDF_CUDA_TRY(cudaDeviceSetLimit(cudaLimitStackSize, required_stack_size));
  }
}

retained_kernel compile_kernel(std::string const& matcher,
                               regex_ir::operation_kind operation,
                               regex_ir::executor_kind executor,
                               std::string wrapper,
                               std::string_view name)
{
  auto threads                      = block_size(operation, executor);
  auto module                       = regex_ir::nvvm::assemble(matcher, std::move(wrapper));
  auto fragment                     = get_nvvm_fragment(std::string{name}, module);
  rtcx::memory_fragment fragments[] = {
    {.data = fragment->view(), .type = rtcx::binary_type::LTO_IR, .name = nullptr}};
  auto compiled   = get_lto_linked_kernel(std::string{name}, {}, fragments);
  auto attributes = cudaFuncAttributes{};
  CUDF_CUDA_TRY(cudaFuncGetAttributes(&attributes, compiled.get().get()));
  auto maximum_threads = attributes.maxThreadsPerBlock;
  CUDF_EXPECTS(maximum_threads >= 32, "regex JIT kernel does not support a full warp");
  threads = std::min(threads, static_cast<std::uint32_t>(maximum_threads));
  threads -= threads % 32U;
  return {std::move(compiled), threads};
}

template <typename... Args>
void launch(retained_kernel const& prepared,
            input_data const& input,
            cuda::stream_ref stream,
            Args... args)
{
  if (input.rows == 0) { return; }
  auto chars      = const_cast<char*>(input.chars);
  auto offsets    = const_cast<void*>(input.offsets);
  auto validity   = const_cast<bitmask_type*>(input.validity);
  auto row_offset = input.row_offset;
  auto rows       = input.rows;
  auto threads    = prepared.threads;
  auto grid =
    static_cast<std::uint32_t>((static_cast<std::uint32_t>(rows) + threads - 1) / threads);
  prepared.value.launch_with(
    {grid, 1, 1}, {threads, 1, 1}, 0, stream, chars, offsets, validity, row_offset, rows, args...);
}

template <typename... Args>
void launch_warp_per_row(retained_kernel const& prepared,
                         input_data const& input,
                         cuda::stream_ref stream,
                         Args... args)
{
  if (input.rows == 0) { return; }
  auto chars      = const_cast<char*>(input.chars);
  auto offsets    = const_cast<void*>(input.offsets);
  auto validity   = const_cast<bitmask_type*>(input.validity);
  auto row_offset = input.row_offset;
  auto rows       = input.rows;
  auto threads    = prepared.threads;
  auto warps      = threads / 32U;
  auto grid = static_cast<std::uint32_t>((static_cast<std::uint32_t>(rows) + warps - 1U) / warps);
  prepared.value.launch_with(
    {grid, 1, 1}, {threads, 1, 1}, 0, stream, chars, offsets, validity, row_offset, rows, args...);
}

struct span_cache_plan {
  size_type capacity{};

  explicit operator bool() const { return capacity > 0; }
};

span_cache_plan select_span_cache(input_data const& input,
                                  regex_jit_program const& prog,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  constexpr auto max_scratch_bytes = std::size_t{64} << 20;
  constexpr auto max_capacity      = size_type{4};
  constexpr auto minimum_rows      = size_type{131072};
  constexpr auto maximum_samples   = size_type{2048};

  if (!regex_jit_program_accessor::has_span_cache(prog) || input.rows == 0) { return {}; }
  auto split_operation = prog.operation() == regex_operation::SPLIT ||
                         prog.operation() == regex_operation::RSPLIT ||
                         prog.operation() == regex_operation::SPLIT_RECORD ||
                         prog.operation() == regex_operation::RSPLIT_RECORD;
  auto values          = regex_jit_program_accessor::cache_values(prog);
  auto bytes_per_match = static_cast<std::size_t>(values) * sizeof(std::int64_t);
  if (bytes_per_match == 0) { return {}; }
  auto records       = max_scratch_bytes / static_cast<std::size_t>(input.rows) / bytes_per_match;
  auto fixed_records = split_operation ? std::size_t{1} : std::size_t{0};
  if (records <= fixed_records) { return {}; }
  auto capacity = std::min<std::size_t>(max_capacity, records - fixed_records);

  auto policy = regex_jit_program_accessor::options(prog).span_cache_policy;
  if (policy == regex_jit_span_cache_policy::OFF) { return {}; }
  if (policy == regex_jit_span_cache_policy::FORCE) { return {static_cast<size_type>(capacity)}; }
  if (input.rows < minimum_rows) { return {}; }

  auto executor_weight = [&] {
    switch (regex_jit_program_accessor::executor(prog)) {
      case regex_ir::executor_kind::RECURSIVE_THOMPSON: return 4U;
      case regex_ir::executor_kind::ASSERTION_AWARE_DETERMINISTIC:
      case regex_ir::executor_kind::PRIORITIZED_DETERMINISTIC:
      case regex_ir::executor_kind::TAGGED_PRIORITIZED_DETERMINISTIC: return 2U;
      case regex_ir::executor_kind::STREAMING_PRIORITIZED_GLUSHKOV:
        return split_operation ? 2U : 0U;
      default: return 0U;
    }
  }();
  if (split_operation) { executor_weight *= 2; }
  if (executor_weight == 0) { return {}; }
  auto known_average_bytes = static_cast<double>(input.chars_bytes) / input.rows;
  if (known_average_bytes * executor_weight < 384.0) { return {}; }

  auto samples = std::min(input.rows, maximum_samples);
  rmm::device_uvector<std::uint64_t> statistics(4, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(statistics.data(), 0, statistics.size() * sizeof(std::uint64_t), stream.get()));
  auto& prepared  = regex_jit_program_accessor::sample_kernel(prog, input.offset64);
  auto chars      = const_cast<char*>(input.chars);
  auto offsets    = const_cast<void*>(input.offsets);
  auto validity   = const_cast<bitmask_type*>(input.validity);
  auto row_offset = input.row_offset;
  auto rows       = input.rows;
  auto threads    = prepared.threads;
  auto grid =
    static_cast<std::uint32_t>((static_cast<std::uint32_t>(samples) + threads - 1) / threads);
  prepared.value.launch_with({grid, 1, 1},
                             {threads, 1, 1},
                             0,
                             stream,
                             chars,
                             offsets,
                             validity,
                             row_offset,
                             rows,
                             samples,
                             static_cast<size_type>(capacity),
                             statistics.data());
  std::array<std::uint64_t, 4> host_statistics{};
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_statistics.data(),
                                statistics.data(),
                                statistics.size() * sizeof(std::uint64_t),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  auto valid = host_statistics[0];
  if (valid == 0) { return {}; }
  auto average_bytes   = static_cast<double>(host_statistics[1]) / valid;
  auto average_matches = static_cast<double>(host_statistics[2]) / valid;
  auto overflow_rate   = static_cast<double>(host_statistics[3]) / valid;
  auto beneficial      = average_bytes * executor_weight >= 384.0;
  auto bounded         = average_matches <= std::min<double>(capacity, 1.25);
  return beneficial && bounded && overflow_rate <= 0.01
           ? span_cache_plan{static_cast<size_type>(capacity)}
           : span_cache_plan{};
}

template <typename Offset>
struct cached_enumeration_writer {
  char const* chars;
  Offset const* input_offsets;
  size_type row_offset;
  std::int64_t const* cache;
  std::uint8_t const* overflow;
  size_type const* output_offsets;
  pair_t* output;
  size_type capacity;
  size_type capture_slots;
  size_type groups;
  bool findall;

  __device__ void operator()(size_type row) const
  {
    if (overflow[row] != 0) { return; }
    auto output_begin = output_offsets[row];
    auto output_count = output_offsets[row + 1] - output_begin;
    auto matches      = findall ? output_count : output_count / groups;
    auto row_chars    = chars + input_offsets[row_offset + row];
    for (size_type match = 0; match < matches; ++match) {
      auto* captures = cache + (static_cast<std::size_t>(row) * capacity + match) * capture_slots;
      auto selected_groups = findall ? size_type{1} : groups;
      for (size_type group = 0; group < selected_groups; ++group) {
        auto slot     = findall ? (groups == 0 ? 0 : 2) : 2 * (group + 1);
        auto begin    = captures[slot];
        auto end      = captures[slot + 1];
        auto index    = output_begin + match * selected_groups + group;
        output[index] = begin >= 0 && end >= 0
                          ? pair_t{end == begin ? reinterpret_cast<char const*>(input_offsets)
                                                : row_chars + begin,
                                   static_cast<size_type>(end - begin)}
                          : pair_t{nullptr, 0};
      }
    }
  }
};

template <typename Offset>
struct cached_split_writer {
  char const* chars;
  Offset const* input_offsets;
  size_type row_offset;
  std::int64_t const* cache;
  std::uint8_t const* overflow;
  size_type const* effective_offsets;
  size_type const* full_offsets;
  pair_t* output;
  size_type capacity;
  bool reverse;

  __device__ void operator()(size_type row) const
  {
    if (overflow[row] != 0) { return; }
    auto effective_begin = effective_offsets[row];
    auto effective_count = effective_offsets[row + 1] - effective_begin;
    auto full_count      = full_offsets[row + 1] - full_offsets[row];
    auto truncated       = full_count > effective_count;
    auto row_chars       = chars + input_offsets[row_offset + row];
    auto row_size        = static_cast<std::int64_t>(input_offsets[row_offset + row + 1] -
                                              input_offsets[row_offset + row]);
    auto* row_spans =
      cache + static_cast<std::size_t>(row) * (static_cast<std::size_t>(capacity) + 1) * 2;
    for (size_type token = 0; token < effective_count; ++token) {
      size_type source = token;
      if (reverse && truncated) {
        auto removed = full_count - effective_count;
        source       = token == 0 ? 0 : removed + token;
      }
      auto begin = row_spans[static_cast<std::size_t>(source) * 2];
      auto end   = row_spans[static_cast<std::size_t>(source) * 2 + 1];
      if (reverse && truncated && token == 0) {
        auto merged = full_count - effective_count;
        end         = row_spans[static_cast<std::size_t>(merged) * 2 + 1];
      } else if (!reverse && truncated && token + 1 == effective_count) {
        end = row_size;
      }
      output[effective_begin + token] =
        pair_t{end == begin ? reinterpret_cast<char const*>(input_offsets) : row_chars + begin,
               static_cast<size_type>(end - begin)};
    }
  }
};
std::unique_ptr<column> fixed_result(strings_column_view const& input,
                                     regex_jit_program const& prog,
                                     data_type output_type,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  auto result =
    make_numeric_column(output_type,
                        input.size(),
                        cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr()),
                        input.null_count(),
                        stream,
                        mr.get_output_mr());
  if (input.is_empty()) { return result; }
  auto data          = get_input_data(input, stream);
  auto* warp         = regex_jit_program_accessor::warp_literal_kernel(prog, data.offset64);
  auto non_null_rows = input.size() - input.null_count();
  auto warp_parallel = non_null_rows > 0 && data.chars_bytes / non_null_rows > 64;
  if (warp != nullptr && warp_parallel) {
    launch_warp_per_row(*warp, data, stream, result->mutable_view().head<void>());
  } else {
    auto& prepared = regex_jit_program_accessor::kernel(prog, data.offset64);
    launch(prepared, data, stream, result->mutable_view().head<void>());
  }
  return result;
}

std::unique_ptr<column> make_strings(rmm::device_uvector<pair_t> const& pairs,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  return cudf::make_strings_column(
    device_span<pair_t const>{pairs.data(), pairs.size()}, stream, mr.get_output_mr());
}

std::unique_ptr<table> extract_impl(strings_column_view const& input,
                                    regex_jit_program const& prog,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  auto groups = prog.groups_count();
  CUDF_EXPECTS(groups > 0, "Group indicators not found in regex pattern");

  if (input.is_empty()) {
    std::vector<std::unique_ptr<column>> columns;
    columns.reserve(groups);
    for (size_type group = 0; group < groups; ++group) {
      columns.emplace_back(make_empty_column(type_id::STRING));
    }
    return std::make_unique<table>(std::move(columns));
  }

  auto pair_count = static_cast<std::size_t>(input.size()) * static_cast<std::size_t>(groups);
  rmm::device_uvector<pair_t> pairs(pair_count, stream, mr.get_temporary_mr());
  auto data      = get_input_data(input, stream);
  auto& prepared = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(prepared, data, stream, pairs.data());

  std::vector<device_span<pair_t const>> spans;
  spans.reserve(groups);
  for (size_type group = 0; group < groups; ++group) {
    spans.emplace_back(pairs.data() + static_cast<std::size_t>(group) * input.size(), input.size());
  }
  auto columns = cudf::make_strings_column_batch(spans, stream, mr.get_output_mr());
  return std::make_unique<table>(std::move(columns));
}

std::unique_ptr<column> extract_single_impl(strings_column_view const& input,
                                            regex_jit_program const& prog,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  auto groups = prog.groups_count();
  CUDF_EXPECTS(groups > 0, "capture groups not found in regex pattern", std::invalid_argument);

  rmm::device_uvector<pair_t> pairs(input.size(), stream, mr.get_temporary_mr());
  auto data      = get_input_data(input, stream);
  auto& prepared = regex_jit_program_accessor::kernel(prog, data.offset64);
  launch(prepared, data, stream, pairs.data());
  return make_strings(pairs, stream, mr);
}

std::unique_ptr<column> enumerate_impl(strings_column_view const& input,
                                       regex_jit_program const& prog,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  auto capture_count = prog.groups_count();
  auto findall       = prog.operation() == regex_operation::FINDALL;
  auto groups        = prog.capture() == strings::capture_groups::EXTRACT ? capture_count : 0;
  if (findall) {
    CUDF_EXPECTS(groups <= 1, "findall does not support more than 1 capture group");
  } else {
    CUDF_EXPECTS(groups > 0, "extract_all requires group indicators in the regex pattern.");
  }
  if (input.is_empty()) { return make_empty_lists_column(data_type{type_id::STRING}); }

  auto counts     = make_numeric_column(data_type{type_id::INT32},
                                    input.size(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr.get_temporary_mr());
  auto validity   = make_numeric_column(data_type{type_id::BOOL8},
                                      input.size(),
                                      mask_state::UNALLOCATED,
                                      stream,
                                      mr.get_temporary_mr());
  auto data       = get_input_data(input, stream);
  auto cache_plan = select_span_cache(data, prog, stream, mr.get_temporary_mr());
  std::optional<rmm::device_uvector<std::int64_t>> cache;
  std::optional<rmm::device_uvector<std::uint8_t>> overflow;
  if (cache_plan) {
    auto values = regex_jit_program_accessor::cache_values(prog);
    cache.emplace(static_cast<std::size_t>(input.size()) * cache_plan.capacity * values,
                  stream,
                  mr.get_temporary_mr());
    overflow.emplace(input.size(), stream, mr.get_temporary_mr());
    CUDF_CUDA_TRY(cudaMemsetAsync(overflow->data(), 0, overflow->size(), stream.get()));
    auto& size_kernel = regex_jit_program_accessor::cache_size_kernel(prog, data.offset64);
    launch(size_kernel,
           data,
           stream,
           counts->mutable_view().head<void>(),
           validity->mutable_view().head<void>(),
           cache->data(),
           cache_plan.capacity,
           overflow->data());
  } else {
    auto& size_kernel = regex_jit_program_accessor::kernel(prog, data.offset64);
    launch(size_kernel,
           data,
           stream,
           counts->mutable_view().head<void>(),
           validity->mutable_view().head<void>());
  }

  auto [offsets, total] = cudf::detail::make_offsets_child_column(
    counts->view().begin<size_type>(), counts->view().end<size_type>(), stream, mr);
  rmm::device_uvector<pair_t> pairs(total, stream, mr.get_temporary_mr());
  if (total > 0) {
    if (cache_plan) {
      auto write_cached = [&](auto* input_offsets) {
        thrust::for_each_n(
          rmm::exec_policy_nosync(stream),
          thrust::make_counting_iterator<size_type>(0),
          input.size(),
          cached_enumeration_writer<std::remove_pointer_t<decltype(input_offsets)>>{
            data.chars,
            input_offsets,
            data.row_offset,
            cache->data(),
            overflow->data(),
            offsets->view().data<size_type>(),
            pairs.data(),
            cache_plan.capacity,
            regex_jit_program_accessor::cache_values(prog),
            groups,
            findall});
      };
      if (data.offset64) {
        write_cached(static_cast<std::int64_t const*>(data.offsets));
      } else {
        write_cached(static_cast<std::int32_t const*>(data.offsets));
      }
      auto& emit_kernel = regex_jit_program_accessor::cache_emit_kernel(prog, data.offset64);
      launch(emit_kernel,
             data,
             stream,
             pairs.data(),
             offsets->view().data<size_type>(),
             overflow->data());
    } else {
      auto& emit_kernel = regex_jit_program_accessor::kernel(prog, data.offset64, true);
      launch(emit_kernel, data, stream, pairs.data(), offsets->view().data<size_type>());
    }
  }
  auto strings_output = make_strings(pairs, stream, mr);

  rmm::device_buffer null_mask;
  size_type null_count;
  if (findall) {
    null_mask  = cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr());
    null_count = input.null_count();
  } else {
    auto converted = cudf::bools_to_mask(validity->view(), stream, mr.get_output_mr());
    null_mask      = std::move(*converted.first);
    null_count     = converted.second;
  }
  return make_lists_column(
    input.size(), std::move(offsets), std::move(strings_output), null_count, std::move(null_mask));
}

using replacement_piece = regex_ir::nvvm::replacement_piece;

std::vector<replacement_piece> literal_replacement(std::string_view replacement)
{
  return {{std::string{replacement}, std::nullopt}};
}

std::vector<replacement_piece> parse_backref_replacement(std::string_view replacement,
                                                         std::optional<size_type> group_count)
{
  CUDF_EXPECTS(!replacement.empty(), "Parameter replacement must not be empty");
  auto uses_backslash = false;
  for (std::size_t index = 0; index + 1 < replacement.size(); ++index) {
    if (replacement[index] == '\\' &&
        std::isdigit(static_cast<unsigned char>(replacement[index + 1])) != 0) {
      uses_backslash = true;
      break;
    }
  }

  std::vector<replacement_piece> result;
  std::string literal;
  auto flush_literal = [&] {
    if (!literal.empty()) {
      result.push_back({std::move(literal), std::nullopt});
      literal.clear();
    }
  };
  for (std::size_t position = 0; position < replacement.size();) {
    auto capture_begin = std::string_view::npos;
    auto capture_end   = std::string_view::npos;
    if (uses_backslash && replacement[position] == '\\' && position + 1 < replacement.size() &&
        std::isdigit(static_cast<unsigned char>(replacement[position + 1])) != 0) {
      capture_begin = position + 1;
      capture_end   = capture_begin;
      while (capture_end < replacement.size() &&
             std::isdigit(static_cast<unsigned char>(replacement[capture_end])) != 0) {
        ++capture_end;
      }
    } else if (!uses_backslash && replacement[position] == '$' &&
               position + 3 < replacement.size() && replacement[position + 1] == '{' &&
               std::isdigit(static_cast<unsigned char>(replacement[position + 2])) != 0) {
      capture_begin = position + 2;
      capture_end   = capture_begin;
      while (capture_end < replacement.size() &&
             std::isdigit(static_cast<unsigned char>(replacement[capture_end])) != 0) {
        ++capture_end;
      }
      if (capture_end == replacement.size() || replacement[capture_end] != '}') {
        capture_begin = std::string_view::npos;
      }
    }

    if (capture_begin == std::string_view::npos) {
      literal.push_back(replacement[position++]);
      continue;
    }

    std::uint64_t capture = 0;
    for (auto index = capture_begin; index < capture_end; ++index) {
      capture = capture * 10U + static_cast<std::uint64_t>(replacement[index] - '0');
    }
    auto max_group =
      group_count.has_value() ? std::min(*group_count, size_type{99}) : size_type{99};
    CUDF_EXPECTS(capture <= static_cast<std::uint64_t>(max_group),
                 "Group index numbers must be in the range 0 to group count");
    flush_literal();
    result.push_back({{}, static_cast<size_type>(capture)});
    position = uses_backslash ? capture_end : capture_end + 1;
  }
  flush_literal();
  return result;
}

std::unique_ptr<column> replace_impl(strings_column_view const& input,
                                     regex_jit_program const& prog,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  if (prog.pattern().empty()) {
    return std::make_unique<column>(input.parent(), stream, mr.get_output_mr());
  }
  auto sizes      = make_numeric_column(data_type{type_id::INT32},
                                   input.size(),
                                   mask_state::UNALLOCATED,
                                   stream,
                                   mr.get_temporary_mr());
  auto data       = get_input_data(input, stream);
  auto cache_plan = select_span_cache(data, prog, stream, mr.get_temporary_mr());
  std::optional<rmm::device_uvector<std::int64_t>> cache;
  std::optional<rmm::device_uvector<std::uint8_t>> overflow;
  std::optional<rmm::device_uvector<size_type>> match_counts;
  cudf::detail::device_scalar<std::int32_t> size_overflow(0, stream, mr.get_temporary_mr());
  if (cache_plan) {
    auto values = regex_jit_program_accessor::cache_values(prog);
    cache.emplace(static_cast<std::size_t>(input.size()) * cache_plan.capacity * values,
                  stream,
                  mr.get_temporary_mr());
    overflow.emplace(input.size(), stream, mr.get_temporary_mr());
    match_counts.emplace(input.size(), stream, mr.get_temporary_mr());
    CUDF_CUDA_TRY(cudaMemsetAsync(overflow->data(), 0, overflow->size(), stream.get()));
    auto& size_kernel = regex_jit_program_accessor::cache_size_kernel(prog, data.offset64);
    launch(size_kernel,
           data,
           stream,
           sizes->mutable_view().head<void>(),
           cache->data(),
           cache_plan.capacity,
           overflow->data(),
           match_counts->data(),
           size_overflow.data());
  } else {
    auto& size_kernel = regex_jit_program_accessor::kernel(prog, data.offset64);
    launch(size_kernel, data, stream, sizes->mutable_view().head<void>(), size_overflow.data());
  }
  CUDF_EXPECTS(size_overflow.value(stream) == 0,
               "Size of output string exceeds the column size limit",
               std::overflow_error);
  auto [offsets, total] = cudf::strings::detail::make_offsets_child_column(
    sizes->view().begin<size_type>(), sizes->view().end<size_type>(), stream, mr);
  rmm::device_buffer chars(total, stream, mr.get_output_mr());
  if (total > 0) {
    auto output_offset64 = offsets->type().id() == type_id::INT64;
    if (cache_plan) {
      auto& emit_kernel =
        regex_jit_program_accessor::cache_emit_kernel(prog, data.offset64, output_offset64);
      launch(emit_kernel,
             data,
             stream,
             chars.data(),
             offsets->view().head<void>(),
             cache->data(),
             cache_plan.capacity,
             overflow->data(),
             match_counts->data());
    } else {
      auto& emit_kernel =
        regex_jit_program_accessor::kernel(prog, data.offset64, true, output_offset64);
      launch(emit_kernel, data, stream, chars.data(), offsets->view().head<void>());
    }
  }
  return make_strings_column(
    input.size(),
    std::move(offsets),
    std::move(chars),
    input.null_count(),
    cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr()));
}

struct split_result {
  rmm::device_uvector<pair_t> pairs;
  std::unique_ptr<column> offsets;
  size_type columns;
};

struct split_pair_reader {
  pair_t const* pairs;
  size_type const* offsets;
  size_type column_index;

  __device__ pair_t operator()(size_type row) const
  {
    auto begin = offsets[row];
    auto count = offsets[row + 1] - begin;
    return column_index < count ? pairs[begin + column_index] : pair_t{nullptr, 0};
  }
};

split_result generate_split_pairs(strings_column_view const& input,
                                  regex_jit_program const& prog,
                                  bool offsets_are_output,
                                  cuda::stream_ref stream,
                                  cudf::memory_resources mr)
{
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  auto reverse = prog.operation() == regex_operation::RSPLIT ||
                 prog.operation() == regex_operation::RSPLIT_RECORD;
  auto maxsplit = regex_jit_program_accessor::options(prog).maxsplit;
  auto reverse_limited =
    reverse && maxsplit > 0 && maxsplit < std::numeric_limits<size_type>::max();

  auto counts     = make_numeric_column(data_type{type_id::INT32},
                                    input.size(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr.get_temporary_mr());
  auto data       = get_input_data(input, stream);
  auto cache_plan = select_span_cache(data, prog, stream, mr.get_temporary_mr());
  std::optional<rmm::device_uvector<std::int64_t>> cache;
  std::optional<rmm::device_uvector<std::uint8_t>> overflow;
  if (cache_plan) {
    cache.emplace(static_cast<std::size_t>(input.size()) * (cache_plan.capacity + 1) * 2,
                  stream,
                  mr.get_temporary_mr());
    overflow.emplace(input.size(), stream, mr.get_temporary_mr());
    CUDF_CUDA_TRY(cudaMemsetAsync(overflow->data(), 0, overflow->size(), stream.get()));
    auto& size_kernel = regex_jit_program_accessor::cache_size_kernel(prog, data.offset64);
    launch(size_kernel,
           data,
           stream,
           counts->mutable_view().head<void>(),
           cache->data(),
           cache_plan.capacity,
           overflow->data());
  } else {
    auto& size_kernel = regex_jit_program_accessor::kernel(prog, data.offset64);
    launch(size_kernel, data, stream, counts->mutable_view().head<void>());
  }

  auto maximum = cudf::reduce(counts->view(),
                              *make_max_aggregation<reduce_aggregation>(),
                              data_type{type_id::INT32},
                              stream,
                              mr.get_temporary_mr());
  auto columns = static_cast<numeric_scalar<size_type> const&>(*maximum).value(stream);
  if (reverse_limited) { columns = std::min(columns, maxsplit + 1); }
  columns = std::max(columns, size_type{1});

  auto effective_count =
    cuda::proclaim_return_type<size_type>([reverse_limited, maxsplit] __device__(size_type count) {
      return reverse_limited ? cuda::std::min(count, maxsplit + 1) : count;
    });
  auto effective_begin =
    cuda::transform_iterator(counts->view().begin<size_type>(), effective_count);
  auto effective_end = effective_begin + input.size();
  auto offsets_mr =
    offsets_are_output ? mr : cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()};
  auto [effective_offsets, effective_total] =
    cudf::detail::make_offsets_child_column(effective_begin, effective_end, stream, offsets_mr);

  std::unique_ptr<column> full_offsets;
  auto full_total = effective_total;
  if (reverse_limited) {
    std::tie(full_offsets, full_total) = cudf::detail::make_offsets_child_column(
      counts->view().begin<size_type>(),
      counts->view().end<size_type>(),
      stream,
      cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()});
  }

  rmm::device_uvector<pair_t> pairs(effective_total, stream, mr.get_temporary_mr());
  rmm::device_uvector<std::int64_t> spans(
    static_cast<std::size_t>(full_total) * 2, stream, mr.get_temporary_mr());
  if (effective_total > 0) {
    auto effective_data = effective_offsets->view().data<size_type>();
    auto full_data      = reverse_limited ? full_offsets->view().data<size_type>() : effective_data;
    if (cache_plan) {
      auto write_cached = [&](auto* input_offsets) {
        thrust::for_each_n(
          rmm::exec_policy_nosync(stream),
          thrust::make_counting_iterator<size_type>(0),
          input.size(),
          cached_split_writer<std::remove_pointer_t<decltype(input_offsets)>>{data.chars,
                                                                              input_offsets,
                                                                              data.row_offset,
                                                                              cache->data(),
                                                                              overflow->data(),
                                                                              effective_data,
                                                                              full_data,
                                                                              pairs.data(),
                                                                              cache_plan.capacity,
                                                                              reverse});
      };
      if (data.offset64) {
        write_cached(static_cast<std::int64_t const*>(data.offsets));
      } else {
        write_cached(static_cast<std::int32_t const*>(data.offsets));
      }
      auto& emit_kernel = regex_jit_program_accessor::cache_emit_kernel(prog, data.offset64);
      launch(emit_kernel,
             data,
             stream,
             pairs.data(),
             effective_data,
             full_data,
             spans.data(),
             overflow->data());
    } else {
      auto& emit_kernel = regex_jit_program_accessor::kernel(prog, data.offset64, true);
      launch(emit_kernel, data, stream, pairs.data(), effective_data, full_data, spans.data());
    }
  }
  return {std::move(pairs), std::move(effective_offsets), columns};
}

std::unique_ptr<column> split_record_impl(strings_column_view const& input,
                                          regex_jit_program const& prog,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  if (input.is_empty()) { return make_empty_lists_column(data_type{type_id::STRING}); }

  auto result         = generate_split_pairs(input, prog, true, stream, mr);
  auto strings_output = make_strings(result.pairs, stream, mr);
  return make_lists_column(input.size(),
                           std::move(result.offsets),
                           std::move(strings_output),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr.get_output_mr()));
}

std::unique_ptr<table> split_table_impl(strings_column_view const& input,
                                        regex_jit_program const& prog,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  if (input.is_empty()) {
    std::vector<std::unique_ptr<column>> columns;
    columns.emplace_back(make_empty_column(type_id::STRING));
    return std::make_unique<table>(std::move(columns));
  }

  auto result       = generate_split_pairs(input, prog, false, stream, mr);
  auto offsets_data = result.offsets->view().data<size_type>();
  std::vector<std::unique_ptr<column>> columns;
  columns.reserve(result.columns);
  for (size_type index = 0; index < result.columns; ++index) {
    auto begin = cudf::detail::make_counting_transform_iterator(
      0, split_pair_reader{result.pairs.data(), offsets_data, index});
    columns.emplace_back(cudf::strings::detail::make_strings_column(
      begin, begin + input.size(), stream, mr.get_output_mr()));
  }
  return std::make_unique<table>(std::move(columns));
}
}  // namespace

retained_kernel const& regex_jit_program_accessor::kernel(regex_jit_program const& prog,
                                                          bool offset64,
                                                          bool emit,
                                                          bool output_offset64)
{
  auto index =
    emit && (prog.operation() == regex_operation::REPLACE ||
             prog.operation() == regex_operation::REPLACE_WITH_BACKREFS)
      ? 2U + static_cast<std::size_t>(offset64) * 2U + static_cast<std::size_t>(output_offset64)
      : static_cast<std::size_t>(offset64) + (emit ? 2U : 0U);
  return prog._impl->kernels.at(index);
}

std::unique_ptr<regex_jit_program> regex_jit_program::create(
  std::string_view pattern,
  regex_operation operation,
  regex_jit_program_options const& program_options,
  strings::regex_flags flags,
  strings::capture_groups captures)
{
  return std::unique_ptr<regex_jit_program>(
    new regex_jit_program(pattern, operation, program_options, flags, captures));
}

regex_jit_program::regex_jit_program(std::string_view pattern,
                                     regex_operation operation,
                                     regex_jit_program_options const& program_options,
                                     strings::regex_flags flags,
                                     strings::capture_groups captures)
  : _pattern{pattern},
    _operation{operation},
    _flags{flags},
    _captures{captures},
    _options{program_options}
{
  try {
    auto compile_pattern = normalize_pattern(pattern);
    if (operation == regex_operation::MATCHES) {
      compile_pattern = std::format(R"(\A(?:{}))", compile_pattern);
    }

    auto compile_options = make_compile_options(flags);
    auto internal        = internal_operation(operation);
    if (operation == regex_operation::FINDALL &&
        captures == strings::capture_groups::NON_CAPTURE) {
      internal = regex_ir::operation_kind::FIND_ALL;
    }
    auto replacement_operation =
      operation == regex_operation::REPLACE || operation == regex_operation::REPLACE_WITH_BACKREFS;
    auto native_replace   = replacement_operation && !program_options.max_replace_count.has_value();
    auto replacement_text = program_options.replacement.value_or("");
    if (operation == regex_operation::REPLACE_WITH_BACKREFS) {
      CUDF_EXPECTS(program_options.replacement.has_value(),
                   "replace_with_backrefs requires a replacement in regex_jit_program_options");
    }
    auto replacement      = literal_replacement(replacement_text);
    auto validation_error = std::optional<std::string>{};
    if (native_replace && operation == regex_operation::REPLACE_WITH_BACKREFS) {
      try {
        replacement = parse_backref_replacement(replacement_text, std::nullopt);
      } catch (cudf::logic_error const& error) {
        validation_error = error.what();
        replacement      = literal_replacement("");
        native_replace   = false;
      }
    }

    auto compiled = [&] {
      if (!native_replace) {
        return regex_ir::compile(compile_pattern, internal, std::nullopt, compile_options);
      }
      try {
        return regex_ir::compile(compile_pattern,
                                 regex_ir::operation_kind::REPLACE,
                                 regex_ir::nvvm::encode_replacement(replacement),
                                 compile_options);
      } catch (std::invalid_argument const& error) {
        if (operation != regex_operation::REPLACE_WITH_BACKREFS) { throw; }
        auto fallback = regex_ir::compile(compile_pattern, internal, std::nullopt, compile_options);
        validation_error = error.what();
        replacement      = literal_replacement("");
        native_replace   = false;
        return fallback;
      }
    }();
    auto capture_count = static_cast<size_type>(compiled.capture_count);
    if (!native_replace && operation == regex_operation::REPLACE_WITH_BACKREFS &&
        !validation_error.has_value()) {
      try {
        replacement = parse_backref_replacement(replacement_text, capture_count);
      } catch (cudf::logic_error const& error) {
        validation_error = error.what();
        replacement      = literal_replacement("");
      }
    }
    auto executor             = compiled.executor;
    auto exact_ascii_literal  = std::move(compiled.exact_ascii_literal);
    auto matcher              = std::move(compiled.nvvm_ir);
    auto kernels              = std::vector<retained_kernel>{};
    auto warp_literal_kernels = std::vector<retained_kernel>{};
    auto cache_size_kernels   = std::vector<retained_kernel>{};
    auto cache_emit_kernels   = std::vector<retained_kernel>{};
    auto sample_kernels       = std::vector<retained_kernel>{};
    auto cache_values         = size_type{0};
    auto kernel_operation = replacement_operation ? regex_ir::operation_kind::REPLACE : internal;

    auto add_pass = [&](auto&& make_wrapper, std::string_view name) {
      for (auto offset64 : {false, true}) {
        kernels.push_back(
          compile_kernel(matcher, kernel_operation, executor, make_wrapper(offset64), name));
      }
    };
    auto add_output_offset_pass = [&](auto&& make_wrapper, std::string_view name) {
      for (auto offset64 : {false, true}) {
        for (auto output_offset64 : {false, true}) {
          kernels.push_back(compile_kernel(
            matcher, kernel_operation, executor, make_wrapper(offset64, output_offset64), name));
        }
      }
    };
    ensure_stack_size();
    switch (operation) {
      case regex_operation::CONTAINS:
      case regex_operation::MATCHES:
      case regex_operation::COUNT:
      case regex_operation::FIND:
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_fixed_kernel(offset64, internal, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.fixed");
        if (operation == regex_operation::CONTAINS) {
          if (exact_ascii_literal.has_value()) {
            for (auto offset64 : {false, true}) {
              auto prepared    = compile_kernel(matcher,
                                             internal,
                                             executor,
                                             regex_ir::nvvm::make_warp_literal_contains_kernel(
                                               offset64, *exact_ascii_literal, KERNEL_ENTRY),
                                             "cudf.experimental.regex.warp_literal_contains");
              prepared.threads = 256;
              warp_literal_kernels.push_back(std::move(prepared));
            }
          }
        }
        break;
      case regex_operation::EXTRACT:
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_capture_kernel(
              offset64, (capture_count + 1) * 2, 0, capture_count, true, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.extract");
        break;
      case regex_operation::EXTRACT_SINGLE: {
        CUDF_EXPECTS(program_options.group.has_value(),
                     "extract_single requires a group in regex_jit_program_options");
        auto group = *program_options.group;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_capture_kernel(
              offset64, (capture_count + 1) * 2, group, 1, false, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.extract_single");
        break;
      }
      case regex_operation::EXTRACT_ALL_RECORD:
      case regex_operation::FINDALL: {
        auto findall = operation == regex_operation::FINDALL;
        auto groups  = captures == strings::capture_groups::EXTRACT ? capture_count : 0;
        auto slots   = (capture_count + 1) * 2;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_enumeration_size_kernel(
              offset64, slots, findall ? 1 : groups, !findall, false, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.enumerate_size");
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_enumeration_emit_kernel(
              offset64, slots, groups, findall, false, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.enumerate_emit");
        if (program_options.span_cache_policy != regex_jit_span_cache_policy::OFF) {
          cache_values = slots;
          for (auto offset64 : {false, true}) {
            cache_size_kernels.push_back(
              compile_kernel(matcher,
                             kernel_operation,
                             executor,
                             regex_ir::nvvm::make_enumeration_size_kernel(
                               offset64, slots, findall ? 1 : groups, !findall, true, KERNEL_ENTRY),
                             "cudf.experimental.regex.enumerate_cache_size"));
            cache_emit_kernels.push_back(
              compile_kernel(matcher,
                             kernel_operation,
                             executor,
                             regex_ir::nvvm::make_enumeration_emit_kernel(
                               offset64, slots, groups, findall, true, KERNEL_ENTRY),
                             "cudf.experimental.regex.enumerate_overflow_emit"));
            sample_kernels.push_back(compile_kernel(matcher,
                                                    kernel_operation,
                                                    executor,
                                                    regex_ir::nvvm::make_span_cache_sample_kernel(
                                                      offset64, false, slots, -1, KERNEL_ENTRY),
                                                    "cudf.experimental.regex.enumerate_sample"));
          }
        }
        break;
      }
      case regex_operation::REPLACE:
      case regex_operation::REPLACE_WITH_BACKREFS: {
        if (native_replace) {
          add_pass(
            [&](bool offset64) {
              return regex_ir::nvvm::make_replace_kernel(offset64, false, false, KERNEL_ENTRY);
            },
            "cudf.experimental.regex.replace_size");
          add_output_offset_pass(
            [&](bool offset64, bool output_offset64) {
              return regex_ir::nvvm::make_replace_kernel(
                offset64, true, output_offset64, KERNEL_ENTRY);
            },
            "cudf.experimental.regex.replace_emit");
          break;
        }
        auto limit =
          program_options.max_replace_count.has_value() && *program_options.max_replace_count > 0
            ? *program_options.max_replace_count
          : program_options.max_replace_count == 0 ? 0
                                                   : std::numeric_limits<size_type>::max();
        auto slots = (capture_count + 1) * 2;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_limited_replace_kernel(
              offset64, false, false, false, replacement, slots, limit, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.replace_size");
        add_output_offset_pass(
          [&](bool offset64, bool output_offset64) {
            return regex_ir::nvvm::make_limited_replace_kernel(
              offset64, true, output_offset64, false, replacement, slots, limit, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.replace_emit");
        if (program_options.span_cache_policy != regex_jit_span_cache_policy::OFF) {
          cache_values = slots;
          for (auto offset64 : {false, true}) {
            cache_size_kernels.push_back(compile_kernel(
              matcher,
              kernel_operation,
              executor,
              regex_ir::nvvm::make_limited_replace_kernel(
                offset64, false, false, true, replacement, slots, limit, KERNEL_ENTRY),
              "cudf.experimental.regex.replace_cache_size"));
            sample_kernels.push_back(compile_kernel(matcher,
                                                    kernel_operation,
                                                    executor,
                                                    regex_ir::nvvm::make_span_cache_sample_kernel(
                                                      offset64, false, slots, limit, KERNEL_ENTRY),
                                                    "cudf.experimental.regex.replace_sample"));
            for (auto output_offset64 : {false, true}) {
              cache_emit_kernels.push_back(compile_kernel(
                matcher,
                kernel_operation,
                executor,
                regex_ir::nvvm::make_limited_replace_kernel(
                  offset64, true, output_offset64, true, replacement, slots, limit, KERNEL_ENTRY),
                "cudf.experimental.regex.replace_cache_emit"));
            }
          }
        }
        break;
      }
      case regex_operation::SPLIT:
      case regex_operation::RSPLIT:
      case regex_operation::SPLIT_RECORD:
      case regex_operation::RSPLIT_RECORD: {
        auto reverse =
          operation == regex_operation::RSPLIT || operation == regex_operation::RSPLIT_RECORD;
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_split_size_kernel(
              offset64, reverse ? -1 : program_options.maxsplit, false, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.split_size");
        add_pass(
          [&](bool offset64) {
            return regex_ir::nvvm::make_split_emit_kernel(
              offset64, reverse, program_options.maxsplit, false, KERNEL_ENTRY);
          },
          "cudf.experimental.regex.split_emit");
        if (program_options.span_cache_policy != regex_jit_span_cache_policy::OFF) {
          cache_values = 2;
          for (auto offset64 : {false, true}) {
            cache_size_kernels.push_back(compile_kernel(
              matcher,
              kernel_operation,
              executor,
              regex_ir::nvvm::make_split_size_kernel(
                offset64, reverse ? -1 : program_options.maxsplit, true, KERNEL_ENTRY),
              "cudf.experimental.regex.split_cache_size"));
            cache_emit_kernels.push_back(
              compile_kernel(matcher,
                             kernel_operation,
                             executor,
                             regex_ir::nvvm::make_split_emit_kernel(
                               offset64, reverse, program_options.maxsplit, true, KERNEL_ENTRY),
                             "cudf.experimental.regex.split_overflow_emit"));
            sample_kernels.push_back(compile_kernel(
              matcher,
              kernel_operation,
              executor,
              regex_ir::nvvm::make_span_cache_sample_kernel(
                offset64, true, 2, reverse ? -1 : program_options.maxsplit, KERNEL_ENTRY),
              "cudf.experimental.regex.split_sample"));
          }
        }
        break;
      }
    }
    _impl = std::make_unique<regex_jit_program_impl>(
      regex_jit_program_impl{capture_count,
                             std::move(kernels),
                             std::move(cache_size_kernels),
                             std::move(cache_emit_kernels),
                             std::move(sample_kernels),
                             std::move(warp_literal_kernels),
                             executor,
                             cache_values,
                             std::move(validation_error)});
  } catch (std::invalid_argument const& error) {
    CUDF_FAIL(error.what());
  }
}

regex_jit_program::~regex_jit_program()                                       = default;
regex_jit_program::regex_jit_program(regex_jit_program&& other) noexcept      = default;
regex_jit_program& regex_jit_program::operator=(regex_jit_program&&) noexcept = default;

std::string regex_jit_program::pattern() const { return _pattern; }

strings::regex_flags regex_jit_program::flags() const { return _flags; }

strings::capture_groups regex_jit_program::capture() const { return _captures; }

regex_operation regex_jit_program::operation() const { return _operation; }

size_type regex_jit_program::groups_count() const { return _impl->capture_count; }

std::unique_ptr<column> contains_re(strings_column_view const& input,
                                    regex_jit_program const& prog,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::CONTAINS, "contains_re");
  return fixed_result(input, prog, data_type{type_id::BOOL8}, stream, mr);
}

std::unique_ptr<column> matches_re(strings_column_view const& input,
                                   regex_jit_program const& prog,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::MATCHES, "matches_re");
  return fixed_result(input, prog, data_type{type_id::BOOL8}, stream, mr);
}

std::unique_ptr<column> count_re(strings_column_view const& input,
                                 regex_jit_program const& prog,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::COUNT, "count_re");
  return fixed_result(input, prog, data_type{type_id::INT32}, stream, mr);
}

std::unique_ptr<column> find_re(strings_column_view const& input,
                                regex_jit_program const& prog,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::FIND, "find_re");
  return fixed_result(input, prog, data_type{type_id::INT32}, stream, mr);
}

std::unique_ptr<table> extract(strings_column_view const& input,
                               regex_jit_program const& prog,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::EXTRACT, "extract");
  return extract_impl(input, prog, stream, mr);
}

std::unique_ptr<column> extract_single(strings_column_view const& input,
                                       regex_jit_program const& prog,
                                       size_type group,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::EXTRACT_SINGLE, "extract_single");
  auto& options = regex_jit_program_accessor::options(prog);
  CUDF_EXPECTS(options.group == group,
               "extract_single group does not match the regex_jit_program group");
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS(group >= 0 && group < prog.groups_count(),
               "group parameter outside the range of capture groups found in the regex pattern",
               std::invalid_argument);
  return extract_single_impl(input, prog, stream, mr);
}

std::unique_ptr<column> extract_all_record(strings_column_view const& input,
                                           regex_jit_program const& prog,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::EXTRACT_ALL_RECORD, "extract_all_record");
  return enumerate_impl(input, prog, stream, mr);
}

std::unique_ptr<column> findall(strings_column_view const& input,
                                regex_jit_program const& prog,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::FINDALL, "findall");
  return enumerate_impl(input, prog, stream, mr);
}

std::unique_ptr<column> replace_re(strings_column_view const& input,
                                   regex_jit_program const& prog,
                                   string_scalar const& replacement,
                                   std::optional<size_type> max_replace_count,
                                   cuda::stream_ref stream,
                                   cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::REPLACE, "replace_re");
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  auto& options = regex_jit_program_accessor::options(prog);
  CUDF_EXPECTS(options.replacement.value_or("") == replacement.to_string(stream),
               "replace_re replacement does not match the regex_jit_program replacement");
  CUDF_EXPECTS(options.max_replace_count == max_replace_count,
               "replace_re max_replace_count does not match the regex_jit_program limit");
  return replace_impl(input, prog, stream, mr);
}

std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                              regex_jit_program const& prog,
                                              std::string_view replacement,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::REPLACE_WITH_BACKREFS, "replace_with_backrefs");
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");
  auto& options = regex_jit_program_accessor::options(prog);
  CUDF_EXPECTS(
    options.replacement == replacement,
    "replace_with_backrefs replacement does not match the regex_jit_program replacement");
  auto& validation_error = regex_jit_program_accessor::validation_error(prog);
  CUDF_EXPECTS(!validation_error.has_value(), validation_error.value_or(""));
  return replace_impl(input, prog, stream, mr);
}

std::unique_ptr<table> split_re(strings_column_view const& input,
                                regex_jit_program const& prog,
                                size_type maxsplit,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::SPLIT, "split_re");
  expect_split_parameters(prog, maxsplit, "split_re");
  return split_table_impl(input, prog, stream, mr);
}

std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                 regex_jit_program const& prog,
                                 size_type maxsplit,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::RSPLIT, "rsplit_re");
  expect_split_parameters(prog, maxsplit, "rsplit_re");
  return split_table_impl(input, prog, stream, mr);
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        regex_jit_program const& prog,
                                        size_type maxsplit,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::SPLIT_RECORD, "split_record_re");
  expect_split_parameters(prog, maxsplit, "split_record_re");
  return split_record_impl(input, prog, stream, mr);
}

std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                         regex_jit_program const& prog,
                                         size_type maxsplit,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
{
  expect_operation(prog, regex_operation::RSPLIT_RECORD, "rsplit_record_re");
  expect_split_parameters(prog, maxsplit, "rsplit_record_re");
  return split_record_impl(input, prog, stream, mr);
}

}  // namespace cudf::experimental
